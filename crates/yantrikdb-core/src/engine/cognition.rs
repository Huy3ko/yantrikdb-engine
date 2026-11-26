use rusqlite::params;

use crate::error::Result;
use crate::types::*;

use super::{now, YantrikDB};

/// Upper bound on memories scanned per `think()` for attribute-value claim
/// extraction (v0.7.23 prototype). Bounded so the opt-in pass stays O(1) per
/// call on large stores; recent memories are processed first.
const ATTR_CLAIM_SCAN_LIMIT: usize = 500;

impl YantrikDB {
    // ── Cognition loop (V3) ──

    /// **v0.7.23 prototype.** Extract "<subject> is <value>" assertions from
    /// recent active memories and write them into the claim layer via
    /// [`Self::ingest_claim`], so the existing
    /// [`crate::conflict::scan_claim_conflicts`] detector fires on attribute-
    /// value updates made through the free-text `record()` path.
    ///
    /// Idempotent: `ingest_claim`'s `ON CONFLICT(src,dst,rel_type,extractor,
    /// polarity,namespace)` scoping dedups identical re-extractions, so
    /// re-running across `think()` calls does not multiply claims. Returns the
    /// number of `(subject, value)` claims ingested this pass. Best-effort per
    /// memory: a single malformed claim is skipped, not propagated.
    pub(crate) fn ingest_attribute_claims(&self, max_memories: usize) -> Result<usize> {
        // Snapshot (rid, text, namespace) under the conn lock, then release it
        // BEFORE calling ingest_claim (which acquires the conn lock itself).
        let rows: Vec<(String, String, String)> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT rid, text, namespace FROM memories \
                 WHERE consolidation_status = 'active' \
                 ORDER BY created_at DESC LIMIT ?1",
            )?;
            let mapped = stmt.query_map(params![max_memories as i64], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut ingested = 0usize;
        for (rid, text, namespace) in rows {
            for claim in crate::cognition::attribute_claims::extract_attribute_value_claims(&text) {
                match self.ingest_claim(
                    &claim.subject,
                    &claim.rel,
                    &claim.value,
                    &namespace,
                    1,          // polarity: positive assertion
                    "asserted", // modality
                    None,       // valid_from
                    None,       // valid_to
                    "attr_value_v1",
                    Some("0.1.0"),
                    "medium",   // confidence_band
                    Some(&rid), // provenance
                    None,       // span_start
                    None,       // span_end
                    1.0,        // weight
                ) {
                    Ok(_) => ingested += 1,
                    Err(e) => tracing::warn!(
                        rid = %rid, error = %e,
                        "ingest_attribute_claims: skipped a claim"
                    ),
                }
            }
        }
        Ok(ingested)
    }

    /// Run the full cognition loop: trigger detection, conflict scanning,
    /// consolidation, and pattern mining. Returns a prioritized list of triggers
    /// and summary of actions taken.
    #[tracing::instrument(skip(self, config))]
    pub fn think(&self, config: &ThinkConfig) -> Result<ThinkResult> {
        let start = crate::time::Instant::now();
        let ts = now();

        // Operating diagnostics for THIS pass. A failing phase is recorded
        // here and the pass continues: one damaged index must not cost the
        // user the other six phases, and a call that returns Err tells the
        // caller nothing about what DID run. See `MaintenanceDiagnostic`.
        let mut diagnostics: Vec<MaintenanceDiagnostic> = Vec::new();

        // Phase -1 (#95): materialize what the caller just wrote, BEFORE
        // anything reads it.
        //
        // Every phase below reads derived state — the conflict scans read
        // `claims`, consolidation reads `memory_entities`, the trigger checks
        // read `entities`. None of that exists at `record()` time any more:
        // the foreground write enqueues `OP_MATERIALIZE_RECORD_POST` and the
        // extraction runs on the background materializer pool's 100 ms idle
        // timer. So `record(); record(); think()` was a race, and the caller
        // lost it by default — the documented contradiction example returned
        // `conflicts_found: 0` immediately and `1` after a `sleep`, on
        // identical input. Draining here makes think() read a store that
        // contains the writes that preceded it, which is the contract every
        // caller already assumed it had.
        //
        // FIRST STATEMENT ON PURPOSE, and it must stay first. The drain
        // dispatches into materializers that take the conn lock, which is not
        // reentrant — running it after any phase that holds a guard would
        // deadlock. Nothing above this line takes one.
        //
        // Bounded and best-effort: `drain_materializer_backlog` caps the work
        // (see its docs for the backlog tradeoff), and a drain FAILURE must
        // not abort cognition. A store that cannot materialize is exactly a
        // store that still needs its triggers expired and its existing
        // conflicts scanned — degrading to the old racy behaviour beats
        // returning an error from a maintenance call.
        match self.drain_materializer_backlog() {
            Ok(0) => {}
            Ok(n) => tracing::debug!(applied = n, "think: drained materializer backlog"),
            Err(e) => {
                diagnostics.push(MaintenanceDiagnostic::failed(
                    "materializer_drain",
                    None,
                    &e,
                ));
                tracing::warn!(
                    error = %e,
                    "think: materializer drain failed; scanning possibly-stale derived state"
                )
            }
        }

        // #142: `conflicts_found` used to sum only the two Phase-2 scans, but
        // Phase 1's `check_redundancy` ALSO persists conflicts (learned
        // substitution categories, via create_conflict) and returns only its
        // triggers — so a real, stored conflict reported as 0. Counting rows
        // written across the whole cycle makes the number mean "what think()
        // actually persisted", and stays correct when a fourth writer appears.
        let conflicts_before: i64 = if config.run_conflict_scan {
            match self
                .conn()
                .query_row("SELECT COUNT(*) FROM conflicts", [], |r| r.get(0))
            {
                Ok(n) => n,
                Err(e) => {
                    // Without the baseline the delta below is meaningless, and
                    // `conflicts_found: 0` would read as "none exist".
                    diagnostics.push(MaintenanceDiagnostic::failed("conflict_count", None, &e));
                    -1
                }
            }
        } else {
            0
        };

        // Phase 0: Expire old triggers. Degrades: an unusable trigger table
        // must not stop the conflict scan.
        let expired = match crate::triggers::expire_triggers(self, ts) {
            Ok(n) => n,
            Err(e) => {
                diagnostics.push(MaintenanceDiagnostic::failed("trigger_expiry", None, &e));
                0
            }
        };

        // Phase 1: Run all trigger checks. Each check is independent — a
        // missing index used by one of them is a reason to record a
        // diagnostic, not to abandon the other seven.
        let mut all_triggers = Vec::new();
        let mut run = |phase: &str, r: Result<Vec<Trigger>>| match r {
            Ok(mut v) => all_triggers.append(&mut v),
            Err(e) => diagnostics.push(MaintenanceDiagnostic::failed(phase, None, &e)),
        };
        run(
            "decay_check",
            crate::triggers::check_decay_triggers(
                self,
                config.importance_threshold,
                config.decay_threshold,
                config.max_triggers,
            ),
        );
        run(
            "consolidation_check",
            crate::triggers::check_consolidation_triggers(self, config.min_active_memories),
        );
        run(
            "conflict_escalation",
            crate::triggers::check_conflict_escalation(self),
        );
        run(
            "temporal_drift",
            crate::triggers::check_temporal_drift(self),
        );
        run(
            "relationship_insight",
            crate::triggers::check_relationship_insight(self),
        );
        run("valence_trend", crate::triggers::check_valence_trend(self));
        run(
            "entity_anomaly",
            crate::triggers::check_entity_anomaly(self),
        );
        drop(run);

        // The redundancy scan reports namespaces it could not evaluate through
        // the same diagnostics vector, so it runs after the closure that
        // borrows it is dropped.
        match crate::triggers::check_redundancy_with_diagnostics(
            self,
            config.redundancy_sim_threshold,
            &mut diagnostics,
        ) {
            Ok(mut v) => all_triggers.append(&mut v),
            Err(e) => diagnostics.push(MaintenanceDiagnostic::failed("redundancy_scan", None, &e)),
        }

        // Phase 1.9 (v0.7.23 prototype): bridge free-text attribute-value
        // facts into the claim layer so the conflict scan below can see
        // "<subject> is <value>" updates made via plain record(). Opt-in and
        // skipped under encryption (stored text is ciphertext). Best-effort:
        // a failure here must not abort the cognition loop.
        if config.extract_attribute_claims && config.run_conflict_scan && !self.is_encrypted() {
            if let Err(e) = self.ingest_attribute_claims(ATTR_CLAIM_SCAN_LIMIT) {
                diagnostics.push(MaintenanceDiagnostic::failed("attribute_claims", None, &e));
                tracing::warn!(error = %e, "attribute-claim extraction pass failed");
            }
        }

        // Phase 2: Scan for conflicts BEFORE consolidation
        // (so contradictions are flagged before similar memories get merged)
        // Uses consolidation_limit to cap work per call (incremental processing).
        let conflicts_found = if config.run_conflict_scan {
            if let Err(e) =
                crate::conflict::scan_conflicts_limited(self, config.consolidation_limit)
            {
                diagnostics.push(MaintenanceDiagnostic::failed("conflict_scan", None, &e));
            }
            // RFC 006 Phase 1: also scan claim-based conflicts (scoped, with severity + reason codes)
            if let Err(e) = crate::conflict::scan_claim_conflicts(self, config.consolidation_limit)
            {
                diagnostics.push(MaintenanceDiagnostic::failed(
                    "claim_conflict_scan",
                    None,
                    &e,
                ));
            }
            // Delta over the whole cycle rather than the sum of these two scans'
            // return values — see the note at `conflicts_before`. Saturating
            // because consolidation can tombstone rows, and a negative "found"
            // would be worse than an undercount.
            let after: i64 =
                match self
                    .conn()
                    .query_row("SELECT COUNT(*) FROM conflicts", [], |r| r.get(0))
                {
                    Ok(n) => n,
                    Err(e) => {
                        diagnostics.push(MaintenanceDiagnostic::failed("conflict_count", None, &e));
                        conflicts_before
                    }
                };
            if conflicts_before < 0 || after < 0 {
                // A count that could not be taken is not a zero.
                0
            } else {
                (after - conflicts_before).max(0) as usize
            }
        } else {
            0
        };

        // Phase 2.5: substitution categories ready for gossip expansion.
        // F3c: this joins `substitution_members` too, so a damaged index made
        // it abort the WHOLE pass via `?` — the one phase whose degradation
        // was supposed to be handled locally.
        match self.check_gossip_triggers() {
            Ok(mut v) => all_triggers.append(&mut v),
            Err(e) => diagnostics.push(MaintenanceDiagnostic::failed("gossip_scan", None, &e)),
        }

        // Phase 3: Run consolidation if configured
        let consolidation_count = if config.run_consolidation {
            match self.stats(None) {
                Ok(stats) if stats.active_memories >= config.min_active_memories => {
                    match crate::consolidate::consolidate(
                        self,
                        config.consolidation_sim_threshold,
                        config.consolidation_time_window_days,
                        config.consolidation_min_cluster,
                        config.consolidation_limit,
                        config.consolidation_require_entity_overlap,
                        false,
                    ) {
                        Ok(results) => results.len(),
                        Err(e) => {
                            diagnostics.push(MaintenanceDiagnostic::failed(
                                "consolidation",
                                None,
                                &e,
                            ));
                            0
                        }
                    }
                }
                Ok(_) => 0,
                Err(e) => {
                    diagnostics.push(MaintenanceDiagnostic::failed(
                        "consolidation_stats",
                        None,
                        &e,
                    ));
                    0
                }
            }
        } else {
            0
        };

        // Phase 4: Mine patterns
        let pattern_result = if config.run_pattern_mining {
            match crate::patterns::mine_patterns(self, &PatternConfig::default()) {
                Ok(r) => r,
                Err(e) => {
                    diagnostics.push(MaintenanceDiagnostic::failed("pattern_mining", None, &e));
                    PatternMiningResult {
                        new_patterns: 0,
                        updated_patterns: 0,
                        stale_patterns: 0,
                    }
                }
            }
        } else {
            PatternMiningResult {
                new_patterns: 0,
                updated_patterns: 0,
                stale_patterns: 0,
            }
        };

        // Phase 4.5: Adaptive learning — the v0.10 Item 2 self-sufficient
        // loop. Typed report (persisted to meta.last_learning_report by
        // run_learning itself) so a learner FAILURE is observable rather
        // than silently identical to "not enough evidence".
        match self.run_learning() {
            Ok(report) if report.outcome == "swapped" || report.outcome == "rolled_back" => {
                all_triggers.push(Trigger {
                    trigger_type: "weights_updated".to_string(),
                    reason: format!(
                        "Ranking weights {} (generation {}): {}",
                        report.outcome, report.generation, report.detail
                    ),
                    urgency: 0.3,
                    source_rids: vec![],
                    suggested_action: "acknowledge".to_string(),
                    context: std::collections::HashMap::new(),
                });
            }
            Ok(_) => {} // abstained (insufficient/no-new evidence) or rejected
            Err(e) => {
                diagnostics.push(MaintenanceDiagnostic::failed("learning_loop", None, &e));
                tracing::warn!(error = %e, "learning loop failed");
            }
        }

        // Phase 5: Generate pattern_discovered triggers for new patterns
        // Cross-domain and entity_bridge patterns get higher urgency ("surprise" triggers)
        if pattern_result.new_patterns > 0 {
            let new_patterns = match crate::patterns::get_patterns(self, None, Some("active"), 10) {
                Ok(v) => v,
                Err(e) => {
                    diagnostics.push(MaintenanceDiagnostic::failed("pattern_readback", None, &e));
                    Vec::new()
                }
            };
            for p in new_patterns {
                let mut context = std::collections::HashMap::new();
                context.insert(
                    "pattern_type".to_string(),
                    serde_json::json!(p.pattern_type),
                );
                context.insert("confidence".to_string(), serde_json::json!(p.confidence));
                context.insert("description".to_string(), serde_json::json!(p.description));

                // Cross-domain patterns are "surprise" discoveries — boost urgency
                let (urgency, trigger_type, action) = match p.pattern_type.as_str() {
                    "cross_domain" => {
                        // Extract domain info from context for richer trigger
                        if let Some(ctx) = p.context.as_object() {
                            if let (Some(da), Some(db)) = (
                                ctx.get("domain_a").and_then(|v| v.as_str()),
                                ctx.get("domain_b").and_then(|v| v.as_str()),
                            ) {
                                context.insert("domain_a".to_string(), serde_json::json!(da));
                                context.insert("domain_b".to_string(), serde_json::json!(db));
                            }
                        }
                        (
                            (p.confidence * 0.8).max(0.5),
                            "surprise_connection",
                            "explore_cross_domain",
                        )
                    }
                    "entity_bridge" => {
                        // Entity bridges reveal hidden connectors between domains
                        (
                            (p.confidence * 0.7).max(0.4),
                            "entity_bridge_discovered",
                            "explore_entity_bridge",
                        )
                    }
                    _ => (p.confidence * 0.5, "pattern_discovered", "explore_pattern"),
                };

                // Add session context if available
                let active_sessions = self.active_sessions.read();
                if !active_sessions.is_empty() {
                    let session_ids: Vec<&String> = active_sessions.values().collect();
                    context.insert(
                        "active_sessions".to_string(),
                        serde_json::json!(session_ids),
                    );
                }

                all_triggers.push(Trigger {
                    trigger_type: trigger_type.to_string(),
                    reason: format!("New pattern discovered: {}", p.description),
                    urgency,
                    source_rids: p.evidence_rids,
                    suggested_action: action.to_string(),
                    context,
                });
            }
        }

        // Phase 5.5: Session awareness triggers
        // Check for gaps between sessions and surface context from last session
        {
            let conn = self.conn();
            // Degrades: the sessions table is not part of the substitution
            // index, but a prepare failure here must not cost the phases that
            // come after it.
            let phase = (|| -> Result<Vec<Trigger>> {
                let mut out = Vec::new();
                let mut session_stmt = conn.prepare(
                    "SELECT session_id, client_id, ended_at, summary, avg_valence, memory_count, topics \
                     FROM sessions WHERE status = 'ended' \
                     ORDER BY ended_at DESC LIMIT 1",
                )?;
                if let Ok((
                    session_id,
                    client_id,
                    ended_at,
                    summary,
                    avg_valence,
                    memory_count,
                    topics_json,
                )) = session_stmt.query_row([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<f64>>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                }) {
                    let gap_hours = (ts - ended_at) / 3600.0;
                    let topics: Vec<String> =
                        serde_json::from_str(&topics_json).unwrap_or_default();

                    // Generate session awareness trigger if there's a meaningful gap
                    if gap_hours > 4.0 {
                        let mut context = std::collections::HashMap::new();
                        context
                            .insert("last_session_id".to_string(), serde_json::json!(session_id));
                        context.insert("client_id".to_string(), serde_json::json!(client_id));
                        context.insert("gap_hours".to_string(), serde_json::json!(gap_hours));
                        context.insert(
                            "last_session_memory_count".to_string(),
                            serde_json::json!(memory_count),
                        );
                        context
                            .insert("last_session_topics".to_string(), serde_json::json!(topics));
                        if let Some(ref s) = summary {
                            context
                                .insert("last_session_summary".to_string(), serde_json::json!(s));
                        }
                        if let Some(v) = avg_valence {
                            context
                                .insert("last_session_valence".to_string(), serde_json::json!(v));
                        }

                        let urgency = if gap_hours > 72.0 {
                            0.7
                        } else if gap_hours > 24.0 {
                            0.5
                        } else {
                            0.3
                        };
                        let reason = if gap_hours > 24.0 {
                            format!(
                            "It's been {:.0} hours since your last session. Last time: {} memories stored{}.",
                            gap_hours,
                            memory_count,
                            summary.as_ref().map(|s| format!(" — {}", s)).unwrap_or_default(),
                        )
                        } else {
                            format!(
                                "Welcome back ({:.0}h gap). Last session: {} memories{}.",
                                gap_hours,
                                memory_count,
                                if !topics.is_empty() {
                                    format!(
                                        " about {}",
                                        topics
                                            .iter()
                                            .take(3)
                                            .cloned()
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    )
                                } else {
                                    String::new()
                                },
                            )
                        };

                        out.push(Trigger {
                            trigger_type: "session_awareness".to_string(),
                            reason,
                            urgency,
                            source_rids: vec![],
                            suggested_action: "acknowledge".to_string(),
                            context,
                        });
                    }
                }
                Ok(out)
            })();
            match phase {
                Ok(mut v) => all_triggers.append(&mut v),
                Err(e) => {
                    diagnostics.push(MaintenanceDiagnostic::failed("session_awareness", None, &e))
                }
            }
        }

        // Phase 6: Deduplicate, cooldown-filter, persist triggers.
        //
        // This one RAISES even in degrade mode: if the triggers cannot be
        // stored, the pass produced its main output nowhere, and a caller that
        // received `triggers: []` with no error would believe it was clean.
        let filtered = crate::triggers::filter_and_persist_triggers(self, all_triggers, ts)?;

        // Phase 7: Sort by urgency, truncate
        let mut final_triggers = filtered;
        final_triggers.sort_by(|a, b| b.urgency.total_cmp(&a.urgency));
        final_triggers.truncate(config.max_triggers);

        // Phase 7: Derive personality traits. Degrades with a diagnostic:
        // `personality_updated: false` alone cannot be told apart from
        // "nothing to update".
        let personality_updated = if config.run_personality {
            match crate::personality::derive_personality(self) {
                Ok(_) => true,
                Err(e) => {
                    diagnostics.push(MaintenanceDiagnostic::failed("personality", None, &e));
                    false
                }
            }
        } else {
            false
        };

        // Record last_think_at — and settle the maintenance-debt ledger when
        // this pass actually LOOKED at the accumulated writes. The conflict
        // scan (entity + claim) is the pass that reads new content, so only
        // a think with run_conflict_scan clears `writes_since_think`; a
        // think called with the scan disabled still stamps last_think_at
        // (pre-existing behavior, unchanged) but leaves the write debt
        // standing — nothing examined those writes, so the ledger must keep
        // saying so.
        {
            let conn = self.conn();
            let ledger = if config.run_conflict_scan {
                Self::clear_maintenance_debt_on(&conn, ts)
            } else {
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_think_at', ?1)",
                    params![ts.to_string()],
                )
                .map(|_| ())
                .map_err(Into::into)
            };
            // Raises even in degrade mode: the ledger is how the store knows
            // which writes have been examined. A pass that did not settle it
            // has not done its job, and silence would make the debt permanent.
            ledger?;
        }

        // Log think op (informational, not materialized on remote)
        if let Err(e) = self.log_op(
            "think",
            None,
            &serde_json::json!({
                "triggers_count": final_triggers.len(),
                "consolidation_count": consolidation_count,
                "conflicts_found": conflicts_found,
                "new_patterns": pattern_result.new_patterns,
                "diagnostics": diagnostics.len(),
            }),
            None,
        ) {
            diagnostics.push(MaintenanceDiagnostic::failed("op_log", None, &e));
        }

        let duration_ms = start.elapsed_ms();

        let status = if diagnostics.is_empty() {
            "ok".to_string()
        } else {
            "degraded".to_string()
        };
        // `strict` restores the pre-diagnostics contract for callers that
        // cannot act on a partial result: a degraded pass is an error.
        if config.strict {
            if let Some(first) = diagnostics.first() {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "maintenance degraded in {}{}: {} (strict mode)",
                    first.phase,
                    first
                        .namespace
                        .as_deref()
                        .map(|n| format!("@{n}"))
                        .unwrap_or_default(),
                    first.cause
                )));
            }
        }

        Ok(ThinkResult {
            triggers: final_triggers,
            consolidation_count,
            conflicts_found,
            patterns_new: pattern_result.new_patterns,
            patterns_updated: pattern_result.updated_patterns,
            expired_triggers: expired,
            personality_updated,
            duration_ms,
            status,
            diagnostics,
        })
    }

    // ── Gossip expansion triggers (V14) ──

    /// Check substitution categories that are ready for LLM gossip expansion.
    ///
    /// Conditions per category:
    /// - At least 3 confirmed members (source in seed/user_confirmed)
    /// - No gossip trigger for this category in last 7 days
    fn check_gossip_triggers(&self) -> Result<Vec<Trigger>> {
        let mut triggers = Vec::new();
        let conn = self.conn();

        // Find categories with enough confirmed members AND at least one non-seed member
        // (seed-only categories don't need gossip expansion until the user has interacted)
        let mut stmt = conn.prepare(
            "SELECT c.id, c.name,
                    COUNT(*) as confirmed_count,
                    GROUP_CONCAT(m.token_display, ', ') as member_list
             FROM substitution_categories c
             JOIN substitution_members m ON m.category_id = c.id
             WHERE c.status IN ('active', 'provisional')
               AND m.status = 'active'
               AND m.source IN ('seed', 'user_confirmed')
             GROUP BY c.id
             HAVING confirmed_count >= 3
               AND SUM(CASE WHEN m.source != 'seed' THEN 1 ELSE 0 END) >= 1",
        )?;

        let candidates: Vec<(String, String, i64, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let cooldown_secs = 7.0 * 86400.0; // 7 days

        for (cat_id, cat_name, _count, member_list) in candidates {
            let cooldown_key = format!("gossip_{}", cat_name);

            // Check cooldown
            let recent: bool = conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM trigger_log
                 WHERE cooldown_key = ?1 AND created_at > ?2",
                    params![cooldown_key, crate::time::now_secs() - cooldown_secs],
                    |row| row.get(0),
                )
                .unwrap_or(false);

            if recent {
                continue;
            }

            let members: Vec<String> = member_list.split(", ").map(|s| s.to_string()).collect();

            let mut context = std::collections::HashMap::new();
            context.insert("category_id".to_string(), serde_json::json!(cat_id));
            context.insert("category_name".to_string(), serde_json::json!(cat_name));
            context.insert("confirmed_members".to_string(), serde_json::json!(members));

            triggers.push(Trigger {
                trigger_type: "gossip_expand_category".to_string(),
                urgency: 0.3,
                reason: format!(
                    "Category '{}' has {} confirmed members and is ready for vocabulary expansion",
                    cat_name,
                    members.len()
                ),
                suggested_action: format!(
                    "Ask LLM to suggest additional members for the '{}' category",
                    cat_name
                ),
                source_rids: vec![],
                context,
            });
        }

        Ok(triggers)
    }

    // ── Trigger lifecycle API ──

    /// Mark a trigger as delivered (surfaced to host app).
    pub fn deliver_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'delivered', delivered_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'pending'",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_deliver",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "delivered_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Mark a trigger as acknowledged (user saw it).
    pub fn acknowledge_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'acknowledged', acknowledged_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'delivered'",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_ack",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "acknowledged_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Mark a trigger as acted upon.
    pub fn act_on_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'acted', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_act",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "acted_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Dismiss a trigger (user doesn't want to act on it).
    pub fn dismiss_trigger(&self, trigger_id: &str) -> Result<bool> {
        let ts = now();
        let changes = self.conn().execute(
            "UPDATE trigger_log SET status = 'dismissed', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('pending', 'delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
        if changes > 0 {
            self.log_op(
                "trigger_dismiss",
                Some(trigger_id),
                &serde_json::json!({"trigger_id": trigger_id, "dismissed_at": ts}),
                None,
            )?;
        }
        Ok(changes > 0)
    }

    /// Get pending triggers sorted by urgency.
    pub fn get_pending_triggers(&self, limit: usize) -> Result<Vec<PersistedTrigger>> {
        crate::triggers::get_pending_triggers(self, limit)
    }

    /// Get trigger history with optional type filter.
    pub fn get_trigger_history(
        &self,
        trigger_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<PersistedTrigger>> {
        crate::triggers::get_trigger_history(self, trigger_type, limit)
    }

    // ── Personality API (V11) ──

    /// Get the current personality profile without recomputing.
    pub fn get_personality(&self) -> Result<crate::types::PersonalityProfile> {
        crate::personality::get_personality(self)
    }

    /// Force recompute personality traits from memory signals.
    pub fn derive_personality(&self) -> Result<crate::types::PersonalityProfile> {
        crate::personality::derive_personality(self)
    }

    /// Manually override a personality trait score (for testing).
    pub fn set_personality_trait(&self, name: &str, score: f64) -> Result<bool> {
        crate::personality::set_personality_trait(self, name, score)
    }

    /// Get detected patterns.
    pub fn get_patterns(
        &self,
        pattern_type: Option<&str>,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Pattern>> {
        crate::patterns::get_patterns(self, pattern_type, status, limit)
    }
}
