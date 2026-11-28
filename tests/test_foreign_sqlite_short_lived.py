"""Regressionstest: kurzlebige Fremdverbindung darf keinen Schreibverlust erzeugen.

Der Wächter sucht in /proc/self/maps nach einem zweiten Mapping der `-shm`.
Eine Verbindung, die zwischen zwei Scans öffnet und schließt, hinterlässt kein
Mapping — ihr Close löscht aber die `-shm` (sie hält sich für den letzten
Nutzer, weil POSIX-Locks prozessweit gelten). Der nächste Engine-Commit landet
dann in einem WAL, den niemand koordiniert: `record()` gab eine rid zurück und
die Zeile war nie sichtbar.

Der Test ist diskriminierend: ohne Fix schlägt er mit genau diesem Symptom fehl;
mit Fix wird der Schreibvorgang mit `ForeignSqliteInstance` verweigert.
"""

import sqlite3

import pytest

import yantrikdb

DIM = 8
VEC = [1.0] + [0.0] * (DIM - 1)


def _record(db, text):
    return db.record(
        text, "semantic", 0.5, 0.0, 86400.0, {}, VEC,
        "default", 0.8, "general", "user", None,
    )


def test_short_lived_foreign_connection_does_not_lose_a_write(tmp_path):
    path = tmp_path / "store.db"
    db = yantrikdb.YantrikDB(str(path), DIM)
    # Warm up: the store must be in WAL mode with a live -shm.
    _record(db, "warm")

    # A second SQLite library (Python's) opens, commits and closes — all
    # between two detector scans, so /proc/self/maps never shows it.
    con = sqlite3.connect(str(path))
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('probe', 'x')")
    con.commit()
    con.close()

    try:
        rid = _record(db, "after the foreign close")
    except Exception as exc:  # noqa: BLE001 — the refusal is the wanted outcome
        msg = f"{type(exc).__name__}: {exc}"
        assert (
            "ForeignSqliteInstance" in msg or "SQLite library" in msg
        ), f"refused, but not by the #225 guard: {msg}"
        return

    # No refusal: the engine claims the write succeeded, so the row must exist.
    con = sqlite3.connect(str(path))
    try:
        rows = con.execute("SELECT COUNT(*) FROM memories WHERE rid = ?", (rid,)).fetchone()[0]
    finally:
        con.close()
    assert rows == 1, (
        "record() returned a rid but wrote nothing: a short-lived foreign "
        "connection in this process silently voided the write (issue #225)"
    )
