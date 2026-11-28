"""Der neue shm-Identitaetscheck darf legitime Zugriffe nicht ablehnen.

Geprueft werden die Faelle, die im Betrieb vorkommen und NICHT als
Fremdbibliothek im eigenen Prozess gelten:

  * ein anderer PROZESS oeffnet und schliesst denselben Store (erlaubt: die
    POSIX-Locks sind pro Prozess, der Kernel serialisiert);
  * ein anderer Prozess macht einen WAL-Checkpoint;
  * die Engine schliesst und oeffnet denselben Store neu;
  * viele Schreibvorgaenge mit zwischenzeitlichem Lesen (kein Wechsel der
    Dateiidentitaet).
"""

import subprocess
import sys

import yantrikdb

DIM = 8
VEC = [1.0] + [0.0] * (DIM - 1)


def _record(db, text):
    return db.record(
        text, "semantic", 0.5, 0.0, 86400.0, {}, VEC,
        "default", 0.8, "general", "user", None,
    )


def _visible(path, rid):
    """Read the row in a SEPARATE process.

    Reading with Python's sqlite3 in this process would be the very hazard the
    guard exists for (#225: two SQLite libraries, one store, one process) — the
    first version of this test died with a bus error proving it.
    """
    code = (
        "import sqlite3,sys;"
        "c=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True);"
        "print(c.execute('SELECT COUNT(*) FROM memories WHERE rid=?',(sys.argv[2],)).fetchone()[0])"
    )
    out = subprocess.run([sys.executable, "-c", code, path, rid],
                         check=True, capture_output=True, text=True, timeout=60)
    return out.stdout.strip() == "1"


def _other_process(path, code):
    """Run a snippet in a SEPARATE process (its own SQLite library instance)."""
    subprocess.run([sys.executable, "-c", code, path], check=True, timeout=60)


def test_another_process_opening_and_closing_does_not_refuse(tmp_path):
    path = str(tmp_path / "store.db")
    db = yantrikdb.YantrikDB(path, DIM)
    _record(db, "warm")

    _other_process(path, "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);"
                         "c.execute(\"INSERT OR REPLACE INTO meta (key,value) VALUES ('p','x')\");"
                         "c.commit();c.close()")

    rid = _record(db, "after another process")
    assert _visible(path, rid)


def test_another_process_checkpoint_does_not_refuse(tmp_path):
    path = str(tmp_path / "store.db")
    db = yantrikdb.YantrikDB(path, DIM)
    for i in range(5):
        _record(db, f"warm {i}")

    _other_process(path, "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);"
                         "c.execute('PRAGMA wal_checkpoint(TRUNCATE)');c.close()")

    rid = _record(db, "after a foreign checkpoint")
    assert _visible(path, rid)


def test_reopening_the_engine_does_not_refuse(tmp_path):
    path = str(tmp_path / "store.db")
    db = yantrikdb.YantrikDB(path, DIM)
    _record(db, "before close")
    del db

    db2 = yantrikdb.YantrikDB(path, DIM)
    rid = _record(db2, "after reopen")
    assert _visible(path, rid)


def test_many_writes_do_not_trip_the_identity_check(tmp_path):
    path = str(tmp_path / "store.db")
    db = yantrikdb.YantrikDB(path, DIM)
    for i in range(50):
        rid = _record(db, f"row {i}")
        assert _visible(path, rid)
