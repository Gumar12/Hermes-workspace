"""``meta.schema_version`` must not go backwards when an older process shares the ledger.

Every process that opens the ledger runs ``_ensure_schema``, and it ends by
stamping meta with ITS OWN module constant:

    INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)

A worker that imported the module before the task_id migration carries
``_VERIFY_SCHEMA_VERSION = 1``. Once the migration has run once, that worker's
next open writes meta=1 over a value of 2 — so the marker describes a schema one
column behind the one the database actually has.

Measured on the live builder ledger (read through a copy, never the original):
meta read 1 while ``task_id`` was already present, and running the
pre-migration initializer over a ledger stamped 2 changed it back to 1.

Pinning ``_VERIFY_SCHEMA_VERSION`` to an older value stands in for that older
process: the code under test is the real initializer, not a re-implementation of
its SQL, so the test fails if the regression comes back for any reason.
"""

import sqlite3
from pathlib import Path

import pytest

from agent import verification_evidence as ve


@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")


def _stamped_db(tmp_path: Path, *, version: str, migrated: bool) -> Path:
    """A ledger carrying `version` in meta, with the v2 column only if migrated."""
    db = tmp_path / f"v{version}.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE verification_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,"
        " session_id TEXT NOT NULL, cwd TEXT NOT NULL, root TEXT NOT NULL,"
        " command TEXT NOT NULL, canonical_command TEXT NOT NULL, kind TEXT NOT NULL,"
        " scope TEXT NOT NULL, status TEXT NOT NULL, exit_code INTEGER NOT NULL,"
        " output_summary TEXT NOT NULL" + (", task_id TEXT" if migrated else "") + ")"
    )
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (version,))
    conn.commit()
    conn.close()
    return db


def _meta_version(db) -> str:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    finally:
        conn.close()


def _open(db):
    return sqlite3.connect(db)


def test_an_older_initializer_does_not_regress_the_schema_marker(monkeypatch, tmp_path):
    """A process holding the pre-migration constant must not rewrite meta to 1.

    The ledger is already migrated and stamped 2 — exactly the state this card
    leaves behind. An older worker opening it finds nothing to migrate and
    should leave the marker alone.
    """
    db = _stamped_db(tmp_path, version="2", migrated=True)
    monkeypatch.setattr(ve, "_VERIFY_SCHEMA_VERSION", 1)

    ve._ensure_schema(_open(db))

    assert _meta_version(db) == "2", (
        "an older process rewrote meta=1 over a ledger already at 2: the marker "
        "no longer describes the schema the database actually has"
    )


def test_the_marker_still_advances_on_a_legacy_ledger(monkeypatch, tmp_path):
    """The guard must not freeze a real v1 ledger at 1 — it still bumps to 2."""
    db = _stamped_db(tmp_path, version="1", migrated=False)
    monkeypatch.setattr(ve, "_db_path", lambda: db)

    ve._ensure_schema(_open(db))

    assert _meta_version(db) == "2"
    assert "task_id" in [r[1] for r in _open(db).execute("PRAGMA table_info(verification_events)")]


def test_reopening_an_already_migrated_ledger_keeps_the_marker(monkeypatch, tmp_path):
    """The normal path twice over: still 2, no drift."""
    db = _stamped_db(tmp_path, version="1", migrated=False)
    monkeypatch.setattr(ve, "_db_path", lambda: db)

    ve._ensure_schema(_open(db))
    ve._ensure_schema(_open(db))

    assert _meta_version(db) == "2"