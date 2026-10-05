"""The ledger must carry ``task_id`` onto rows written by an OLD database.

``verification_events`` is created with ``CREATE TABLE IF NOT EXISTS``, so editing
the DDL only describes the column for a FRESH ledger. An existing database
matches the IF NOT EXISTS and every later statement is skipped — its columns stay
frozen at whatever shipped first. The insert then fails with "no column named
task_id", and because ``record_terminal_result`` is called under ``_quiet()``
(tools/terminal_tool_result.py) the failure is swallowed to DEBUG: the worker
keeps running and simply stops recording evidence. A green run that recorded
nothing.

So the column needs a migration path, and these tests pin the migration itself:
they build a schema-v1 ledger by hand (no fixture of the current DDL, which
would pass even with no migration at all) and require that recording into it
adds the column and stores the value.
"""

import sqlite3

import pytest

from agent import verification_evidence as ve


@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    """The ledger is inert unless verify-on-stop is enabled; these tests exercise the ledger."""
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")


# The columns that shipped with schema_version=1, in their original physical order.
_SCHEMA_V1_COLUMNS = (
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "created_at TEXT NOT NULL",
    "session_id TEXT NOT NULL",
    "cwd TEXT NOT NULL",
    "root TEXT NOT NULL",
    "command TEXT NOT NULL",
    "canonical_command TEXT NOT NULL",
    "kind TEXT NOT NULL",
    "scope TEXT NOT NULL",
    "status TEXT NOT NULL",
    "exit_code INTEGER NOT NULL",
    "output_summary TEXT NOT NULL",
)


def _legacy_db(tmp_path):
    """A hand-built schema-v1 ledger: no task_id, and schema_version 1."""
    db = tmp_path / "verification_evidence.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE verification_events (" + ", ".join(_SCHEMA_V1_COLUMNS) + ")"
    )
    conn.execute(
        "CREATE TABLE verification_state ("
        "session_id TEXT NOT NULL, root TEXT NOT NULL, last_event_id INTEGER,"
        " last_edit_at TEXT, changed_paths_json TEXT NOT NULL DEFAULT '[]',"
        " PRIMARY KEY (session_id, root))"
    )
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()
    return db


def _columns(db):
    conn = sqlite3.connect(db)
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(verification_events)")]
    finally:
        conn.close()


def _event_row(db, event_id):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return dict(conn.execute(
            "SELECT * FROM verification_events WHERE id = ?", (event_id,)
        ).fetchone())
    finally:
        conn.close()


def _python_project(root):
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")


def test_recording_into_a_legacy_ledger_migrates_the_schema(monkeypatch, tmp_path):
    """The regression itself: an old ledger must not make recording fail silently."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_legacy01")
    db = _legacy_db(tmp_path)
    monkeypatch.setattr(ve, "_db_path", lambda: db)
    _python_project(tmp_path)
    assert "task_id" not in _columns(db)

    recorded = ve.record_terminal_result(
        command="python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path, session_id="s1", exit_code=0, output="1 passed",
    )

    assert recorded is not None, "recording into a legacy ledger returned nothing"
    assert "task_id" in _columns(db)
    assert _event_row(db, recorded["id"])["task_id"] == "t_legacy01"


def test_migration_is_appended_after_the_existing_columns(monkeypatch, tmp_path):
    """ALTER TABLE appends, so the physical order of v1 columns must not shift.

    verification_status does ``SELECT *`` and reads by name, so a moved column
    would not break it — but the original column order is part of what the
    migration promises not to disturb.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(ve, "_db_path", lambda: _legacy_db(tmp_path))
    _python_project(tmp_path)

    ve.record_verify_run(root=tmp_path, session_id="s1", ok=True)

    columns = _columns(tmp_path / "verification_evidence.db")
    assert columns[:-1] == [ddl.split()[0] for ddl in _SCHEMA_V1_COLUMNS]
    assert columns[-1] == "task_id"


def test_migration_is_idempotent_across_repeated_recordings(monkeypatch, tmp_path):
    """Re-running the initializer must not fail or duplicate the column."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(ve, "_db_path", lambda: _legacy_db(tmp_path))
    _python_project(tmp_path)

    ve._ensure_schema(sqlite3.connect(tmp_path / "verification_evidence.db"))
    ve._ensure_schema(sqlite3.connect(tmp_path / "verification_evidence.db"))

    assert _columns(tmp_path / "verification_evidence.db").count("task_id") == 1


def test_migrated_ledger_is_not_re_altered_on_every_open(monkeypatch, tmp_path):
    """Once the column exists, opening the ledger must not attempt DDL again.

    ``add_column_if_missing`` swallows the duplicate-column error, so dropping
    the PRAGMA guard still produces a correct schema — but it turns every ledger
    open into an attempted schema write, which takes a schema lock and contends
    with other processes holding the WAL. The guard keeps the hot path
    read-only, so it is pinned here rather than left to review.

    Asserted by counting the ALTER attempts rather than by tracing SQL:
    sqlite3 does not invoke the trace callback for a statement that raises, so
    a failed duplicate-column ALTER would be invisible to a trace assertion.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(ve, "_db_path", lambda: _legacy_db(tmp_path))
    _python_project(tmp_path)
    ve.record_terminal_result(
        command="python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path, session_id="s1", exit_code=0, output="1 passed",
    )

    conn = sqlite3.connect(tmp_path / "verification_evidence.db")
    attempts: list[str] = []

    class _AlterCountingConnection:
        """Delegates to a real connection, recording ALTER attempts.

        sqlite3.Connection is a static C type: ``execute`` cannot be
        monkeypatched per instance, so the counting happens in a wrapper handed
        in its place (the fd-leak test tracks ``close()`` the same way).
        """

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE" in str(sql):
                attempts.append(str(sql))
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    counting = _AlterCountingConnection(conn)
    try:
        ve._ensure_schema(counting)
    finally:
        conn.close()

    assert attempts == []


def test_fresh_ledger_records_task_id_and_leaves_it_null_without_one(
    monkeypatch, tmp_path,
):
    """A brand-new ledger still gets the column, and a non-worker records NULL."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = tmp_path / "verification_evidence.db"
    monkeypatch.setattr(ve, "_db_path", lambda: db)
    _python_project(tmp_path)

    recorded = ve.record_terminal_result(
        command="python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path, session_id="s1", exit_code=0, output="1 passed",
    )

    assert recorded is not None
    assert "task_id" in _columns(db)
    assert _event_row(db, recorded["id"])["task_id"] is None