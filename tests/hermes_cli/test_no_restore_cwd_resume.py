"""Regression for --in/--no-restore-cwd across CLI history hydration paths."""

import contextlib
import os
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from hermes_cli._parser import build_top_level_parser
from hermes_cli import main as main_mod
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.cli_session_mixin import CLISessionMixin


class _ResumeHarness(CLIAgentSetupMixin, CLISessionMixin):
    """Call real resume methods against a real SQLite store, without agent/model startup."""

    def __init__(self, db, sid, no_restore_cwd):
        self._session_db = db
        self.session_id = sid
        self._resumed = True
        self.no_restore_cwd = no_restore_cwd
        self.conversation_history = []
        self._resume_history_error = None
        self.tool_progress_mode = "off"
        self._console_print = lambda *args, **kwargs: None
        self._restore_session_yolo = lambda *args, **kwargs: None
        self._restore_session_model = lambda *args, **kwargs: None


@pytest.mark.parametrize("load_method", ["_preload_resumed_session", "_load_resumed_history_late"])
@pytest.mark.parametrize("no_restore", [True, False])
def test_resume_workspace_after_history_load(tmp_path, monkeypatch, load_method, no_restore):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    sid = "cwd_resume_fixture"
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(sid, source="cli", cwd=str(a))
        db.append_message(sid, "user", "fixture question")
        db.append_message(sid, "assistant", "fixture response")
        monkeypatch.chdir(b)
        monkeypatch.setenv("TERMINAL_CWD", str(b))
        flags = ["chat", "--resume", sid]
        if no_restore:
            flags.extend(["--in", str(b), "--no-restore-cwd"])
        # --in itself pins the workspace; the historical inverse omits both flags.
        args = build_top_level_parser()[0].parse_args(flags)
        with patch.object(main_mod, "_session_db", lambda: contextlib.nullcontext(db)), \
             patch.object(main_mod, "_resolve_session_by_name_or_id", lambda value: value):
            main_mod._resolve_chat_session_args(args, use_tui=False)
        assert os.getcwd() == str(b if no_restore else a)
        assert args.no_restore_cwd is no_restore
        assert getattr(_ResumeHarness(db, sid, args.no_restore_cwd), load_method)()
        assert os.getcwd() == str(b if no_restore else a)
        assert os.environ["TERMINAL_CWD"] == str(b if no_restore else a)


def test_cmd_chat_passes_no_restore_cwd_to_classic_cli(monkeypatch, tmp_path):
    import cli

    b = tmp_path / "B"
    b.mkdir()
    monkeypatch.chdir(b)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda value: value)
    monkeypatch.setattr(main_mod, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_consent.offer_consent_before_chat", lambda args: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_process.begin_process", lambda *args: None)
    seen = {}
    monkeypatch.setattr(cli, "main", lambda **kwargs: seen.update(kwargs))
    args = build_top_level_parser()[0].parse_args(["chat", "--cli", "--resume", "fixture", "--in", str(b), "--no-restore-cwd"])
    main_mod.cmd_chat(args)
    assert seen["no_restore_cwd"] is True


def test_classic_main_passes_no_restore_cwd_to_builder(monkeypatch):
    import cli

    seen = {}
    monkeypatch.setattr("hermes_cli.process_identity.register_self", lambda *args: None)
    monkeypatch.setattr("hermes_cli.shared_profile_warning.shared_profile_warning", lambda: None)
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *args: None)
    monkeypatch.setattr(cli, "_install_single_query_signal_handlers", lambda *args: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *args: None)
    class FakeCLI:
        def run(self):
            return None
    def fake_build(*args, **kwargs):
        seen.update(kwargs)
        return FakeCLI()
    monkeypatch.setattr(cli, "_build_cli_from_args", fake_build)
    cli.main(resume="fixture", no_restore_cwd=True)
    assert seen["no_restore_cwd"] is True


def test_classic_cli_stores_no_restore_cwd(monkeypatch):
    import cli

    monkeypatch.setattr(cli.HermesCLI, "_init_display_options", lambda *args: None)
    monkeypatch.setattr(cli.HermesCLI, "_init_model_routing", lambda *args: None)
    monkeypatch.setattr(cli.HermesCLI, "_init_runtime_state", lambda *args: None)
    assert cli.HermesCLI(no_restore_cwd=True).no_restore_cwd is True
    assert cli.HermesCLI().no_restore_cwd is False


def test_classic_cli_receives_no_restore_cwd(monkeypatch):
    import cli

    seen = {}
    class FakeCLI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.ignore_rules = True
            self.session_id = "fixture"

    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr("agent.skill_commands.resolve_auto_load_skills", lambda config: [])
    cli._build_cli_from_args(None, None, None, None, None, None, None, None, None,
                             False, "fixture", False, False, True, None,
                             no_restore_cwd=True)
    assert seen["no_restore_cwd"] is True
