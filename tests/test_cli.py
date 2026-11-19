"""hermes super-agent status."""

import argparse
import json

from helpers import end_turn, spawn_worker, start_turn


def _run(ctx, *argv):
    entry = ctx.cli_commands["super-agent"]
    parser = argparse.ArgumentParser(prog="hermes super-agent")
    entry["setup_fn"](parser)
    parser.set_defaults(func=entry["handler_fn"])
    args = parser.parse_args(list(argv))
    return args.func(args)


def test_status_lists_settings_and_tasks(plugin, make_ctx, capsys, monkeypatch):
    monkeypatch.setenv("SUPER_AGENT_TURN_SOFT_SECONDS", "45")
    ctx = make_ctx()
    start_turn(plugin)
    spawn_worker(plugin)
    spawn_worker(plugin, goal="Rotate the Cloudflare token", sub_id="sa-1-def", child_session="child-2")
    plugin._subagent_stop(parent_session_id="s1", child_session_id="child-2", child_status="completed",
                          child_summary=json.dumps({"status": "needs_decision", "question": "Which zone?"}),
                          tool_call_history=[], duration_ms=1000)
    end_turn(plugin)

    assert _run(ctx, "status") == 0
    out = capsys.readouterr().out
    assert out.startswith(f"super-agent {plugin.__version__}")
    assert "ledger.db" in out and "turn_soft_seconds" in out and "45.0   (changed)" in out
    assert "worker_repeat_limit" in out and "Host check" in out
    lines = out.splitlines()
    first = next(i for i, line in enumerate(lines) if "running" in line and "Fix the 522" in line)
    second = next(i for i, line in enumerate(lines) if "needs_decision" in line)
    assert first < second, "running tasks come first"
    assert "asks: Which zone?" in out and "chat s1" in out


def test_status_json_and_default_command(plugin, make_ctx, capsys):
    ctx = make_ctx()
    start_turn(plugin)
    spawn_worker(plugin)
    assert _run(ctx, "status", "--json", "--limit", "1") == 0
    info = json.loads(capsys.readouterr().out)
    assert info["version"] == plugin.__version__ and info["ledger"].endswith("ledger.db")
    assert [t["id"] for t in info["tasks"]] == ["sa-0-abc"]
    assert info["settings"]["governed_platforms"] == {"value": ["weixin"], "default": ["weixin"]}

    assert _run(ctx) == 0, "no subcommand means status"
    assert "Recent subagent tasks" in capsys.readouterr().out


def test_status_with_an_empty_ledger(plugin, make_ctx, capsys):
    ctx = make_ctx()
    assert _run(ctx, "status") == 0
    assert "No subagent tasks recorded yet." in capsys.readouterr().out


def test_register_without_cli_support(plugin):
    class Bare:
        def get_config(self, key, default=None):
            return default

        def register_hook(self, name, fn, **kwargs):
            pass

        def register_middleware(self, kind, fn):
            pass
    plugin.register(Bare())


def _gateway_ledger(plugin, tmp_path, monkeypatch):
    """A ledger the gateway wrote, with a task it still runs (in a process this one cannot see)."""
    import sqlite3
    path = tmp_path / "gateway" / "ledger.db"
    path.parent.mkdir()
    plugin.ledger.load(path)
    plugin.ledger.start("sa-7-live", "s1", "Rebuild the index", "child-7")
    plugin.ledger._update("sa-7-live", lambda t: t.update(pid=0, proc=""))
    plugin.ledger._db.close()
    plugin.ledger._db = None
    plugin._reset_all()
    monkeypatch.setattr(plugin.ledger, "_default_path", lambda: path)
    monkeypatch.setattr(plugin.ledger, "_open_default", lambda: sqlite3.connect(str(path)))
    return path


def test_status_reads_the_ledger_without_changing_it(plugin, make_ctx, capsys, tmp_path, monkeypatch):
    path = _gateway_ledger(plugin, tmp_path, monkeypatch)
    before = path.read_bytes()
    ctx = make_ctx()
    assert _run(ctx, "status") == 0
    out = capsys.readouterr().out
    assert str(path) in out and "running" in out and "Rebuild the index" in out
    assert path.read_bytes() == before, "nothing is marked lost, expired or created"
    assert plugin.ledger.get("sa-7-live")["status"] == "running"


def test_status_before_any_ledger_exists(plugin, make_ctx, capsys, tmp_path, monkeypatch):
    plugin.ledger._db.close()
    plugin.ledger._db = None
    missing = tmp_path / "none" / "ledger.db"
    monkeypatch.setattr(plugin.ledger, "_default_path", lambda: missing)
    ctx = make_ctx()
    assert _run(ctx, "status") == 0
    assert "not created yet" in capsys.readouterr().out
    assert not missing.parent.exists()
