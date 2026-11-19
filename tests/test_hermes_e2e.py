"""End-to-end checks against a real Hermes checkout (skipped when Hermes is not importable).

Loads the plugin through Hermes' own plugin loader from a temporary HERMES_HOME and drives it
through Hermes' hook dispatch, so hook kwargs, the worker-thread/contextvar model, arg rewriting
and config lookup are the real ones.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

import pytest
import yaml

from helpers import PLUGIN_DIR

plugins_mod = pytest.importorskip("hermes_cli.plugins")

SESSION = "e2e-session"
CONTEXT = (
    "Known: https://app.example.com/ping returns 522 since the 10:00 deploy. Target: /srv/app/wrangler.toml. "
    "Do not touch other routes. Stop after one redeploy. Evidence: status line of the public URL."
)


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "super-agent", ignore=shutil.ignore_patterns("__pycache__"))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {
            "enabled": ["super-agent"],
            "entries": {"super-agent": {"settings": {
                "handoff_min_context_chars": 40,
                "terminal_auto_detach_seconds": 1,
            }}},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod.discover_plugins(force=True)
    manager = plugins_mod.get_plugin_manager()
    assert manager.has_hook("pre_tool_call"), "plugin did not load through the Hermes loader"
    yield manager


def _pre_tool(name, args, **kw):
    ids = {"session_id": SESSION, "turn_id": "t1", "task_id": "main", "tool_call_id": f"call-{time.time_ns()}"}
    ids.update(kw)
    return plugins_mod._dispatch_pre_tool_call_hooks(name, args, **ids)


def _start_turn(turn="t1"):
    return plugins_mod.invoke_hook(
        "pre_llm_call", session_id=SESSION, task_id="main", turn_id=turn, platform="weixin",
        user_message="帮我修一下 522", conversation_history=[{"role": "user", "content": "帮我修一下 522"}],
        is_first_turn=True, model="m")


def test_main_agent_through_hermes_dispatch(hermes):
    _start_turn()
    block, _ = _pre_tool("process_manage", {"action": "wait"})
    assert block and "blocks the chat" in block
    block, modified = _pre_tool("terminal", {"command": "ls"})
    assert block is None and modified is None


def test_handoff_gate_rewrites_args_through_hermes(hermes):
    _start_turn()
    block, _ = _pre_tool("delegate_task", {"tasks": [{"goal": "修复 522", "context": "看下"}]})
    assert block and "Handoff returned once" in block

    block, modified = _pre_tool("delegate_task", {"tasks": [{"goal": "修复 522", "context": CONTEXT}]})
    assert block is None
    task = modified["tasks"][0]
    assert task["context"].startswith(CONTEXT)
    assert "Handoff briefing" in task["context"] and "帮我修一下 522" in task["context"]
    assert "output_schema" not in task


def test_single_goal_handoff_keeps_the_goal_through_hermes(hermes):
    _start_turn()
    block, modified = _pre_tool("delegate_task", {"goal": "修复 522", "context": CONTEXT})
    assert block is None
    assert modified["goal"] == "修复 522", "Hermes merges modify args into the original call"
    assert "Handoff briefing" in modified["context"]


def test_worker_limits_through_hermes(hermes):
    _start_turn()
    _, modified = _pre_tool("delegate_task", {"tasks": [{"goal": "修复 522", "context": CONTEXT}]})
    plugins_mod.invoke_hook("subagent_start", parent_session_id=SESSION, parent_turn_id="t1",
                            parent_subagent_id=None, child_session_id="child-e2e",
                            child_subagent_id="sa-0-e2e", child_role="leaf", child_goal="修复 522")
    child = {"task_id": "sa-0-e2e", "session_id": "child-e2e", "turn_id": "ct"}
    curl = {"command": "curl -sI https://app.example.com/ping"}
    for i in range(5):
        plugins_mod.invoke_hook("post_tool_call", tool_name="terminal", args=curl,
                                result='{"output": "HTTP/2 522", "exit_code": 0}', duration_ms=200,
                                status="success", error_type=None, error_message=None,
                                api_request_id=f"r{i}", tool_call_id=f"c{i}", **child)
    block, _ = _pre_tool("terminal", {"command": "ls"}, **child)
    assert block and "STOP" in block

    plugins_mod.invoke_hook("subagent_stop", parent_session_id=SESSION, parent_turn_id="t1",
                            child_session_id="child-e2e", child_role="leaf", child_summary="done",
                            child_status="completed", tool_call_history=[], duration_ms=5000)
    results = _start_turn("t2")
    text = "\n".join(str(r.get("context", "")) for r in results if isinstance(r, dict))
    assert "The plugin stopped it" in text

    db_path = os.path.join(os.environ["HERMES_HOME"], "plugin-data", "super-agent", "ledger.db")
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT parent, status FROM tasks WHERE id = 'sa-0-e2e'").fetchone()
    assert row == (SESSION, "blocked"), "the ledger lives in Hermes' plugin-data directory"


def test_background_detach_with_real_wait_loop(hermes):
    """The tool_execution middleware arms the timer on the tool thread; Hermes' own wait loop then yields."""
    from tools.environments.base import BaseEnvironment

    _start_turn()
    yielded = {}

    def yield_handler(proc, output_so_far):
        yielded["output"] = output_so_far
        return {"yielded_session_id": "proc_e2e", "output": output_so_far, "pid": proc.pid}

    proc = subprocess.Popen([sys.executable, "-c", "import time; print('working', flush=True); time.sleep(8)"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    env_cls = type("_E2EEnv", (BaseEnvironment,), {})
    env_cls.__abstractmethods__ = frozenset()
    env = object.__new__(env_cls)
    from hermes_cli.middleware import run_tool_execution_middleware

    def run(args):
        return BaseEnvironment._wait_for_process(env, proc, timeout=60, watch_interrupt_tid=threading.get_ident(),
                                                 yield_handler=yield_handler)
    started = time.monotonic()
    try:
        result = run_tool_execution_middleware("terminal", {"command": "sleep 8"}, run, session_id=SESSION,
                                               task_id="main", turn_id="t1", tool_call_id="call-detach")
    finally:
        proc.kill()
        proc.wait()
    assert result.get("yielded_session_id") == "proc_e2e"
    assert time.monotonic() - started < 6, "yield should fire after ~1s, not wait for the process"
    from tools.interrupt import consume_yield
    assert not consume_yield(threading.get_ident()), "no yield may stay pending for the next command"


def test_plugin_passes_the_catalog_core_override_check(hermes):
    from pathlib import Path
    from hermes_cli.plugin_validate_core_override import core_override_findings
    assert core_override_findings(Path(PLUGIN_DIR)) == []


def test_tool_names_the_plugin_relies_on_exist_in_hermes(hermes):
    import model_tools  # noqa: F401  (registers Hermes' built-in tools)
    from tools.registry import registry

    plugin = next(m for name, m in sys.modules.items()
                  if name.endswith("super-agent.secretary") or name.endswith("super_agent.secretary")
                  or name.endswith("proactive-agent.secretary") or name.endswith("proactive_agent.secretary"))
    names = set(plugin.RED_ALLOWED_TOOLS) | {"terminal", "delegate_task", "process_manage", "write_file", "patch"}
    missing = sorted(n for n in names if registry.get_entry(n) is None)
    assert not missing, f"tools renamed or removed in Hermes: {missing}"


def test_report_cut_short_by_the_user_reaches_the_next_turn_through_hermes(hermes):
    import inspect
    assert "session_key" in inspect.signature(plugins_mod.PluginContext.inject_message).parameters

    _start_turn()
    plugins_mod.invoke_hook("subagent_start", parent_session_id=SESSION, parent_turn_id="t1",
                            parent_subagent_id=None, child_session_id="child-r", child_subagent_id="sa-0-r",
                            child_role="leaf", child_goal="清理磁盘")
    plugins_mod.invoke_hook("post_tool_call", tool_name="delegate_task", args={}, task_id="main", session_id=SESSION,
                            turn_id="t1", tool_call_id="c-d", api_request_id="r-d", duration_ms=50, status="ok",
                            error_type=None, error_message=None,
                            result='{"status": "dispatched", "mode": "background", "subagent_ids": ["sa-0-r"]}')
    plugins_mod.invoke_hook("on_session_end", session_id=SESSION, task_id="main", turn_id="t1", completed=True,
                            failed=False, interrupted=False, turn_exit_reason="text_response(finish_reason=stop)",
                            model="m", platform="weixin")
    plugins_mod.invoke_hook("subagent_stop", parent_session_id=SESSION, parent_turn_id="t1",
                            child_session_id="child-r", child_role="leaf", child_status="completed",
                            child_summary='{"status": "completed", "summary": "freed 3.4 GiB", "evidence": ["df -h"]}',
                            tool_call_history=[], duration_ms=5000)
    _start_turn("t2")
    redirect = {"role": "user", "content": "返回机制呢？", "api_content": (
        "[Context from the interrupted assistant response]\n[This response was interrupted by a user correction.]"
        "\n\n返回机制呢？")}
    plugins_mod.invoke_hook("pre_api_request", session_id=SESSION, turn_id="t2", task_id="main", platform="weixin",
                            conversation_history=[redirect], request_messages=[])
    plugins_mod.invoke_hook("on_session_end", session_id=SESSION, task_id="main", turn_id="t2", completed=True,
                            failed=False, interrupted=False, turn_exit_reason="text_response(finish_reason=stop)",
                            model="m", platform="weixin")
    results = _start_turn("t3")
    text = "\n".join(str(r.get("context", "")) for r in results if isinstance(r, dict))
    assert "has not heard" in text and "freed 3.4 GiB" in text


def _child_ids():
    return {"task_id": "sa-0-e2e", "session_id": "child-e2e", "turn_id": "ct"}


def _spawn_e2e_worker():
    _start_turn()
    plugins_mod.invoke_hook("subagent_start", parent_session_id=SESSION, parent_turn_id="t1",
                            parent_subagent_id=None, child_session_id="child-e2e",
                            child_subagent_id="sa-0-e2e", child_role="leaf", child_goal="清理缓存")


def test_hermes_approval_refusals_reach_the_worker_as_a_note(hermes):
    from tools.approval import _COMMAND_GATE
    from tools.terminal_tool import _error_json

    _spawn_e2e_worker()
    denied = _error_json(_COMMAND_GATE.cli_denied.format(breaker=""), status="blocked")
    pending = _error_json("", status="pending_approval", approval_pending=True, command="systemctl restart nginx",
                          description="restart a service", pattern_key="systemctl", smart_denied=False,
                          allow_permanent=True)
    for command, result, expected in (("rm -rf /srv/cache", denied, "refused this call"),
                                      ("systemctl restart nginx", pending, "waiting for the user's approval")):
        out = plugins_mod.invoke_hook("transform_tool_result", tool_name="terminal", args={"command": command},
                                      result=result, duration_ms=5, status="error", error_type="tool_error",
                                      error_message=None, tool_call_id=f"c-{command}", **_child_ids())
        note = json.loads(next(r for r in out if isinstance(r, str)))["super_agent_note"]
        assert expected in note and "needs_decision" in note


def test_real_terminal_tool_yields_through_the_middleware(hermes):
    """The thread the middleware runs on is the one Hermes' terminal tool waits on."""
    import json as _json
    from hermes_cli.middleware import run_tool_execution_middleware
    from tools.terminal_tool import terminal_tool

    _start_turn()
    args = {"command": "echo working; sleep 8", "timeout": 60}
    started = time.monotonic()
    out = run_tool_execution_middleware("terminal", args, lambda a: terminal_tool(task_id="e2e-term", **a),
                                        session_id=SESSION, task_id="main", turn_id="t1", tool_call_id="call-term")
    result = _json.loads(out)
    try:
        assert result.get("status") == "yielded_to_background", result
        assert time.monotonic() - started < 6
    finally:
        from tools.process_registry import process_registry
        if result.get("session_id"):
            process_registry.kill_process(result["session_id"])


def test_startup_preflight_warns_once_and_stays_silent_when_healthy(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "super-agent", ignore=shutil.ignore_patterns("__pycache__"))

    # 1. Misconfigured host: inline delegation on weixin + high max_iterations
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["super-agent"]},
        "display": {"platforms": {"weixin": {"interim_assistant_messages": False}}},
        "delegation": {"max_iterations": 200},
        "compression": {"in_place": False},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod.discover_plugins(force=True)

    r1 = _start_turn("t1")
    text1 = "\n".join(str(r.get("context", "")) for r in r1 if isinstance(r, dict))
    assert "weixin" in text1 and "inline" in text1
    assert "delegation.max_iterations is 200" in text1
    assert "compression.in_place is false" in text1

    r2 = _start_turn("t2")
    text2 = "\n".join(str(r.get("context", "")) for r in r2 if isinstance(r, dict))
    assert "Startup host self-check" not in text2

    # 2. Healthy host: background delegation + max_iterations 90 -> nothing injected
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["super-agent"]},
        "display": {"interim_assistant_messages": False},
        "delegation": {"gateway_mode": "background", "max_iterations": 90},
    }), encoding="utf-8")
    plugins_mod.discover_plugins(force=True)

    r3 = _start_turn("t3")
    text3 = "\n".join(str(r.get("context", "")) for r in r3 if isinstance(r, dict))
    assert "Startup host self-check" not in text3


def test_status_command_through_hermes_cli_parser(hermes, capsys):
    """The command Hermes' loader registered, wired the way hermes_cli/main.py::_attach_plugin_cli_command
    does it (importing hermes_cli.main runs Hermes' startup bootstrap, so the few lines are repeated here)."""
    import argparse

    info = hermes._cli_commands["super-agent"]
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command").add_parser(info["name"], help=info["help"],
                                                            description=info.get("description", ""))
    info["setup_fn"](sub)
    sub.set_defaults(func=info["handler_fn"])
    args = parser.parse_args(["super-agent", "status", "--json"])
    assert args.func(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ledger"].endswith(os.path.join("plugin-data", "super-agent", "ledger.db"))
    assert out["settings"]["handoff_min_context_chars"]["value"] == 40, "settings come from Hermes' config"


def _plugin_module(manager, name):
    """A submodule of the plugin as loaded now (earlier loads leave stale copies in sys.modules)."""
    hook = next(fn for fn in manager._hooks["on_human_input_request"]
                if fn.__name__ == "_on_human_input_request")
    return hook.__globals__[name]


def test_waiting_for_the_user_pauses_the_turn_clock_through_hermes(hermes, monkeypatch):
    """Hermes' own human-input wait (approval, clarify, sudo) fires the hooks with the chat's session."""
    from tools.human_input_hooks import human_input_request

    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", SESSION)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "weixin")
    _start_turn()
    secretary = _plugin_module(hermes, "secretary")
    with human_input_request("approval", prompt="systemctl restart nginx") as human:
        assert secretary._turns[SESSION]["waiting"], "the wait reached the plugin for this chat"
        human.outcome = "once"
    assert not secretary._turns[SESSION]["waiting"]


def test_real_write_file_reports_the_absolute_path(hermes, tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "app" / "wrangler.toml"
    target.parent.mkdir()
    out = write_file_tool(str(target), "name = 'app'\n", task_id="e2e-write")
    assert _plugin_module(hermes, "workers").written_paths({"path": str(target)}, out) == [str(target)], out


def test_a_rotating_compaction_keeps_the_turn_and_the_result_follows_the_chat(hermes, tmp_path, monkeypatch):
    """Hermes' compaction keeps the session id by default (compression.in_place: true). With in_place false
    it publishes a new session, re-points the agent at it mid-turn (agent.session_id, HERMES_SESSION_ID) and
    tells no hook. Observed here: later hooks of the turn carry the new id with the same turn id, and
    subagent_stop reads the parent's id when the subagent ends, so it names the new one."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from hermes_cli.config import DEFAULT_CONFIG
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent
    from tools.delegate_tool_results import _fire_subagent_stop_hooks

    assert DEFAULT_CONFIG["compression"]["in_place"] is True
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    db = open_persistence(profile_scope(home=Path(os.environ["HERMES_HOME"])))
    try:
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, persistence=db, session_id=SESSION, skip_context_files=True,
                        skip_memory=True, platform="weixin")
        agent.compression_in_place = False
        compressor = MagicMock()
        compressor.compress.return_value = [{"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                                            {"role": "user", "content": "帮我修一下 522"}]
        compressor.compression_count, compressor.last_prompt_tokens, compressor.last_completion_tokens = 1, 0, 0
        compressor._last_summary_error, compressor._last_compress_aborted = None, False
        agent.context_compressor = compressor

        _start_turn()
        plugins_mod.invoke_hook("subagent_start", parent_session_id=SESSION, parent_turn_id="t1",
                                parent_subagent_id=None, child_session_id="child-z", child_subagent_id="sa-0-z",
                                child_role="leaf", child_goal="重新部署")
        secretary = _plugin_module(hermes, "secretary")
        secretary._turns[SESSION]["start"] -= 200

        agent._compress_context([{"role": "user", "content": f"m{i}"} for i in range(10)], "sys",
                                approx_tokens=10_000)
        new_id = agent.session_id
        assert new_id != SESSION
        from gateway.session_context import get_session_env
        assert get_session_env("HERMES_SESSION_ID", "") == new_id

        block, _ = _pre_tool("terminal", {"command": "ls"}, session_id=new_id)
        assert block and "more than 120s" in block, "the turn kept its clock under the new id"
        assert SESSION not in secretary._turns

        child = MagicMock(session_id="child-z")
        summary = json.dumps({"status": "completed", "summary": "已重新部署", "evidence": ["HTTP/2 200"]})
        _fire_subagent_stop_hooks([{"task_index": 0, "summary": summary, "status": "completed",
                                    "duration_seconds": 3}], {0: child}, agent)
        ledger = _plugin_module(hermes, "ledger")
        assert ledger.get("sa-0-z")["parent"] == new_id
        assert any("HTTP/2 200" in n for n in ledger.pop_notes(new_id))
    finally:
        db.close()
