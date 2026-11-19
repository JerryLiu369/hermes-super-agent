"""Tests for startup host-capability self-check (Issue #10)."""

import builtins
import concurrent.futures
import importlib.util
import sqlite3

import pytest
import yaml

from helpers import start_turn


def _hermes_importable() -> bool:
    try:
        return importlib.util.find_spec("hermes_cli.config_effective") is not None
    except ImportError:
        return False


# The host config checks read the config through Hermes' own resolvers and skip themselves when Hermes is not
# importable (as in CI), so tests that expect one of their findings need a Hermes checkout on the path.
requires_hermes = pytest.mark.skipif(not _hermes_importable(), reason="Hermes is not importable")


def _write_hermes_config(tmp_path, monkeypatch, data):
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))


def test_clean_host_injects_nothing(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background", "max_iterations": 90},
    })
    make_ctx()
    assert start_turn(plugin, turn="t1") is None
    assert start_turn(plugin, turn="t2") is None


def test_memory_fallback_reported_once(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background", "max_iterations": 90},
    })
    monkeypatch.setattr(plugin.ledger, "_open_default", lambda: sqlite3.connect(":memory:", check_same_thread=False))
    make_ctx()
    first = start_turn(plugin, turn="t1")
    assert plugin.ledger.is_memory_fallback() is True
    assert first and ":memory:" in first["context"]
    assert start_turn(plugin, turn="t2") is None, "startup notice must be injected only once"


def test_explicit_ledger_path_never_triggers_fallback_warning(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background", "max_iterations": 90},
    })
    plugin.ledger.load(tmp_path / "explicit.db")
    assert plugin.ledger.is_memory_fallback() is False

    # Even if load(":memory:") is called explicitly in a unit test, it should not false-positive.
    plugin.ledger.load(":memory:")
    assert plugin.ledger.is_memory_fallback() is False


def test_ledger_open_error_reported_once(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background", "max_iterations": 90},
    })
    monkeypatch.setattr(plugin.reports, "recover_lost", lambda: (_ for _ in ()).throw(PermissionError("read-only fs")))
    make_ctx()

    first = start_turn(plugin, turn="t1")
    assert first and "Task ledger failed to open" in first["context"] and "read-only fs" in first["context"]
    assert start_turn(plugin, turn="t2") is None


@requires_hermes
@pytest.mark.parametrize("mode, warned", [
    (None, True), ("auto", True), ("inline", True), ("Background", False), ("async", False),
])
def test_gateway_mode_other_than_background_is_reported(plugin, make_ctx, tmp_path, monkeypatch, mode, warned):
    # Interim assistant messages are on, so auto would run in the background today; the plugin still asks for
    # background, because interim messages are the user's display choice.
    delegation = {"max_iterations": 90} if mode is None else {"gateway_mode": mode, "max_iterations": 90}
    _write_hermes_config(tmp_path, monkeypatch, {"display": {"interim_assistant_messages": True},
                                                 "delegation": delegation})
    make_ctx()
    first = start_turn(plugin, turn="t1")
    text = first["context"] if first else ""
    assert ("delegation.gateway_mode is" in text) is warned
    if warned:
        assert "Set delegation.gateway_mode: background" in text and "interim_assistant_messages:" not in text
        assert ("(the default)" in text) is (mode is None)
    assert start_turn(plugin, turn="t2") is None


@requires_hermes
def test_gateway_mode_is_not_reported_when_nothing_is_governed(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {"delegation": {"max_iterations": 90}})
    make_ctx(config={"governed_platforms": []})
    assert not any("gateway_mode" in i for i in plugin.preflight.check_host())


@requires_hermes
def test_high_or_unset_max_iterations_warns_once(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background", "max_iterations": 200},
    })
    make_ctx()

    first = start_turn(plugin, turn="t1")
    assert first and "delegation.max_iterations is 200" in first["context"] and "90" in first["context"]
    assert start_turn(plugin, turn="t2") is None


@requires_hermes
def test_ungoverned_or_child_does_not_consume_notice(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background"},
    })
    make_ctx()

    assert plugin._pre_llm_call(session_id="cli-1", turn_id="t1", platform="cli") is None
    assert plugin._pre_llm_call(session_id="child-1", task_id="sa-0-abc", turn_id="t1", platform="subagent") is None
    assert plugin._pre_llm_call(session_id="cron_1", task_id="cron:1:1", turn_id="t1", platform="cron") is None

    governed = start_turn(plugin, session="wx-1", platform="weixin", turn="t1")
    assert governed and "delegation.max_iterations is unset" in governed["context"]
    assert start_turn(plugin, session="wx-1", platform="weixin", turn="t2") is None


@requires_hermes
def test_pop_notice_is_thread_safe_and_delivered_once(plugin, make_ctx, tmp_path, monkeypatch):
    _write_hermes_config(tmp_path, monkeypatch, {"delegation": {"max_iterations": 250}})
    make_ctx()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: plugin.preflight.pop_notice(), range(8)))

    non_empty = [r for r in results if r is not None]
    assert len(non_empty) == 1
    assert "delegation.max_iterations" in non_empty[0]


@requires_hermes
def test_preflight_does_not_wake_or_pollute_reports(plugin, make_ctx, tmp_path, monkeypatch):
    woken = []
    _write_hermes_config(tmp_path, monkeypatch, {"delegation": {"max_iterations": 250}})
    make_ctx(inject_message=lambda msg, **kw: woken.append(msg) or True)

    notice = start_turn(plugin, turn="t1")
    assert notice and "delegation.max_iterations" in notice["context"]
    plugin._on_turn_end(session_id="s1", task_id="main", turn_id="t1", completed=True, failed=False, interrupted=False)
    assert woken == []
    assert start_turn(plugin, turn="t2") is None


def test_hermes_modules_unavailable_degrades_gracefully(plugin, make_ctx, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(("gateway.", "hermes_cli.", "tools.")):
            raise ImportError(f"no {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    issues = plugin.preflight.check_host()
    assert isinstance(issues, list)


@requires_hermes
def test_missing_yield_calls_reported(plugin, make_ctx, tmp_path, monkeypatch):
    import sys
    _write_hermes_config(tmp_path, monkeypatch, {
        "delegation": {"gateway_mode": "background", "max_iterations": 90},
    })
    monkeypatch.setitem(sys.modules, "tools.interrupt", None)
    make_ctx()
    first = start_turn(plugin, turn="t1")
    assert first and "Background detach is unavailable" in first["context"]


@requires_hermes
@pytest.mark.parametrize("value, warned", [(False, True), ("no", True), (True, False), ("true", False), (None, False)])
def test_rotating_compaction_is_reported(plugin, make_ctx, tmp_path, monkeypatch, value, warned):
    data = {"delegation": {"gateway_mode": "background", "max_iterations": 90}}
    if value is not None:
        data["compression"] = {"in_place": value}
    _write_hermes_config(tmp_path, monkeypatch, data)
    make_ctx()
    first = start_turn(plugin, turn="t1")
    assert bool(first and "compression.in_place is false" in first["context"]) is warned
