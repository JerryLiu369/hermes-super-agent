"""Startup host-capability self-check (zero model calls, reported once to the main agent)."""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional

from . import config, ledger, secretary

logger = logging.getLogger("hermes.plugins.super_agent")

_MAX_ITERATIONS_THRESHOLD = 120
_RECOMMENDED_MAX_ITERATIONS = 90

_lock = threading.Lock()
_staged: bool = False
_checked: bool = False
_ledger_error: Optional[str] = None
_pending_notice: Optional[str] = None


# delegation.gateway_mode values with which Hermes never runs a delegation inside the turn
# (tools/delegate_tool_config.py::_GATEWAY_MODES). "auto", the default, does on some platforms.
_BACKGROUND_MODES = ("background", "async")


def check_host(ledger_error: Optional[str] = None) -> List[str]:
    """Check host capabilities once at startup without calling any model. Returns issue strings."""
    issues: List[str] = []

    # 1. Task ledger persistence
    if ledger_error:
        issues.append(
            f"Task ledger failed to open ({ledger_error}); subagent task history and crash recovery are unavailable."
        )
    elif ledger.is_memory_fallback():
        issues.append(
            "Task ledger could not write to plugin-data/super-agent/ledger.db and fell back to :memory:; "
            "task history and crash recovery will not survive a gateway restart."
        )

    # Plugin-host isolation: the plugin runs outside the Hermes process. Hook arguments still arrive,
    # but the tool thread and the gateway's per-turn context (the chat's route key) do not.
    if secretary.in_plugin_host():
        issues.append(
            "Hermes runs this plugin in a separate plugin-host process (plugins.isolation: host). Long terminal "
            "commands stay in the foreground (no background detach), a report the user cut into waits for their "
            "next message instead of waking the main agent, and `hermes super-agent status` is not available. "
            "Set plugins.isolation: in_process for all of it."
        )

    # 2, 3 & 5. Host config checks on Hermes' effective config (degrade safely if Hermes is not importable).
    try:
        from hermes_cli.config_effective import load_user_config_effective
    except Exception as e:
        logger.debug("super-agent: skipping host config preflight (Hermes modules unavailable: %s)", e)
        return issues

    # 4. Background detach relies on tools.interrupt, which is not on Hermes' published plugin API.
    try:
        if (float(config.get("terminal_auto_detach_seconds")) > 0 and not secretary.in_plugin_host()
                and not secretary.detach_available()):
            issues.append(
                "Background detach is unavailable: this Hermes version has no tools.interrupt.request_yield/"
                "consume_yield, so long terminal commands stay in the foreground and block the turn. "
                "Delegate slow commands or run them with background=true."
            )
    except Exception as e:
        logger.debug("super-agent: detach availability check failed: %s", e)

    try:
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        default_max_iter = int((DEFAULT_CONFIG.get("delegation") or {}).get("max_iterations") or 250)
    except Exception:
        default_max_iter = 250

    try:
        user_config = load_user_config_effective()
    except Exception as e:
        logger.debug("super-agent: could not load effective user config for preflight: %s", e)
        return issues

    # 2. Only gateway_mode: background keeps every governed chat from waiting for its subagents. Under "auto"
    # Hermes runs them inside the turn wherever interim assistant messages are off, and that is the user's
    # display choice, not the plugin's.
    delegation_cfg = user_config.get("delegation") if isinstance(user_config, dict) else None
    raw_mode = delegation_cfg.get("gateway_mode") if isinstance(delegation_cfg, dict) else None
    mode = str(raw_mode or "auto").strip().lower()
    if config.governed_platforms() and mode not in _BACKGROUND_MODES:
        issues.append(
            f"delegation.gateway_mode is {mode}{' (the default)' if not raw_mode else ''}, so Hermes may run "
            "subagents inside the turn and the chat waits until they finish (with auto, it does so on any platform "
            "whose interim assistant messages are off, as on weixin by default). Set delegation.gateway_mode: "
            "background in ~/.hermes/config.yaml."
        )

    # 3. Check whether delegation.max_iterations is unset (defaults to 250) or notably high (> 120).
    raw_max_iter = delegation_cfg.get("max_iterations") if isinstance(delegation_cfg, dict) else None
    if raw_max_iter is None:
        max_iter = default_max_iter
        is_unset = True
    else:
        try:
            max_iter = int(raw_max_iter)
            is_unset = False
        except (TypeError, ValueError):
            max_iter = default_max_iter
            is_unset = True

    if max_iter > _MAX_ITERATIONS_THRESHOLD:
        if is_unset:
            issues.append(
                f"delegation.max_iterations is unset (Hermes defaults to {max_iter}); recommend setting "
                f"delegation.max_iterations: {_RECOMMENDED_MAX_ITERATIONS} in ~/.hermes/config.yaml to bound "
                f"worker subagent steps."
            )
        else:
            issues.append(
                f"delegation.max_iterations is {max_iter} (> {_MAX_ITERATIONS_THRESHOLD}); recommend lowering "
                f"it to ~{_RECOMMENDED_MAX_ITERATIONS} in ~/.hermes/config.yaml to bound worker subagent steps."
            )

    # 5. A rotating compaction gives the chat a new session id that no hook links to the old one.
    compression_cfg = user_config.get("compression") if isinstance(user_config, dict) else None
    in_place = compression_cfg.get("in_place") if isinstance(compression_cfg, dict) else None
    if in_place is not None and not _truthy(in_place):
        issues.append(
            "compression.in_place is false, so every context compaction moves the chat to a new session id. The "
            "plugin then loses track of what happened in the chat before it: running and unfinished subagent tasks, "
            "open questions and result notes still waiting. Set compression.in_place: true (Hermes' default) in "
            "~/.hermes/config.yaml."
        )

    return issues


def _truthy(value: Any) -> bool:
    # How Hermes reads compression.in_place (agent/agent_init.py, utils.env.is_truthy_value).
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _format_notice(issues: List[str]) -> Optional[str]:
    if not issues:
        return None
    return (
        "[super-agent] Startup host self-check found potential configuration issues "
        "(tell the user briefly once):\n" + "\n".join(f"- {item}" for item in issues)
    )


def stage_startup(ledger_error: Optional[str] = None) -> None:
    """Arm the one-time startup check without touching Hermes config locks during plugin load."""
    global _staged, _checked, _ledger_error, _pending_notice
    with _lock:
        _staged = True
        _checked = False
        _ledger_error = ledger_error
        _pending_notice = None


def set_ledger_error(error: str) -> None:
    """The ledger failed when it was first opened, after load: check (again) before the next notice."""
    global _staged, _checked, _ledger_error, _pending_notice
    with _lock:
        _staged = True
        _checked = False
        _ledger_error = error
        _pending_notice = None


def run(ledger_error: Optional[str] = None) -> Optional[str]:
    """Run startup checks once and stage a single-delivery notice if any issues are found."""
    global _staged, _checked, _ledger_error, _pending_notice
    issues = check_host(ledger_error=ledger_error)
    notice = _format_notice(issues)
    with _lock:
        _staged = False
        _checked = True
        _ledger_error = ledger_error
        _pending_notice = notice
    if notice:
        logger.warning("super-agent: startup host self-check found %d issue(s)", len(issues))
    return notice


def pop_notice() -> Optional[str]:
    """Return and clear the staged startup notice (thread-safe, evaluated and delivered at most once)."""
    global _staged, _checked, _pending_notice
    with _lock:
        if _staged and not _checked:
            _checked = True
            _staged = False
            err = _ledger_error
            try:
                issues = check_host(ledger_error=err)
            except Exception:
                logger.exception("super-agent: startup preflight failed")
                issues = []
            notice = _format_notice(issues)
            if notice:
                logger.warning("super-agent: startup host self-check found %d issue(s)", len(issues))
            return notice
        notice = _pending_notice
        _pending_notice = None
        return notice


def reset_all() -> None:
    global _staged, _checked, _ledger_error, _pending_notice
    with _lock:
        _staged = False
        _checked = False
        _ledger_error = None
        _pending_notice = None
