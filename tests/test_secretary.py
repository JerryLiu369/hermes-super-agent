"""Main agent: turn budget, background-job waits and governed scope."""

import json
from unittest.mock import patch

from helpers import GOOD_CONTEXT, main_tool, start_turn


def test_turn_budget_reminds_then_pauses_edits(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    assert main_tool(plugin, "terminal", {"command": "ls"}) is None

    plugin.secretary._turns["s1"]["start"] -= 65
    assert main_tool(plugin, "terminal", {"command": "ls"}) is None, "the reminder zone never blocks"
    note = plugin._transform_tool_result("terminal", {}, '{"output": "ok"}', session_id="s1", task_id="main")
    assert "Within the next 1-2 steps" in json.loads(note)["super_agent_note"]
    assert plugin._transform_tool_result("terminal", {}, '{"output": "ok"}', session_id="s1", task_id="main") is None

    plugin.secretary._turns["s1"]["start"] -= 60
    blocked = main_tool(plugin, "patch", {"path": "/x"})
    assert blocked["action"] == "block" and "more than 120s" in blocked["message"]
    for allowed, args in (("read_file", {}), ("memory", {}), ("process_manage", {"action": "list"})):
        assert main_tool(plugin, allowed, args) is None, allowed
    res = main_tool(plugin, "delegate_task", {"tasks": [{"goal": "g", "context": GOOD_CONTEXT}]})
    assert res["action"] == "modify"


def test_large_tool_output_does_not_pause_edits(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin._post_tool_call("read_file", {"path": "/x"}, "a" * 500000, session_id="s1", turn_id="t1", task_id="main")
    assert main_tool(plugin, "patch", {"path": "/x"}) is None


def test_new_turn_and_steer_reset_budget(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 200
    assert main_tool(plugin, "terminal", {"command": "ls"})["action"] == "block"
    steer = [{"role": "user", "content": "[OUT-OF-BAND USER MESSAGE] stop, do X instead"}]
    plugin._pre_api_request(session_id="s1", turn_id="t1", request_messages=steer)
    assert main_tool(plugin, "terminal", {"command": "ls"}) is None
    plugin.secretary._turns["s1"]["start"] -= 200
    start_turn(plugin, turn="t2")
    assert main_tool(plugin, "terminal", {"command": "ls"}, turn="t2") is None


def test_ungoverned_platforms_and_cron_are_untouched(plugin, make_ctx):
    make_ctx()
    for session, platform in (("s-cli", "cli"), ("s-tg", "telegram"), ("cron_abc_1", "weixin")):
        start_turn(plugin, session=session, platform=platform)
        assert main_tool(plugin, "process_manage", {"action": "wait"}, session=session) is None
        assert main_tool(plugin, "delegate_task", {"tasks": [{"goal": "g"}]}, session=session) is None


def test_cron_runs_are_recognised_by_hermes_names_and_flag(plugin, monkeypatch):
    context = plugin.context
    assert context.is_cron(task_id="cron:job1:9f2c") and context.is_cron(session_id="cron_job1_20261106_101500")
    assert not context.is_cron(session_id="s1", platform="weixin")
    monkeypatch.setenv("HERMES_CRON_SESSION", "true")
    assert context.is_cron(session_id="s1", platform="weixin")


def test_main_agent_cannot_wait_on_background_jobs(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    assert main_tool(plugin, "process_manage", {"action": "wait"})["action"] == "block"
    assert main_tool(plugin, "process_manage", {"action": "list"}) is None
    assert main_tool(plugin, "terminal", {"command": "grep -rn x ~/.hermes"}) is None


def test_yielded_terminal_result_explains_background(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    raw = json.dumps({"output": "Uploading 40%", "status": "yielded_to_background", "session_id": "proc_1"})
    out = json.loads(plugin._transform_tool_result("terminal", {}, raw, session_id="s1", task_id="main"))
    assert "NOT failed" in out["note"] and out["output"] == "Uploading 40%"


def test_plugin_errors_never_block_tools(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    with patch.object(plugin.secretary, "pre_tool_call", side_effect=RuntimeError("boom")):
        assert main_tool(plugin, "terminal", {"command": "ls"}) is None


def test_red_zone_keeps_only_reading_and_wrap_up(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 130
    for action in ("list", "poll", "log", "kill"):
        assert main_tool(plugin, "process_manage", {"action": action}) is None, action
    for action in ("write", "submit"):
        assert main_tool(plugin, "process_manage", {"action": action})["action"] == "block", action
    for allowed in ("skill_manage", "todo_list", "clarify", "memory", "session_search", "vision_analyze"):
        assert main_tool(plugin, allowed, {}) is None, allowed
    for paused in ("write_file", "patch", "terminal", "execute_code"):
        assert main_tool(plugin, paused, {})["action"] == "block", paused


def test_quoted_yield_json_does_not_replace_the_result(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    raw = 'previous run printed {"status": "yielded_to_background", "session_id": "p1"} and then more output'
    assert plugin._transform_tool_result("terminal", {}, raw, session_id="s1", task_id="main") is None


def test_redirect_counts_as_a_user_message_mid_turn(plugin, make_ctx):
    """busy_input_mode 'interrupt' redirects the live turn instead of steering it."""
    make_ctx()
    start_turn(plugin, history=[{"role": "user", "content": "please fix the site"}])
    plugin.secretary._turns["s1"]["start"] -= 200
    redirect = {"role": "user", "content": "use the staging host",
                "api_content": "[Context from the interrupted assistant response]\n"
                               "[This response was interrupted by a user correction.]\n\nuse the staging host"}
    history = [{"role": "user", "content": "please fix the site"}, redirect]
    plugin._pre_api_request(session_id="s1", turn_id="t1", conversation_history=history,
                            request_messages=[{"role": "user", "content": redirect["api_content"]}])
    assert main_tool(plugin, "terminal", {"command": "ls"}) is None
    assert plugin.secretary.user_words("s1")[-1] == "use the staging host"
    plugin._pre_api_request(session_id="s1", turn_id="t1", conversation_history=history)
    assert plugin.secretary.turn_facts("s1")["interjections"] == 1, "the same row is counted once"


def test_a_review_fork_does_not_reset_the_live_turn(plugin, make_ctx):
    """Hermes' background review shares the session id but runs under its own turn id."""
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 200
    old_steer = [{"role": "user", "content": "[OUT-OF-BAND USER MESSAGE] earlier"}]
    plugin._pre_api_request(session_id="s1", turn_id="review-1", conversation_history=old_steer)
    main_tool(plugin, "skills_list", {}, turn="review-1")
    assert main_tool(plugin, "terminal", {"command": "ls"})["action"] == "block"
    assert plugin.secretary.turn_facts("s1", "t1")["interjections"] == 0
    assert plugin.secretary.turn_facts("s1", "review-1") is None


def _aux(plugin, hook, task="compression", turn="t1", session="s1", platform="weixin"):
    # Hermes' auxiliary-call payload: the parent turn's identity plus aux_task (agent/auxiliary_hooks.py).
    getattr(plugin, hook)(session_id=session, task_id="main", turn_id=turn, platform=platform, aux_task=task,
                          model="m", provider="p", started_at=0.0)


def test_context_compression_is_kept_off_the_turn_budget(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    _aux(plugin, "_pre_auxiliary_call")
    # 50s of work, then a summary call that took 100s
    plugin.secretary._turns["s1"]["start"] -= 150
    plugin.secretary._turns["s1"]["paused_since"] -= 100
    assert main_tool(plugin, "terminal", {"command": "ls"}) is None, "compressing now does not count"
    _aux(plugin, "_post_auxiliary_call")
    assert plugin.secretary.zone("s1") == "green"
    assert main_tool(plugin, "terminal", {"command": "ls"}) is None
    plugin.secretary._turns["s1"]["start"] -= 75
    assert main_tool(plugin, "patch", {"path": "/x"})["action"] == "block", "the agent's own time still counts"


def test_other_auxiliary_calls_and_turns_keep_the_clock(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 130
    for kw in ({"task": "vision"}, {"turn": "review-1"}, {"platform": "subagent"}):
        _aux(plugin, "_pre_auxiliary_call", **kw)
        plugin.secretary._turns["s1"]["paused_since"] -= 100
        _aux(plugin, "_post_auxiliary_call", **kw)
        assert main_tool(plugin, "patch", {"path": "/x"})["action"] == "block", kw
    _aux(plugin, "_post_auxiliary_call")  # an unmatched end changes nothing
    assert main_tool(plugin, "patch", {"path": "/x"})["action"] == "block"


def _human(plugin, hook, request="req-1", session="s1", kind="approval", **extra):
    # Hermes' payload (tools/human_input_hooks.py): no turn id; the resolved call adds the outcome.
    getattr(plugin, hook)(kind=kind, request_id=request, session_id=session, session_key="agent:main:weixin:dm:u1",
                          platform="weixin", prompt="rm -rf build", **extra)


def _passes(plugin, seconds):
    """Let *seconds* go by in the running turn."""
    st = plugin.secretary._turns["s1"]
    st["start"] -= seconds
    if st["paused_since"]:
        st["paused_since"] -= seconds


def test_waiting_for_the_user_is_kept_off_the_turn_budget(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 50
    _human(plugin, "_on_human_input_request")
    _passes(plugin, 90)  # the user takes 90s to approve
    assert plugin.secretary.zone("s1") == "green"
    _human(plugin, "_on_human_input_resolved", outcome="once")
    assert main_tool(plugin, "patch", {"path": "/x"}) is None, "only the agent's own 50s count"

    _human(plugin, "_on_human_input_request", request="req-2", kind="clarify")
    _human(plugin, "_on_human_input_request", request="other", session="s2")
    _passes(plugin, 600)
    _human(plugin, "_on_human_input_resolved", request="req-2", kind="clarify", outcome="answered")
    assert plugin.secretary.zone("s1") == "green"
    plugin.secretary._turns["s1"]["start"] -= 75
    assert main_tool(plugin, "patch", {"path": "/x"})["action"] == "block"


def test_a_compression_that_never_ends_stops_pausing_the_clock(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    _aux(plugin, "_pre_auxiliary_call")
    _passes(plugin, 3600)
    assert main_tool(plugin, "patch", {"path": "/x"})["action"] == "block"


def test_a_wait_for_the_user_overlapping_a_compression_is_not_capped(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    _passes(plugin, 50)
    _human(plugin, "_on_human_input_request")
    _passes(plugin, 1200)  # the user takes 20 minutes; a compression starts near the end
    _aux(plugin, "_pre_auxiliary_call")
    _passes(plugin, 5)
    _human(plugin, "_on_human_input_resolved", outcome="once")
    _passes(plugin, 5)
    assert plugin.secretary.zone("s1") == "green", "the wait is not cut to the compression's cap"
    _aux(plugin, "_post_auxiliary_call")
    assert main_tool(plugin, "patch", {"path": "/x"}) is None


def test_red_zone_allows_reading_the_web(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 130
    for allowed in ("web_search", "web_extract"):
        assert main_tool(plugin, allowed, {"query": "x"}) is None, allowed



def test_a_compaction_that_moves_the_chat_to_a_new_session_keeps_the_turn(plugin, make_ctx, monkeypatch):
    """Hermes' rotating compaction (compression.in_place: false) re-points the agent at a new session id
    in the middle of a turn; the turn id stays, and auxiliary calls keep the id the turn started with."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:weixin:dm:u1")
    make_ctx()
    start_turn(plugin, history=[{"role": "user", "content": "please fix the site"}])
    steer = [{"role": "user", "content": "[OUT-OF-BAND USER MESSAGE] use the staging host"}]
    plugin._pre_api_request(session_id="s1", turn_id="t1", conversation_history=steer)
    plugin.secretary._turns["s1"]["start"] -= 200
    _aux(plugin, "_pre_auxiliary_call")
    _aux(plugin, "_post_auxiliary_call")

    # After the rotation: pre_tool_call carries no platform, and the gateway still binds the old one.
    assert main_tool(plugin, "terminal", {"command": "ls"}, session="s2")["action"] == "block", "the clock kept"
    facts = plugin.secretary.turn_facts("s2", "t1")
    assert facts["interjections"] == 1 and facts["session_key"] == "agent:main:weixin:dm:u1"
    assert facts["sessions"] == ["s1", "s2"]
    assert plugin.secretary.user_words("s2")[-1].endswith("use the staging host")
    assert "s1" not in plugin.secretary._turns and plugin.secretary.user_words("s1") == []
    assert plugin.context._session_platforms == {"s2": "weixin"}

    _aux(plugin, "_pre_auxiliary_call", session="s1")
    assert plugin.secretary._turns["s2"]["compressing"] == 1, "the old id still reaches the turn"
    _aux(plugin, "_post_auxiliary_call", session="s1")
    assert "s1" not in plugin.secretary._turns, "an old id never takes the turn back"
    plugin._on_turn_end(session_id="s2", task_id="main", turn_id="t1", completed=True, failed=False,
                        interrupted=False, turn_exit_reason="text_response(finish_reason=stop)", platform="weixin")
    assert plugin.secretary._turns == {} and plugin.secretary._user_words == {}


def test_a_new_session_with_another_turn_starts_fresh(plugin, make_ctx):
    make_ctx()
    start_turn(plugin)
    plugin.secretary._turns["s1"]["start"] -= 200
    start_turn(plugin, session="s2", turn="t9")
    assert main_tool(plugin, "terminal", {"command": "ls"}, session="s2", turn="t9") is None
    assert "s1" in plugin.secretary._turns


def test_ids_a_compaction_left_behind_age_out(plugin, make_ctx):
    make_ctx()
    plugin.handoff._user_at["old"] = 0.0
    plugin.handoff.user_wrote("s1")
    assert "old" not in plugin.handoff._user_at
    for n in range(plugin.context._MAX_SESSIONS + 5):
        plugin.context.remember_platform(f"s{n}", "weixin")
    assert len(plugin.context._session_platforms) == plugin.context._MAX_SESSIONS
    assert "s0" not in plugin.context._session_platforms
