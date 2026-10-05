# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from dataclasses import FrozenInstanceError

import pytest

from musegadget.reachy_progress import (
    PUBLIC_PROGRESS_PHRASES, BackendActivity, BackendStatus, ProgressPlan, ProgressUpdate,
    backend_activity_from_status, backend_status_from_event, progress_from_activity,
    validate_public_progress, waiting_progress,
)


@pytest.mark.parametrize("request_text, expected", [
    ("Find flights to Tokyo.", "I haven't received a flight progress update yet."),
    ("Compare airfares for tomorrow.", "I haven't received a flight progress update yet."),
    ("Explain why flights get delayed.",
     "I haven't received a flight progress update yet."),
    ("What is the weather?", "I haven't received a weather progress update yet."),
    ("Explain how weather systems form.",
     "I haven't received a weather progress update yet."),
    ("Check the forecast.", "I haven't received a weather progress update yet."),
    ("Plan my trip.", "I haven't received a detailed progress update yet."),
    ("Explain black holes.", "I haven't received a detailed progress update yet."),
])
def test_waiting_cue_names_the_result_without_inventing_an_action(request_text, expected):
    assert waiting_progress(request_text) == expected


@pytest.mark.parametrize("activity, expected", [
    ("Searching flights to Tokyo", "Looking at flights now."),
    ("Looking for flights", "Looking at flights now."),
    ("I'm currently searching for available flights", "Looking at flights now."),
    ("Checking fares now", "Checking fares now."),
    ("Searching for airfares", "Checking fares now."),
    ("Comparing nonstop and connecting flights", "Comparing the options."),
    ("Searching the web", "Searching the web now."),
    ("I am searching on the web", "Searching the web now."),
    ("Searching sources", "Searching sources now."),
    ("Checking the website", "Checking a website now."),
    ("Reading public website", "Checking a website now."),
    ("Using a connector", "Using a connector now."),
    ("Querying travel_connector connector", "Using a connector now."),
])
def test_public_action_maps_to_a_fixed_cue_without_copying_details(activity, expected):
    assert progress_from_activity(activity) == expected


@pytest.mark.parametrize("activity", [
    None, {}, "Thinking", "Researching", "Flight results pending", "Planning a trip",
    "Will compare fares later", "Checking availability", "Please search flights",
    'Searching flights {"access_token":"private"}', "Searching the web https://private.example",
])
def test_unknown_or_private_status_never_becomes_spoken_progress(activity):
    assert progress_from_activity(activity) is None


def test_every_cached_phrase_satisfies_the_public_progress_boundary():
    assert len(PUBLIC_PROGRESS_PHRASES) == len(set(PUBLIC_PROGRESS_PHRASES))
    for phrase in PUBLIC_PROGRESS_PHRASES:
        assert validate_public_progress(phrase) == phrase


def test_public_progress_has_utf8_limits_and_normalizes_spaces_without_word_quotas():
    assert validate_public_progress("  Comparing   the options.  ") == "Comparing the options."
    text = ("I checked the restaurant's published menu and found a sharing platter "
            "that includes two meat dishes and two vegetable dishes.")
    assert validate_public_progress(text) == text
    plan = ProgressPlan("Plan dinner.", 0)
    assert plan.offer(text, 15)
    assert plan.take(20) == ProgressUpdate(text, "frame")
    assert validate_public_progress("é" * 120) == "é" * 120
    assert validate_public_progress("é" * 121) is None


@pytest.mark.parametrize("text", [
    "", None, {}, "Working\nnow", "Working\x00now", "\ud800",
    ' {"text":"Working"}', "[reachy:thinking] Working", "```json",
    "Checking https://private.example", "Checking www.private.example",
    "Bearer private", "Checking api_key private", "Checking access-token private",
    "Checking your password",
])
def test_unspoken_payloads_are_rejected(text):
    assert validate_public_progress(text) is None


def test_waiting_cue_speaks_once_through_a_long_turn_without_updates():
    plan = ProgressPlan("Find flights.", 100)
    expected = ProgressUpdate(
        "I haven't received a flight progress update yet.", "fallback")
    assert plan.take(119.99) is None
    assert plan.take(120) == expected
    for due in (140, 160, 180, 200):
        assert plan.take(due) is None


def test_delayed_consumer_does_not_emit_catch_up_cues_back_to_back():
    plan = ProgressPlan("Explain black holes.", 0)
    expected = ProgressUpdate("I haven't received a detailed progress update yet.", "fallback")
    assert plan.take(20) == expected
    assert plan.offer("I found a useful public source.", 55)
    assert plan.take(55) == ProgressUpdate("I found a useful public source.", "frame")
    assert plan.offer("I found another relevant source.", 74)
    assert plan.take(74.99) is None
    assert plan.take(75) == ProgressUpdate("I found another relevant source.", "frame")


def test_latest_fresh_update_wins_and_spoken_updates_do_not_repeat():
    plan = ProgressPlan("Find flights.", 0)
    plan.offer("Looking at flights now.", 1, source="backend")
    plan.offer("Checking fares now.", 15, source="backend")
    assert plan.take(20) == ProgressUpdate("Checking fares now.", "backend")
    assert not plan.offer("Checking fares now.", 25, source="backend")
    plan.offer("Comparing the options.", 35, source="frame")
    assert plan.take(40) == ProgressUpdate("Comparing the options.", "frame")
    assert plan.take(60) is None


def test_backend_activity_uses_current_then_past_tense_without_claiming_stale_work():
    plan = ProgressPlan("Find flights.", 0)
    web = backend_activity_from_status("Searching web")
    assert web == BackendActivity(
        "Searching the web now.", "Muse's last reported step was searching the web.")
    assert plan.set_status(BackendStatus("working", web), 8.153)
    assert plan.take(20) == ProgressUpdate(
        "Muse's last reported step was searching the web.", "backend")
    sources = backend_activity_from_status("Searching sources")
    assert plan.set_status(BackendStatus("working", sources), 22.183)
    assert plan.take(40) == ProgressUpdate(
        "Muse's last reported step was searching sources.", "backend")
    assert plan.take(60) is None


def test_fresh_backend_activity_does_not_repeat_as_history():
    plan = ProgressPlan("Research this.", 0)
    sources = backend_activity_from_status("Searching sources")
    assert plan.set_status(BackendStatus("working", sources), 15)
    assert plan.take(20) == ProgressUpdate("Searching sources now.", "backend")
    assert plan.take(40) is None


@pytest.mark.parametrize("status", [
    "is working", "is responding", "Checking private records", None,
])
def test_generic_or_unknown_status_preserves_the_specific_milestone(status):
    plan = ProgressPlan("Research this.", 0)
    assert plan.set_status(backend_status_from_event("working", "Searching the web"), 8)
    assert plan.set_status(backend_status_from_event("unknown", status), 15)
    assert plan.take(20) == ProgressUpdate(
        "Muse's last reported step was searching the web.", "backend")


def test_arbitrary_backend_activity_cannot_bypass_the_fixed_phrase_allowlist():
    plan = ProgressPlan("Research this.", 0)
    assert not plan.set_status(BackendStatus("working", BackendActivity("Reading secrets.", "Read secrets.")), 15)
    assert plan.take(20) == ProgressUpdate("I haven't received a detailed progress update yet.", "fallback")


def test_new_identical_backend_report_refreshes_observation_time():
    plan = ProgressPlan("Research this.", 0)
    activity = backend_activity_from_status("Searching web")
    assert plan.set_status(BackendStatus("working", activity), 2)
    assert plan.set_status(BackendStatus("working", activity), 15)
    assert plan.take(20) == ProgressUpdate("Searching the web now.", "backend")


def test_fresh_public_frame_wins_one_cue_without_erasing_backend_history():
    plan = ProgressPlan("Research this.", 0)
    assert plan.set_status(backend_status_from_event("working", "Searching web"), 8)
    assert plan.offer("I found a relevant public source.", 15)
    assert plan.take(20) == ProgressUpdate("I found a relevant public source.", "frame")
    assert plan.take(40) == ProgressUpdate(
        "Muse's last reported step was searching the web.", "backend")


def test_unspoken_frame_survives_the_twenty_second_cadence_with_attribution():
    plan = ProgressPlan("What is the weather?", 0)
    assert plan.take(20) == ProgressUpdate(
        "I haven't received a weather progress update yet.", "fallback")
    plan.offer("I found a useful weather source.", 22)
    assert plan.take(40) == ProgressUpdate(
        "Earlier from Muse: I found a useful weather source.", "frame")
    assert not plan.offer("I found a useful weather source.", 41)
    assert plan.take(60) is None


def test_duplicate_pending_update_does_not_pretend_an_old_frame_is_fresh():
    plan = ProgressPlan("Find flights.", 0)
    assert plan.offer("Looking at flights now.", 1, source="backend")
    assert not plan.offer("Looking at flights now.", 15, source="backend")
    assert plan.take(20) == ProgressUpdate(
        "Earlier from Muse: Looking at flights now.", "backend")


def test_invalid_offer_preserves_an_existing_valid_update():
    plan = ProgressPlan("Find flights.", 0)
    plan.offer("Looking at flights now.", 16, source="backend")
    assert not plan.offer(' {"private":"payload"}', 17)
    assert plan.take(20) == ProgressUpdate("Looking at flights now.", "backend")


def test_stopping_discards_progress_and_prevents_waiting_cues_and_new_offers():
    plan = ProgressPlan("Find flights.", 0)
    plan.offer("Looking at flights now.", 17, source="backend")
    assert plan.set_status(backend_status_from_event("working", "Searching web"), 18)
    plan.stop()
    assert plan.take(20) is None
    assert not plan.offer("Comparing the options.", 40)
    assert not plan.set_status(backend_status_from_event("working", "Searching sources"), 40)
    assert plan.take(1000) is None


def test_a_new_turn_has_its_own_timing_and_progress():
    old = ProgressPlan("Find flights.", 0)
    old.offer("Looking at flights now.", 7)
    old.stop()
    new = ProgressPlan(
        "Explain black holes.", 10, first_delay_s=1, interval_s=2,
        expiry_s=1, max_updates=1)
    assert new.take(10.99) is None
    assert new.take(11) == ProgressUpdate(
        "I haven't received a detailed progress update yet.", "fallback")
    assert new.take(13) is None
    assert old.take(100) is None


def test_progress_updates_are_immutable():
    update = ProgressUpdate("Comparing the options.", "backend")
    with pytest.raises(FrozenInstanceError):
        update.text = "Different words."
    assert update == ProgressUpdate("Comparing the options.", "backend")


@pytest.mark.parametrize("code, text, expected", [
    ("working", "is working", BackendStatus("working")),
    ("responding", "is responding", BackendStatus("responding")),
    ("idle", None, BackendStatus("idle")),
    ("online", "online", BackendStatus("idle")),
    (None, "is working", BackendStatus("unknown")),
    ({}, "Thinking", BackendStatus("unknown")),
    ("private", "Reading private records", BackendStatus("unknown")),
    ("working", "Searching web https://private.example", BackendStatus("working")),
])
def test_event_mapper_keeps_only_verified_phase_and_allowlisted_actions(code, text, expected):
    assert backend_status_from_event(code, text) == expected


def test_backend_status_is_immutable_and_keeps_an_allowlisted_activity():
    status = backend_status_from_event("working", "Searching web")
    assert status == BackendStatus("working", BackendActivity(
        "Searching the web now.", "Muse's last reported step was searching the web."))
    with pytest.raises(FrozenInstanceError):
        status.phase = "responding"


def test_fresh_generic_working_preserves_prior_search_history_without_claiming_current_search():
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event("working", "Searching web"), 12)
    plan.set_status(backend_status_from_event("working", "is working"), 19)
    assert plan.take(20) == ProgressUpdate(
        "Muse's last reported step was searching the web.", "backend")


def test_fresh_responding_phase_combines_with_specific_milestone_history():
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event("working", "Searching web"), 8)
    plan.set_status(backend_status_from_event("responding", "is responding"), 19)
    assert plan.take(20) == ProgressUpdate(
        "Muse is preparing a reply. Its last reported step was searching the web.", "backend")
    assert plan.take(40) is None


def test_actual_operation_wire_sequence_preserves_milestones_through_phase_changes():
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event("working", "is working"), .514)
    plan.set_status(backend_status_from_event("working", "Searching web"), 8.153)
    assert plan.take(20) == ProgressUpdate(
        "Muse's last reported step was searching the web.", "backend")
    plan.set_status(backend_status_from_event("working", "Searching sources"), 22.183)
    plan.set_status(backend_status_from_event("responding", "is responding"), 24.998)
    assert plan.take(24.998) is None
    plan.set_status(backend_status_from_event("working", "Searching sources"), 25.449)
    assert plan.take(40) == ProgressUpdate(
        "Muse's last reported step was searching sources.", "backend")


def test_actual_field_timing_keeps_initial_phase_until_reply_arrives():
    plan = ProgressPlan("Find flights to Paris.", 0)
    plan.set_status(backend_status_from_event("working", "is working"), 2)
    assert plan.take(20) == ProgressUpdate(
        "Muse's latest status is that it's working on your request.", "backend")
    plan.set_status(backend_status_from_event("responding", "is responding"), 27.7)
    plan.stop()
    assert plan.take(40) is None


@pytest.mark.parametrize("phase, current, history", [
    ("working", "Muse says it's working on your request.",
     "Muse's latest status is that it's working on your request."),
    ("responding", "Muse is preparing a reply.",
     "Muse's latest status is that it's preparing a reply."),
])
def test_phase_without_specific_activity_speaks_once_in_current_or_attributed_form(phase, current, history):
    plan = ProgressPlan("Find flights.", 0)
    plan.set_status(backend_status_from_event(phase, "Private words that must not be copied"), 15)
    assert plan.take(20) == ProgressUpdate(current, "backend")
    assert plan.take(40) is None
    stale = ProgressPlan("Find flights.", 0)
    stale.set_status(backend_status_from_event(phase, None), 1)
    assert stale.take(20) == ProgressUpdate(history, "backend")
    assert stale.take(40) is None


@pytest.mark.parametrize("phase", ["online", "idle", "unknown"])
def test_idle_or_unknown_without_any_specific_report_uses_the_no_detail_cue(phase):
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event(phase, None), 15)
    assert plan.take(20) == ProgressUpdate("I haven't received a detailed progress update yet.", "fallback")


def test_latest_unspoken_frame_replaces_older_frame_and_survives_a_long_wait():
    plan = ProgressPlan("Research this.", 0)
    plan.offer("I found one public source.", 1)
    plan.offer("I found a second relevant public source.", 2)
    assert plan.take(100) == ProgressUpdate(
        "Earlier from Muse: I found a second relevant public source.", "frame")
    assert not plan.offer("I found a second relevant public source.", 105)
    assert plan.take(119.99) is None
    assert plan.take(120) is None


def test_full_eighteen_word_frame_keeps_every_word_when_attributed_as_earlier():
    text = "I found three public sources and will compare their published details before giving you a complete answer today."
    assert len(text.split()) == 18
    plan = ProgressPlan("Research this.", 0)
    assert plan.offer(text, 1)
    assert plan.take(20) == ProgressUpdate("Earlier from Muse: " + text, "frame")
    assert not plan.offer(text, 21)


def test_latest_specific_stage_supersedes_history_without_repeating_a_spoken_frame():
    plan = ProgressPlan("Find flights.", 0)
    plan.set_status(backend_status_from_event("working", "Searching web"), 8)
    plan.offer("I found a useful public source.", 15)
    assert plan.take(20) == ProgressUpdate("I found a useful public source.", "frame")
    plan.set_status(backend_status_from_event("working", "Checking fares"), 35)
    assert plan.take(40) == ProgressUpdate("Checking fares now.", "backend")
    assert plan.take(60) is None


def test_unchanged_backend_stage_speaks_once_across_long_wait_and_identical_reports():
    plan = ProgressPlan("Research this.", 0)
    status = backend_status_from_event("working", "Searching web")
    assert plan.set_status(status, 8)
    assert plan.take(20) == ProgressUpdate(
        "Muse's last reported step was searching the web.", "backend")
    for due in (40, 60, 80):
        assert plan.set_status(status, due - 1)
        assert plan.take(due) is None


def test_current_stage_is_not_spoken_again_when_its_wording_becomes_historical():
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event("working", "Searching web"), 15)
    assert plan.take(20) == ProgressUpdate("Searching the web now.", "backend")
    assert plan.take(40) is None
    assert plan.take(60) is None


def test_changed_stage_and_public_frame_remain_eligible_after_silent_ticks():
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event("working", "Searching web"), 15)
    assert plan.take(20) == ProgressUpdate("Searching the web now.", "backend")
    assert plan.take(40) is None
    plan.set_status(backend_status_from_event("working", "Searching sources"), 41)
    assert plan.take(41) == ProgressUpdate("Searching sources now.", "backend")
    assert plan.take(61) is None
    assert plan.offer("I found a useful public source.", 62)
    assert plan.take(62) == ProgressUpdate("I found a useful public source.", "frame")
    assert plan.take(82) is None


def test_new_responding_phase_speaks_once_without_repeating_search_history():
    plan = ProgressPlan("Research this.", 0)
    plan.set_status(backend_status_from_event("working", "Searching web"), 15)
    assert plan.take(20) == ProgressUpdate("Searching the web now.", "backend")
    plan.set_status(backend_status_from_event("responding", "is responding"), 39)
    assert plan.take(40) == ProgressUpdate("Muse is preparing a reply.", "backend")
    assert plan.take(60) is None


def test_same_stage_reported_by_frame_and_backend_does_not_repeat():
    plan = ProgressPlan("Research this.", 0)
    assert plan.offer("Searching the web now.", 1)
    assert plan.take(20) == ProgressUpdate("Earlier from Muse: Searching the web now.", "frame")
    plan.set_status(backend_status_from_event("working", "Searching web"), 39)
    assert plan.take(40) is None
    assert not plan.offer("Muse's last reported step was searching the web.", 41)


def test_first_fresh_specific_backend_activity_speaks_before_the_generic_deadline():
    plan = ProgressPlan("Research this.", 0)
    assert plan.take(8.9) is None
    assert plan.set_status(backend_status_from_event("working", "Searching web"), 9)
    assert plan.take(9) == ProgressUpdate("Searching the web now.", "backend")
    assert plan.set_status(backend_status_from_event("working", "Searching web"), 9.1)
    assert plan.take(28.99) is None
    assert plan.take(29) is None


def test_first_valid_public_progress_frame_speaks_immediately_then_keeps_cadence():
    plan = ProgressPlan("Research this.", 0)
    assert plan.offer("I found a relevant public source.", 9)
    assert plan.take(9) == ProgressUpdate("I found a relevant public source.", "frame")
    assert plan.offer("I found another relevant public source.", 10)
    assert plan.take(28.99) is None
    assert plan.take(29) == ProgressUpdate(
        "Earlier from Muse: I found another relevant public source.", "frame")


def test_generic_phase_does_not_bypass_the_first_deadline():
    plan = ProgressPlan("Research this.", 0)
    assert plan.set_status(backend_status_from_event("working", "is working"), 0)
    assert plan.take(0) is None
    assert plan.take(19.99) is None
    assert plan.take(20) == ProgressUpdate(
        "Muse's latest status is that it's working on your request.", "backend")


@pytest.mark.parametrize("kind", ["frame", "status"])
def test_stale_specific_update_does_not_bypass_the_first_deadline(kind):
    plan = ProgressPlan("Research this.", 0)
    if kind == "frame":
        assert plan.offer("I found a relevant public source.", 0)
    else:
        assert plan.set_status(backend_status_from_event("working", "Searching web"), 0)
    assert plan.take(10) is None
    expected = ("Earlier from Muse: I found a relevant public source."
                if kind == "frame"
                else "Muse's last reported step was searching the web.")
    assert plan.take(20) == ProgressUpdate(expected, "frame" if kind == "frame" else "backend")


@pytest.mark.parametrize("phase", ["working", "responding", "idle"])
def test_generic_phase_burst_preserves_fresh_specific_history_without_claiming_it_is_current(phase):
    plan = ProgressPlan("Research this.", 0)
    assert plan.set_status(backend_status_from_event("working", "Searching web"), 9)
    assert plan.set_status(backend_status_from_event(phase, "is " + phase), 9.05)
    update = plan.take(9.1)
    if phase == "responding":
        assert update == ProgressUpdate(
            "Muse is preparing a reply. Its last reported step was searching the web.",
            "backend")
    else:
        assert update == ProgressUpdate(
            "Muse's last reported step was searching the web.", "backend")


def test_stopping_before_take_suppresses_an_immediate_specific_update():
    plan = ProgressPlan("Research this.", 0)
    assert plan.set_status(backend_status_from_event("working", "Searching web"), 9)
    plan.stop()
    assert plan.take(9) is None
    assert not plan.offer("I found a relevant public source.", 9.1)


def test_fresh_backend_action_replaces_a_stale_unspoken_frame():
    plan = ProgressPlan("Research this.", 0)
    assert plan.offer("I started searching for sources.", 0)
    assert plan.set_status(backend_status_from_event("working", "Checking a website"), 12)
    assert plan.take(12) == ProgressUpdate("Checking a website now.", "backend")
    assert plan.take(32) is None
    assert not plan.offer("I started searching for sources.", 33)
