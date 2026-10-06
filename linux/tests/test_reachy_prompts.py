# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from musegadget.reachy.prompts import (
    CHAT_SETUP_SUFFIX, STREAM_VOICE_CONTEXT, VOICE_CONTEXT, setup_challenge_reply, spoken_request, voice_context,
    with_setup_challenge,
)
from musegadget.reachy.replies import ReplyStyle


def test_stream_context_requests_chunks_and_preserves_motion_choices():
    assert "one JSON object per line" in STREAM_VOICE_CONTEXT
    assert "one complete conversational sentence" in STREAM_VOICE_CONTEXT
    assert "nod" in STREAM_VOICE_CONTEXT and "shake" in STREAM_VOICE_CONTEXT
    assert "[reachy:" not in STREAM_VOICE_CONTEXT


def test_both_voice_formats_describe_actual_robot_capabilities_and_expression_channel():
    for context in (VOICE_CONTEXT, STREAM_VOICE_CONTEXT):
        assert "two independently movable antennas" in context
        assert "six degrees of freedom" in context
        assert "Its eyes are fixed" in context
        assert "does not send camera images" in context
        assert "without requiring robot tools" in context


@pytest.mark.parametrize("mode, expected", [
    ("both", "Both antennas are enabled"),
    ("left", "Only the left antenna is enabled"),
    ("right", "Only the right antenna is enabled"),
    ("none", "Antenna movement is disabled"),
])
def test_voice_context_records_reply_format_and_actual_capabilities(mode, expected):
    context = voice_context(ReplyStyle.EXPRESSIVE_JSON, antenna_mode=mode,
                            face_tracking_enabled=True)
    assert "one JSON object per line" in context
    assert expected in context
    assert "Local face tracking is enabled" in context
    assert "no visual information or identity recognition" in context


def test_voice_context_records_disabled_motion_and_rejects_unknown_antenna_mode():
    context = voice_context(ReplyStyle.MARKER, motion_enabled=False, antenna_mode="both")
    assert "Append one expression marker" in context
    assert "Movement is disabled" in context
    assert "Both antennas are enabled" not in context
    with pytest.raises(ValueError, match="antenna mode"):
        voice_context(ReplyStyle.EXPRESSIVE_JSON, antenna_mode="broken")


def test_side_chat_setup_lets_muse_answer_the_challenge_outside_the_spoken_format():
    assert "ignore the later spoken-response format" in CHAT_SETUP_SUFFIX


def test_side_chat_setup_ends_with_the_challenge_and_names_its_exact_reply():
    assert with_setup_challenge("Set up.", "c0ffee") == "Set up.\n\nInitialization challenge: reply exactly Ready c0ffee"
    assert setup_challenge_reply("c0ffee") == "Ready c0ffee"


def test_a_spoken_request_follows_its_voice_context():
    assert spoken_request("Context.", "What time is it?") == "Context.\n\nThe user's spoken request is: What time is it?"


def test_plain_context_describes_the_robot_then_the_format_then_the_session():
    assert voice_context(ReplyStyle.PLAIN_SHORT, antenna_mode="left") == (
        "You are Muse speaking through Reachy Mini, a tabletop robot with a microphone, speaker, "
        "a head with six degrees of freedom, a rotating body, and two independently movable antennas. "
        "Its eyes are fixed; it has no arms and cannot walk. The adapter does not send camera images "
        "and performs your expressions without requiring robot tools. Answer in one to three short, "
        "plain spoken sentences, leading with the answer. Use no markup, lists, emoji, JSON or "
        "expression markers. Check time-sensitive facts with available tools and say plainly what you "
        "could not verify. These instructions replace earlier Reachy conversation instructions. "
        "Only the left antenna is enabled in this session.")
