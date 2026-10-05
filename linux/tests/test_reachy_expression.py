# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from dataclasses import FrozenInstanceError
import json

import pytest

from musegadget.reachy_expression import (
    ReplyProtocolError, ReplyRevisionError, SentenceStream, SpokenSentence,
    STREAM_VOICE_CONTEXT, spoken_reply, transcript_text,
)


def test_expression_controls_move_without_being_spoken():
    assert spoken_reply("Yes, I can nod. [reachy:nod]") == ("Yes, I can nod.", "nod")
    assert spoken_reply("Hello there.") == ("Hello there.", None)
    assert spoken_reply("Hello. [reachy:unrecognized]") == ("Hello.", None)
    assert spoken_reply("Hello. [reachy:welcoming1]") == ("Hello.", None)


def test_completed_plain_reply_cannot_speak_backend_tool_controls():
    with pytest.raises(ReplyProtocolError):
        spoken_reply('<atem:function_calls><atem:invoke name="system.delegate">')


def test_transcript_reads_only_assistant_text():
    transcript = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Question"}]},
        {"role": "tool", "content": [{"type": "text", "text": "Private tool data"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Hello."},
            {"type": "image", "text": "Do not read this"},
            {"type": "text", "text": "[reachy:happy]"},
        ]},
    ]}
    assert transcript_text(transcript) == "Hello.\n[reachy:happy]"


def test_completed_message_selects_its_own_text_from_a_multi_message_transcript():
    transcript = {"messages": [
        {"id": "old", "role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
        {"id": "current", "role": "assistant", "content": [{"type": "text", "text": "Current answer"}]},
        {"id": "other", "role": "assistant", "content": [{"type": "text", "text": "Other answer"}]},
    ]}
    assert transcript_text(transcript, "current") == "Current answer"
    assert transcript_text(transcript, "missing") == ""


def test_complete_word_and_phrase_frames_play_before_finalization():
    stream = SentenceStream()
    assert stream.mode == "undecided"
    assert stream.feed('{"text":"Wow","expression":"surprised"}') == [
        SpokenSentence("Wow", "surprised")]
    assert stream.mode == "protocol"
    assert stream.feed('\n{"text":"I can nod","expression":"nod"}') == [
        SpokenSentence("I can nod", "nod")]
    assert stream.committed == (
        SpokenSentence("Wow", "surprised"), SpokenSentence("I can nod", "nod"))
    with pytest.raises(FrozenInstanceError):
        stream.committed[0].text = "changed"


@pytest.mark.parametrize("fence", ["", "```json\n", "```ndjson\n", "```\n"])
def test_every_json_chunk_split_keeps_controls_out_of_speech(fence):
    objects = '{"text":"Wow!","expression":"surprised"}\n' + \
              '{"expression":"happy","text":"That is lovely."}'
    reply = fence + objects + ("\n```" if fence else "")
    expected = [SpokenSentence("Wow!", "surprised"), SpokenSentence("That is lovely.", "happy")]
    for split in range(len(reply) + 1):
        stream = SentenceStream()
        emitted = stream.feed(reply[:split]) + stream.feed(reply[split:])
        emitted += stream.finish(reply)
        assert emitted == expected
        assert stream.finish(reply) == []


def test_json_can_stream_character_by_character_and_without_line_separators():
    reply = '{"text":"A","expression":null}{"text":"B","expression":"shake"}'
    stream = SentenceStream()
    emitted = []
    for character in reply:
        emitted.extend(stream.feed(character))
    assert emitted == [SpokenSentence("A", None), SpokenSentence("B", "shake")]
    assert stream.finish(reply) == []


def test_progress_streams_but_is_not_committed_or_replayed_from_the_final_reply():
    progress = '{"text":"I am checking the current availability.","expression":"thinking","kind":"progress"}'
    answer = '{"text":"Three options are available.","expression":"happy"}'
    reply = progress + "\n" + answer
    for split in range(len(reply) + 1):
        stream = SentenceStream()
        emitted = stream.feed(reply[:split]) + stream.feed(reply[split:])
        assert emitted == [
            SpokenSentence("I am checking the current availability.", "thinking", "progress"),
            SpokenSentence("Three options are available.", "happy"),
        ]
        assert stream.committed == (SpokenSentence("Three options are available.", "happy"),)
        assert stream.finish(answer) == []


def test_progress_can_change_or_vanish_without_revising_committed_answers():
    stream = SentenceStream()
    assert stream.feed(
        '{"text":"I am checking availability.","expression":"thinking","kind":"progress"}'
        '{"text":"I found three options.","expression":"neutral","kind":"answer"}'
    ) == [
        SpokenSentence("I am checking availability.", "thinking", "progress"),
        SpokenSentence("I found three options.", "neutral"),
    ]
    final = (
        '{"text":"A different public update.","expression":null,"kind":"progress"}'
        '{"text":"I found three options.","expression":"neutral","kind":"answer"}'
    )
    assert stream.finish(final) == []


def test_progress_only_final_is_not_answer_content():
    progress = '{"text":"I am still working on that.","expression":null,"kind":"progress"}'
    stream = SentenceStream()
    assert stream.feed(progress) == [
        SpokenSentence("I am still working on that.", None, "progress")]
    assert stream.committed == ()
    assert stream.finish(progress) == []
    assert stream.committed == ()


@pytest.mark.parametrize("frame", [
    {"text": "Still working.", "expression": "thinking", "kind": "unknown"},
    {"text": "", "expression": "thinking", "kind": "progress"},
    {"text": "é" * 121, "expression": "thinking", "kind": "progress"},
    {"text": "Still working.", "expression": "happy", "kind": "progress"},
])
def test_invalid_progress_frames_are_rejected(frame):
    with pytest.raises(ReplyProtocolError):
        SentenceStream().finish(json.dumps(frame))


def test_progress_does_not_weaken_answer_revision_detection():
    stream = SentenceStream()
    stream.feed(
        '{"text":"Still working.","expression":"thinking","kind":"progress"}'
        '{"text":"The answer is three.","expression":"neutral"}'
    )
    with pytest.raises(ReplyRevisionError):
        stream.finish('{"text":"The answer is four.","expression":"neutral"}')


def test_authoritative_json_key_order_and_formatting_do_not_replay_audio():
    stream = SentenceStream()
    assert stream.feed('{"text":"Yes!","expression":"happy"}') == [SpokenSentence("Yes!", "happy")]
    final = '''{
      "expression": "happy", "text": "Yes!"
    }
    {"text": "I will nod.", "expression": "nod"}'''
    assert stream.finish(final) == [SpokenSentence("I will nod.", "nod")]
    assert stream.finish(final) == []


@pytest.mark.parametrize("final", [
    '{"text":"No!","expression":"happy"}',
    '{"text":"Yes!","expression":"sad"}',
    "A different answer.",
    "",
])
def test_a_committed_protocol_frame_cannot_be_revised(final):
    stream = SentenceStream()
    stream.feed('{"text":"Yes!","expression":"happy"}')
    with pytest.raises(ReplyRevisionError) as error:
        stream.finish(final)
    assert "Yes!" not in str(error.value)
    assert stream.committed == (SpokenSentence("Yes!", "happy"),)


def test_final_only_protocol_and_plain_transcripts_work():
    assert SentenceStream().finish('{"text":"Absolutely","expression":"nod"}') == [
        SpokenSentence("Absolutely", "nod")]
    stream = SentenceStream()
    assert stream.finish("Yes. I will nod. [reachy:nod]") == [
        SpokenSentence("Yes.", None), SpokenSentence("I will nod.", "nod")]
    assert stream.finish("Yes. I will nod. [reachy:nod]") == []
    assert SentenceStream().finish("") == []


def test_plain_sentences_wait_for_whitespace_and_final_tail_is_sent_once():
    stream = SentenceStream()
    assert stream.feed("Hello.") == []
    assert stream.feed(" Next sentence.") == [SpokenSentence("Hello.", None)]
    assert stream.finish("Hello. Next sentence.") == [SpokenSentence("Next sentence.", None)]
    assert stream.finish("Hello. Next sentence.") == []


@pytest.mark.parametrize("marker", ["nod", "shake", "surprised"])
def test_every_marker_chunk_split_keeps_a_late_expression_action(marker):
    reply = f"Yes. [reachy:{marker}]"
    for split in range(len(reply) + 1):
        stream = SentenceStream()
        emitted = stream.feed(reply[:split]) + stream.feed(reply[split:])
        emitted += stream.finish(reply)
        assert emitted == [SpokenSentence("Yes.", None), SpokenSentence("", marker)]
        assert stream.finish(reply) == []


@pytest.mark.parametrize("reply", [
    "Dr. Smith agrees. Next sentence.",
    "It costs 3.14 dollars. Next sentence.",
    "Visit https://example.com/?q=yes. Next sentence.",
    "U.S. policy changed. Next sentence.",
    "Wait... really? Next sentence.",
    '"Yes!" she said. Next sentence.',
    "Email me@example.com. Next sentence.",
])
def test_plain_streaming_preserves_abbreviations_numbers_urls_and_quotes(reply):
    for split in range(len(reply) + 1):
        stream = SentenceStream()
        emitted = stream.feed(reply[:split]) + stream.feed(reply[split:])
        emitted += stream.finish(reply)
        assert " ".join(item.text for item in emitted) == reply
        assert all(item.text not in ("Dr.", "U.S.", "Wait.", '"Yes!"') for item in emitted)


def test_plain_final_prefix_comparison_is_exact_and_content_free():
    stream = SentenceStream()
    stream.feed("Original answer. ")
    with pytest.raises(ReplyRevisionError) as error:
        stream.finish("Rewritten answer. A new tail.")
    assert "Original" not in str(error.value) and "Rewritten" not in str(error.value)
    assert stream.finish("Original answer. A new tail.") == [SpokenSentence("A new tail.", None)]


def test_accumulated_deltas_can_supply_the_final_reply():
    reply = "A short answer. And a tail [reachy:happy]"
    stream = SentenceStream()
    assert stream.feed(reply) == [SpokenSentence("A short answer.", None)]
    assert stream.finish(reply) == [SpokenSentence("And a tail", "happy")]


@pytest.mark.parametrize("reply", [
    '{"text":"Hello","expression":"welcoming1"}',
    '{"text":42,"expression":"happy"}',
    '{"text":"Hello","expression":false}',
    '{"text":"Hello","expression":"happy","extra":1}',
    '{"text":"Hello"}',
    '{"text":"Hello","expression":"happy","expression":"sad"}',
    '{"text":"Hello","expression":"happy",}',
    '[{"text":"Hello","expression":"happy"}]',
    '{"text":"\\ud800","expression":"happy"}',
    '```json\n{"text":"Hello","expression":"happy"}',
])
def test_invalid_protocol_is_never_spoken(reply):
    stream = SentenceStream()
    with pytest.raises(ReplyProtocolError):
        stream.finish(reply)
    assert stream.committed == ()


def test_plain_preamble_cannot_make_json_controls_speakable():
    stream = SentenceStream()
    with pytest.raises(ReplyProtocolError):
        stream.finish('Here is the reply: {"text":"Hello","expression":"happy"}')
    assert stream.committed == ()


@pytest.mark.parametrize("suffix", ['{"', '{"t', '{"text"', '```json'])
def test_plain_preamble_cannot_make_partial_json_or_fences_speakable(suffix):
    stream = SentenceStream()
    with pytest.raises(ReplyProtocolError):
        stream.finish("Here is the reply: " + suffix)
    assert stream.committed == ()


def test_plain_tail_holds_possible_json_and_fence_openers():
    assert SentenceStream().finish("An unfinished answer {") == [
        SpokenSentence("An unfinished answer", None)]
    assert SentenceStream().finish("An unfinished answer ``") == [
        SpokenSentence("An unfinished answer", None)]


def test_frame_markers_are_removed_from_speech():
    stream = SentenceStream()
    assert stream.feed('{"text":"Hello [reachy:nod]","expression":"happy"}') == [
        SpokenSentence("Hello ", "happy")]
    assert SentenceStream().finish("Hello [reachy:unfinished") == [SpokenSentence("Hello", None)]


@pytest.mark.parametrize("suffix", [
    "[reachy:happy unfinished [aside]. Next.",
    "[Reachy:happy unfinished [aside]. Next.",
    "[reachy:unfinished\n[aside]. Next.",
])
def test_an_incomplete_marker_cannot_be_hidden_by_another_bracket(suffix):
    stream = SentenceStream()
    assert stream.feed("Before " + suffix) == []
    assert stream.finish("Before " + suffix) == [SpokenSentence("Before", None)]


def test_reply_frame_and_text_limits_use_utf8_bytes():
    stream = SentenceStream()
    assert stream.feed("a" * 65536) == []
    with pytest.raises(ReplyProtocolError):
        stream.feed("a")
    with pytest.raises(ReplyProtocolError):
        SentenceStream().finish("é" * 32769)
    with pytest.raises(ReplyProtocolError):
        SentenceStream().feed('{"text":"' + "a" * 8193)
    assert SentenceStream().finish(json.dumps({"text": "a" * 2048, "expression": "neutral"})) == [
        SpokenSentence("a" * 2048, "neutral")]
    with pytest.raises(ReplyProtocolError):
        SentenceStream().finish(json.dumps({"text": "a" * 2049, "expression": "neutral"}))
    with pytest.raises(ReplyProtocolError):
        SentenceStream().finish(json.dumps({"text": "é" * 1025, "expression": "neutral"}))
    oversized_frame = '{"text":"ok",' + " " * 8192 + '"expression":"neutral"}'
    with pytest.raises(ReplyProtocolError):
        SentenceStream().feed(oversized_frame)


def test_stream_context_requests_chunks_and_preserves_motion_choices():
    assert "one JSON object per line" in STREAM_VOICE_CONTEXT
    assert "one complete conversational sentence" in STREAM_VOICE_CONTEXT
    assert "nod" in STREAM_VOICE_CONTEXT and "shake" in STREAM_VOICE_CONTEXT
    assert "[reachy:" not in STREAM_VOICE_CONTEXT


def test_both_voice_formats_describe_actual_robot_capabilities_and_expression_channel():
    from musegadget.reachy_expression import VOICE_CONTEXT
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
    from musegadget.reachy_expression import voice_context
    context = voice_context(stream_replies=True, antenna_mode=mode,
                            face_tracking_enabled=True)
    assert "one JSON object per line" in context
    assert expected in context
    assert "Local face tracking is enabled" in context
    assert "no visual information or identity recognition" in context


def test_voice_context_records_disabled_motion_and_rejects_unknown_antenna_mode():
    from musegadget.reachy_expression import voice_context
    context = voice_context(stream_replies=False, motion_enabled=False, antenna_mode="both")
    assert "Append one expression marker" in context
    assert "Movement is disabled" in context
    assert "Both antennas are enabled" not in context
    with pytest.raises(ValueError, match="antenna mode"):
        voice_context(stream_replies=True, antenna_mode="broken")


def test_detailed_progress_preserves_the_answer_and_its_expressions():
    progress = ("I checked the restaurant's published menu and found a sharing platter "
                "that includes two meat dishes and two vegetable dishes.")
    recommendation = ("For a group evening I would choose a comedy with a mystery, "
                      "because everyone can guess what happens next while sharing dinner.")
    wire = "\n".join(json.dumps(frame) for frame in (
        {"text": progress, "expression": "thinking", "kind": "progress"},
        {"text": recommendation, "expression": "happy"},
        {"text": "How many friends are coming?", "expression": "curious"},
    ))
    expected = [SpokenSentence(progress, "thinking", "progress"),
                SpokenSentence(recommendation, "happy"),
                SpokenSentence("How many friends are coming?", "curious")]
    stream = SentenceStream()
    emitted = []
    for offset in range(0, len(wire), 17):
        emitted.extend(stream.feed(wire[offset:offset + 17]))
    assert emitted == expected
    assert stream.finish(wire) == []
    assert stream.committed == tuple(expected[1:])


@pytest.mark.parametrize("control", ['<atem:function_calls><atem:invoke name="system.delegate">',
                                     '<tool_call>{"name":"search"}</tool_call>',
                                     '<think>Private internal reasoning</think>',
                                     '<function_calls><invoke>Internal arguments</invoke>'])
def test_tool_control_markup_is_never_spoken_from_plain_or_json_replies(control):
    with pytest.raises(ReplyProtocolError, match="Tool control markup"):
        SentenceStream().finish(control)
    with pytest.raises(ReplyProtocolError, match="Tool control markup"):
        SentenceStream().finish(json.dumps({"text": control, "expression": "thinking"}))


def test_split_tool_control_markup_is_rejected_before_a_tool_name_can_be_spoken():
    stream = SentenceStream()
    assert stream.feed("<atem") == []
    with pytest.raises(ReplyProtocolError, match="Tool control markup"):
        stream.feed(':invoke name="system.delegate">')
