import asyncio

import pytest

from musegadget.reachy_capabilities import Expression, HeardAudio, Mode, Partial, ReplyStyle, SpokenLine
from musegadget.reachy_local_backends import LocalHearing, LocalVoice, MuseVoice, RuleNarrator, backends_for
from musegadget.streaming_transcription import StreamingTranscriptionError
from test_reachy_voice import FakeSession, mp3_tone  # noqa: F401


@pytest.mark.parametrize("name, expected", [
    ("happy", Expression.HAPPY), ("nod", Expression.NOD), (Expression.SHAKE, Expression.SHAKE),
    ("Happy", None), ("welcoming1", None), (None, None), (3, None), ([], None),
])
def test_expression_parse_maps_unknown_names_to_no_gesture(name, expected):
    assert Expression.parse(name) is expected


def narrated(reply, style, **options):
    return asyncio.run(RuleNarrator().lines("the request", reply, style, **options))


def test_marker_reply_becomes_one_line_with_its_last_known_expression():
    reply = "Octopuses have three hearts. [reachy:curious] [reachy:surprised]"
    assert narrated(reply, ReplyStyle.MARKER) == (
        SpokenLine("Octopuses have three hearts.", Expression.SURPRISED, "answer"),)


def test_muse_voice_reply_line_names_the_muse_message_that_holds_its_audio():
    reply = "Sunny and warm all day. [reachy:welcoming1]"
    assert narrated(reply, ReplyStyle.MUSE_VOICE, message_id="reply-1") == (
        SpokenLine("Sunny and warm all day.", None, "answer", "reply-1"),)


def test_plain_reply_is_spoken_sentence_by_sentence_without_gestures():
    reply = "Octopuses have three hearts. Two pump blood to the gills!  One pumps it everywhere else."
    assert narrated(reply, ReplyStyle.PLAIN_SHORT, message_id="reply-1") == (
        SpokenLine("Octopuses have three hearts.", None, "answer"),
        SpokenLine("Two pump blood to the gills!", None, "answer"),
        SpokenLine("One pumps it everywhere else.", None, "answer"))


def test_sentence_frames_are_left_to_the_streaming_reply_tracker():
    with pytest.raises(ValueError, match="parsed as they stream"):
        narrated('{"text":"Hi.","expression":"happy"}', ReplyStyle.EXPRESSIVE_JSON)


class Speech:
    def __init__(self):
        self.spoken = []

    async def stream(self, text, rate):
        self.spoken.append((text, rate))
        yield f"{text}@{rate}"


@pytest.mark.parametrize("mode, warmed, request_text, expected", [
    (Mode.ON_ROBOT, True, "What's the weather tomorrow?", SpokenLine("Let me check the weather.", None, "ack")),
    (Mode.ON_ROBOT, True, "Can you plan my trip to Lisbon?", SpokenLine("Let me look into that trip.", None, "ack")),
    (Mode.ON_ROBOT, True, "Hello there.", None),
    (Mode.ON_ROBOT, False, "What's the weather tomorrow?", None),
    (Mode.MUSE_VOICE, True, "What's the weather tomorrow?", None),
])
def test_acknowledgement_is_offered_only_once_its_phrase_can_play_at_once(mode, warmed, request_text, expected):
    async def scenario():
        local = {"speech": Speech(), "transcriber": "stt"} if mode is Mode.ON_ROBOT else {}
        backends = backends_for(mode, FakeSession(), **local)
        if warmed:
            await backends.voice.warm(16000)
        return await backends.narrator.acknowledge(request_text)
    assert asyncio.run(scenario()) == expected


def test_lost_streaming_transcription_has_no_endpoint():
    pytest.importorskip("av")
    class Recognizer:
        async def start(self):
            pass

        def open_turn(self):
            class Turn:
                partial = Partial("", 0, 1)

                def feed(self, pcm):
                    pass

                def finish(self):
                    recognition = asyncio.get_running_loop().create_future()
                    recognition.set_exception(StreamingTranscriptionError("recognizer lost the turn"))
                    return recognition

                def abort(self):
                    pass
            return Turn()

    async def scenario():
        turn = LocalHearing(Recognizer()).open_turn(16000, silence_s=.1, vad=Vad())
        speak_then_pause(turn)
        return await turn.take().ended.endpoint()
    assert asyncio.run(scenario()) is None


class Vad:
    def is_speech(self, pcm, rate):
        np = pytest.importorskip("numpy")
        return bool(np.any(np.frombuffer(pcm, dtype="<i2")))


class StreamingRecognizer:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.fed = 0
        self.finished = []

    async def start(self):
        pass

    def open_turn(self):
        recognizer = self

        class Turn:
            partial = Partial("", 0, 1)

            def feed(self, pcm):
                if recognizer.fail:
                    raise StreamingTranscriptionError("recognizer lost the turn")
                recognizer.fed += len(pcm)

            def finish(self):
                recognizer.finished.append("Hello Reachy.")
                recognition = asyncio.get_running_loop().create_future()
                recognition.set_result("Hello Reachy.")
                return recognition

            def abort(self):
                pass
        return Turn()


def speak_then_pause(turn):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    turn.feed(np.full(8000, .3, dtype=np.float32))
    assert turn.active
    turn.feed(np.zeros(4000, dtype=np.float32))


def heard_endpoint(turn):
    async def endpoint():
        heard = turn.take()
        return heard.failures, heard.transcript_lost, await heard.ended.endpoint()
    return asyncio.run(endpoint())


def test_streaming_hearing_feeds_speech_as_it_arrives_and_finishes_at_the_endpoint():
    pytest.importorskip("av")
    recognizer = StreamingRecognizer()
    turn = LocalHearing(recognizer).open_turn(16000, silence_s=.1, vad=Vad())
    speak_then_pause(turn)
    failures, transcript_lost, endpoint = heard_endpoint(turn)
    assert (failures, transcript_lost, endpoint.text, endpoint.audio[:4]) == (0, False, "Hello Reachy.", b"RIFF")
    assert recognizer.fed == 2 * (8000 + 5 * 320)


def test_failed_streaming_hearing_drops_the_accepted_turn():
    pytest.importorskip("av")
    turn = LocalHearing(StreamingRecognizer(fail=True)).open_turn(16000, silence_s=.1, vad=Vad())
    speak_then_pause(turn)
    assert turn.take() == HeardAudio(None, 1, True)


def test_batch_hearing_transcribes_the_ended_turn_once_started():
    pytest.importorskip("av")
    class Transcriber:
        def __init__(self):
            self.heard = []

        async def transcribe(self, wav):
            self.heard.append(wav[:4])
            return "Turn on the lights."

    transcriber = Transcriber()
    turn = LocalHearing(transcriber).open_turn(16000, silence_s=.1, vad=Vad())
    speak_then_pause(turn)
    failures, transcript_lost, endpoint = heard_endpoint(turn)
    assert (failures, transcript_lost, endpoint.text, transcriber.heard) == (0, False, "Turn on the lights.", [b"RIFF"])


def test_hearing_without_a_transcriber_hands_muse_the_audio():
    pytest.importorskip("av")
    turn = LocalHearing().open_turn(16000, silence_s=.1, vad=Vad())
    speak_then_pause(turn)
    failures, transcript_lost, endpoint = heard_endpoint(turn)
    assert (failures, transcript_lost, endpoint.text, endpoint.audio[:4]) == (0, False, None, b"RIFF")


def test_local_voice_replays_warmed_phrases_and_synthesizes_other_text():
    async def scenario():
        speech = Speech()
        voice = LocalVoice(speech, ("Yes?", "Let me look that up."))
        await voice.warm(24000)
        assert [chunk async for chunk in voice.stream(SpokenLine("Yes?", None, "notice"), 24000)] == ["Yes?@24000"]
        assert [chunk async for chunk in voice.prepare(SpokenLine("Yes?", None, "notice"), 24000)] == ["Yes?@24000"]
        assert [chunk async for chunk in voice.stream(SpokenLine("Hello.", None, "answer"), 16000)] == [
            "Hello.@16000"]
        assert voice.prepare(SpokenLine("Hello.", None, "answer"), 16000) is None
        assert speech.spoken == [("Yes?", 24000), ("Let me look that up.", 24000), ("Hello.", 16000)]
    asyncio.run(scenario())


def test_muse_voice_plays_the_muse_message_mp3_and_closes_it(mp3_tone):  # noqa: F811
    async def scenario():
        session = FakeSession(mp3_tone)
        voice = MuseVoice(session)
        chunks = [chunk async for chunk in voice.stream(SpokenLine("", None, "answer", "reply-1"), 16000)]
        assert session.tts_requests == ["reply-1"] and session.tts_closed == ["reply-1"]
        assert sum(len(chunk) for chunk in chunks) >= 1280
        assert voice.prepare(SpokenLine("", None, "answer", "reply-1"), 16000) is None
    asyncio.run(scenario())


@pytest.mark.parametrize("mode, options, style, voice", [
    (Mode.MUSE_VOICE, {}, ReplyStyle.MUSE_VOICE, MuseVoice),
    (Mode.ON_ROBOT, {}, ReplyStyle.MARKER, LocalVoice),
    (Mode.ON_ROBOT, {"stream_replies": True}, ReplyStyle.EXPRESSIVE_JSON, LocalVoice),
])
def test_each_mode_assembles_its_reply_style_voice_and_hearing(mode, options, style, voice):
    local = {"speech": "tts", "transcriber": "stt"} if mode is Mode.ON_ROBOT else {}
    backends = backends_for(mode, "session", **local, **options)
    assert (backends.reply_style(), type(backends.voice)) == (style, voice)
    assert backends.hearing.transcribes is (mode is Mode.ON_ROBOT)
