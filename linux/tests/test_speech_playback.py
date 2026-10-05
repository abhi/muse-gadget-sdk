"""Lossless output pacing while an independently captured user is speaking."""

import asyncio

import pytest

from musegadget.speech_playback import PCMPlayer, SpeechPlayback


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds
        await asyncio.sleep(0)


def test_gate_waits_without_affecting_other_tasks_and_counts_the_current_hold():
    async def scenario():
        clock = Clock()
        gate = SpeechPlayback(clock=clock)
        assert await gate.wait_until_quiet() == 0
        gate.set_user_speaking(True)
        waiter = asyncio.create_task(gate.wait_until_quiet())
        await asyncio.sleep(0)
        clock.now = 1.25
        gate.set_user_speaking(True)
        assert gate.user_speaking and gate.paused_s == 1.25
        assert not waiter.done()
        clock.now = 2.5
        gate.set_user_speaking(False)
        assert await waiter == 2.5
        assert not gate.user_speaking and gate.paused_s == 2.5
        clock.now = 3
        gate.set_user_speaking(True)
        clock.now = 3.75
        gate.set_user_speaking(False)
        assert gate.paused_s == 3.25

    asyncio.run(scenario())


def test_cancelling_a_wait_does_not_resume_the_speaker_or_clear_another_wait():
    async def scenario():
        gate = SpeechPlayback()
        gate.set_user_speaking(True)
        first = asyncio.create_task(gate.wait_until_quiet())
        second = asyncio.create_task(gate.wait_until_quiet())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert gate.user_speaking and not second.done()
        gate.set_user_speaking(False)
        await second

    asyncio.run(scenario())


@pytest.mark.parametrize("diagnostics", [False, True])
def test_every_sample_is_sent_once_in_bounded_chunks_and_audio_lead_is_bounded(diagnostics):
    async def scenario():
        clock = Clock()
        sent, queued_end = [], 0.0

        async def push(samples):
            nonlocal queued_end
            sent.extend(samples)
            assert len(samples) <= 4
            queued_end = max(queued_end, clock()) + len(samples) / 100
            assert queued_end - clock() <= .080000001

        player = PCMPlayer(100, push, SpeechPlayback(clock=clock), clock=clock, sleep=clock.sleep,
                           diagnostics=diagnostics)
        await player.play(list(range(11)))
        await player.play(list(range(11, 29)))
        await player.finish()
        assert sent == list(range(29))
        assert clock() == pytest.approx(.34)
        assert player.paused_s == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("diagnostics", [False, True])
def test_pause_drains_only_committed_audio_and_resumes_the_exact_remaining_samples(diagnostics):
    async def scenario():
        clock = Clock()
        gate = SpeechPlayback(clock=clock)
        sent, events = [], []
        paused = asyncio.Event()

        async def push(samples):
            sent.extend(samples)
            if len(sent) == 8:
                gate.set_user_speaking(True)

        async def start():
            events.append(("start", clock()))

        async def pause():
            events.append(("pause", clock()))
            paused.set()

        async def resume():
            events.append(("resume", clock()))

        player = PCMPlayer(100, push, gate, on_start=start, on_pause=pause, on_resume=resume,
                           clock=clock, sleep=clock.sleep, diagnostics=diagnostics)
        task = asyncio.create_task(player.play(list(range(17))))
        await paused.wait()
        assert sent == list(range(8))
        assert events == [("start", 0), ("pause", pytest.approx(.13))]
        assert not task.done()
        # Recognition and backend work can run while output is paused.
        background = asyncio.create_task(asyncio.sleep(0, result="Muse accepted the next question"))
        assert await background == "Muse accepted the next question"
        clock.now = 1
        gate.set_user_speaking(False)
        await task
        await player.finish()
        assert sent == list(range(17))
        assert [name for name, _ in events] == ["start", "pause", "resume"]
        assert events[-1][1] == 1
        assert player.paused_s == pytest.approx(1)

    asyncio.run(scenario())


def test_initial_user_speech_defers_pose_and_all_audio_until_quiet():
    async def scenario():
        gate = SpeechPlayback()
        gate.set_user_speaking(True)
        sent, expressions = [], []

        async def push(samples):
            sent.extend(samples)

        async def start():
            expressions.append("happy")

        player = PCMPlayer(100, push, gate, on_start=start)
        task = asyncio.create_task(player.play([1, 2]))
        await asyncio.sleep(0)
        assert not sent and not expressions
        gate.set_user_speaking(False)
        await task
        assert sent == [1, 2] and expressions == ["happy"]

    asyncio.run(scenario())


def test_user_speech_during_the_first_pose_does_not_allow_an_audio_push():
    async def scenario():
        gate = SpeechPlayback()
        sent, poses = [], []
        paused = asyncio.Event()

        async def push(samples):
            sent.extend(samples)

        async def start():
            poses.append("start")
            gate.set_user_speaking(True)

        async def pause():
            poses.append("pause")
            paused.set()

        async def resume():
            poses.append("resume")

        player = PCMPlayer(100, push, gate, on_start=start, on_pause=pause, on_resume=resume)
        task = asyncio.create_task(player.play([1, 2]))
        await paused.wait()
        assert not sent
        gate.set_user_speaking(False)
        await task
        assert sent == [1, 2] and poses == ["start", "pause", "resume"]

    asyncio.run(scenario())


def test_paused_player_cancellation_never_pushes_the_retained_tail():
    async def scenario():
        gate = SpeechPlayback()
        sent = []
        paused = asyncio.Event()

        async def push(samples):
            sent.extend(samples)
            gate.set_user_speaking(True)

        async def pause():
            paused.set()

        player = PCMPlayer(100, push, gate, on_pause=pause)
        task = asyncio.create_task(player.play(list(range(9))))
        await asyncio.wait_for(paused.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set_user_speaking(False)
        await asyncio.sleep(0)
        assert sent == list(range(4))

    asyncio.run(scenario())


def test_finish_drains_accepted_audio_without_waiting_for_user_to_stop():
    async def scenario():
        clock = Clock()
        gate = SpeechPlayback(clock=clock)

        async def push(samples):
            gate.set_user_speaking(True)

        player = PCMPlayer(100, push, gate, clock=clock, sleep=clock.sleep)
        await player.play([1, 2])
        await player.finish()
        assert clock() == pytest.approx(.07)
        assert gate.user_speaking

    asyncio.run(scenario())


def test_brief_user_speech_that_ends_during_the_committed_tail_does_not_restart_the_pose():
    async def scenario():
        clock = Clock()
        gate = SpeechPlayback(clock=clock)
        sent, poses = [], []

        async def push(samples):
            sent.extend(samples)
            if len(sent) == 8:
                gate.set_user_speaking(True)

        async def sleep(seconds):
            await clock.sleep(seconds)
            if clock() > .08:
                gate.set_user_speaking(False)

        async def start():
            poses.append("start")

        async def pause():
            poses.append("pause")

        async def resume():
            poses.append("resume")

        player = PCMPlayer(100, push, gate, on_start=start, on_pause=pause, on_resume=resume,
                           clock=clock, sleep=sleep)
        await player.play(list(range(13)))
        await player.finish()
        assert sent == list(range(13))
        assert poses == ["start"]

    asyncio.run(scenario())


def test_multiple_interruptions_preserve_audio_and_sentence_callbacks():
    async def scenario():
        clock = Clock()
        gate = SpeechPlayback(clock=clock)
        sent, poses = [], []

        async def push(samples):
            sent.extend(samples)
            if len(sent) in (4, 12):
                gate.set_user_speaking(True)

        async def start():
            poses.append("start")

        async def pause():
            poses.append("pause")
            clock.now += .5
            gate.set_user_speaking(False)

        async def resume():
            poses.append("resume")

        player = PCMPlayer(100, push, gate, on_start=start, on_pause=pause, on_resume=resume,
                           clock=clock, sleep=clock.sleep)
        await player.play(list(range(23)))
        await player.finish()
        assert sent == list(range(23))
        assert poses == ["start", "pause", "resume", "pause", "resume"]
        assert gate.paused_s == pytest.approx(1.22)

    asyncio.run(scenario())


def test_push_failure_is_propagated_without_replaying_accepted_audio():
    async def scenario():
        sent = []

        async def push(samples):
            if sent:
                raise RuntimeError("speaker write failed")
            sent.extend(samples)

        player = PCMPlayer(100, push, SpeechPlayback())
        with pytest.raises(RuntimeError, match="speaker write failed"):
            await player.play(list(range(13)))
        assert sent == list(range(4))

    asyncio.run(scenario())


@pytest.mark.parametrize("rate", [0, -1, True, 1.5])
def test_invalid_sample_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="sample rate"):
        PCMPlayer(rate, None, SpeechPlayback())


@pytest.mark.parametrize("options", [{"chunk_s": 0}, {"lead_s": -1}, {"drain_margin_s": float("nan")}])
def test_invalid_pacing_is_rejected(options):
    with pytest.raises(ValueError, match="pacing"):
        PCMPlayer(16000, None, SpeechPlayback(), **options)


def test_diagnostics_are_opt_in():
    async def scenario():
        clock = Clock()

        async def push(samples):
            pass

        player = PCMPlayer(100, push, SpeechPlayback(clock=clock), clock=clock, sleep=clock.sleep)
        await player.play([1, 2, 3])
        await player.finish()
        assert player.metrics is None

    asyncio.run(scenario())


def test_metrics_separate_push_waits_from_scheduler_lateness_and_count_actual_audio():
    async def scenario():
        clock = Clock()
        sent = []
        push_delays = iter([.01, .02, .03])
        wake_delays = iter([.005, .01, .02, 0])

        async def push(samples):
            clock.now += next(push_delays)
            sent.extend(samples)

        async def sleep(seconds):
            clock.now += seconds + next(wake_delays)
            await asyncio.sleep(0)

        player = PCMPlayer(100, push, SpeechPlayback(clock=clock), clock=clock, sleep=sleep,
                           diagnostics=True)
        await player.play(list(range(12)))
        await player.finish()
        assert sent == list(range(12))
        assert player.metrics == pytest.approx({
            "play_calls": 1, "input_frames": 12, "input_audio_s": .12,
            "pushed_frames": 12, "pushed_audio_s": .12, "push_count": 3,
            "play_wall_s": .11, "drain_wall_s": .07,
            "push_wait_ms_p50": 20, "push_wait_ms_p95": 29, "push_wait_ms_max": 30,
            "scheduler_wake_count": 4,
            "scheduler_wake_lateness_ms_p50": 7.5,
            "scheduler_wake_lateness_ms_p95": 18.5,
            "scheduler_wake_lateness_ms_max": 20,
            "source_gap_ms_max": 0, "paused_s": 0,
            "percentile_window": 512, "push_wait_window_count": 3,
            "scheduler_wake_window_count": 4,
        })
        snapshot = player.metrics
        snapshot["input_frames"] = -1
        assert player.metrics["input_frames"] == 12

    asyncio.run(scenario())


def test_producer_gap_excludes_explicit_drain_and_user_hold_without_double_subtraction():
    async def scenario():
        clock = Clock()
        gate = SpeechPlayback(clock=clock)

        async def push(samples):
            pass

        player = PCMPlayer(100, push, gate, clock=clock, sleep=clock.sleep, diagnostics=True)
        await player.play([1, 2, 3, 4])
        # A real producer stall is retained independently of later intended gaps.
        clock.now += .12
        await player.play([5, 6, 7, 8])
        await player.finish()
        gate.set_user_speaking(True)
        clock.now += 2
        gate.set_user_speaking(False)
        clock.now += .03
        await player.play([9, 10, 11, 12])
        assert player.metrics["source_gap_ms_max"] == pytest.approx(120)
        # A drain that overlaps a user hold must be excluded only once.
        gate.set_user_speaking(True)
        await player.finish()
        clock.now += 1
        gate.set_user_speaking(False)
        clock.now += .2
        await player.play([13, 14, 15, 16])
        assert player.metrics["source_gap_ms_max"] == pytest.approx(200)
        assert player.metrics["paused_s"] == pytest.approx(3.09)
        assert player.metrics["drain_wall_s"] == pytest.approx(.18)

    asyncio.run(scenario())


def test_percentiles_have_bounded_windows_but_counts_and_maxima_cover_all_pushes():
    async def scenario():
        clock = Clock()
        push_count, wake_count = 0, 0

        async def push(samples):
            nonlocal push_count
            push_count += 1
            clock.now += 2 if push_count == 1 else .001

        async def sleep(seconds):
            nonlocal wake_count
            wake_count += 1
            clock.now += seconds + (.5 if wake_count == 1 else .002)
            await asyncio.sleep(0)

        player = PCMPlayer(100000, push, SpeechPlayback(clock=clock), chunk_s=.00001,
                           clock=clock, sleep=sleep, diagnostics=True)
        await player.play(list(range(520)))
        metrics = player.metrics
        assert metrics["push_count"] == metrics["scheduler_wake_count"] == 520
        assert metrics["input_frames"] == metrics["pushed_frames"] == 520
        assert metrics["input_audio_s"] == pytest.approx(.0052)
        assert metrics["push_wait_window_count"] == metrics["scheduler_wake_window_count"] == 512
        assert metrics["push_wait_ms_p50"] == metrics["push_wait_ms_p95"] == pytest.approx(1)
        assert metrics["push_wait_ms_max"] == pytest.approx(2000)
        assert metrics["scheduler_wake_lateness_ms_p50"] == pytest.approx(2)
        assert metrics["scheduler_wake_lateness_ms_p95"] == pytest.approx(2)
        assert metrics["scheduler_wake_lateness_ms_max"] == pytest.approx(500)

    asyncio.run(scenario())


def test_failed_push_wait_is_observed_without_counting_unsent_frames():
    async def scenario():
        clock = Clock()
        sent = []

        async def push(samples):
            clock.now += .01 if not sent else .07
            if sent:
                raise RuntimeError("speaker write failed")
            sent.extend(samples)

        player = PCMPlayer(100, push, SpeechPlayback(clock=clock), clock=clock, sleep=clock.sleep,
                           diagnostics=True)
        with pytest.raises(RuntimeError, match="speaker write failed"):
            await player.play(list(range(8)))
        assert sent == list(range(4))
        metrics = player.metrics
        assert metrics["input_frames"] == 8 and metrics["pushed_frames"] == 4
        assert metrics["push_count"] == 2 and metrics["scheduler_wake_count"] == 1
        assert metrics["push_wait_ms_p50"] == pytest.approx(40)
        assert metrics["push_wait_ms_max"] == pytest.approx(70)
        assert metrics["play_wall_s"] == pytest.approx(.08)

    asyncio.run(scenario())


def test_cancelled_push_wait_is_observed_and_does_not_send_the_remaining_tail():
    async def scenario():
        clock = Clock()
        entered = asyncio.Event()

        async def push(samples):
            entered.set()
            await asyncio.Event().wait()

        player = PCMPlayer(100, push, SpeechPlayback(clock=clock), clock=clock, sleep=clock.sleep,
                           diagnostics=True)
        task = asyncio.create_task(player.play(list(range(8))))
        await entered.wait()
        clock.now = .2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert player.metrics["push_count"] == 1
        assert player.metrics["pushed_frames"] == 0
        assert player.metrics["push_wait_ms_max"] == pytest.approx(200)
        assert player.metrics["play_wall_s"] == pytest.approx(.2)

    asyncio.run(scenario())
