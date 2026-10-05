"""Privacy boundaries for continuous Reachy microphone input."""

import asyncio
from types import SimpleNamespace

import pytest

from test_reachy_duplex_voice import duplex
from test_reachy_voice import FakeHardware, bounded, cancel_task


def test_blocked_playback_resets_keyword_and_resampler_history(duplex, monkeypatch):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    original_resampler = av.AudioResampler
    wake_resamplers = []

    class WakeResampler:
        def __init__(self):
            wake_resamplers.append(self)

        def resample(self, frame):
            return [SimpleNamespace(to_ndarray=frame.to_ndarray)]

    def resampler(*args, **kwargs):
        if kwargs.get("format") == "flt" and kwargs.get("rate") == 16000:
            return WakeResampler()
        return original_resampler(*args, **kwargs)

    monkeypatch.setattr(av, "AudioResampler", resampler)
    monkeypatch.setattr(FakeHardware, "sample_rate", 48000)
    playback_started = asyncio.Event()
    finish_playback = asyncio.Event()

    class Speech:
        async def stream(self, text, rate):
            yield np.zeros(80, np.float32)
            playback_started.set()
            await finish_playback.wait()

    async def run(ctx):
        # Leave an unfinished keyword hypothesis in both the detector and the
        # 48-to-16 kHz streaming converter.
        await ctx.feed((.75, 1))
        assert len(wake_resamplers) == 1

        speaking = asyncio.create_task(
            ctx.conversation._speak(None, text="Public answer.", stream_segment=True)
        )
        try:
            await playback_started.wait()
            await ctx.feed((.4, 1))
            assert len(wake_resamplers) == 2
            finish_playback.set()
            await speaking

            # Commit a read that may have begun while output was blocked, then
            # advance the fake delayed detector. Its pre-playback hypothesis
            # must not survive to open the microphone.
            await ctx.feed((0, 1))
            await ctx.feed((0, 1))
            await ctx.feed((0, 1))
            assert ("listening", "happy") not in ctx.hardware.state_expressions
            assert ctx.records == []
        finally:
            finish_playback.set()
            if not speaking.done():
                await cancel_task(speaking)

    asyncio.run(bounded(duplex(
        run, echo=False, wake=True, speech=Speech(), wake_delay_feeds=2,
    )))


def test_wake_every_turn_queues_invoked_followup_but_keeps_other_speech_local(duplex):
    async def run(ctx):
        ctx.conversation.wake_timeout_s = 0

        await ctx.question(.4)
        assert ctx.records == []

        await ctx.feed((.75, 7), (.1, 16), (0, 6))
        await ctx.entered[0].wait()
        assert ctx.conversation._wake_deadline is None

        # Speech without another invocation remains only in detector pre-roll.
        await ctx.question(.2)
        assert ctx.labels() == [1]

        # A second explicit invocation is accepted while the first Muse turn
        # still owns the backend and output path, then waits in FIFO order.
        await ctx.feed((.75, 7), (.3, 16), (0, 6))
        assert ctx.labels() == [1]
        ctx.release[0].set()
        await ctx.entered[1].wait()
        assert ctx.labels() == [1, 3]
        assert all(24576 not in pcm and 13107 not in pcm for pcm in ctx.records)

    asyncio.run(bounded(duplex(run, wake=True)))
