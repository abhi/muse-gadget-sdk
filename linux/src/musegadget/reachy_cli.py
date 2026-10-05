# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Setup, diagnostics, and startup for the Reachy Mini voice integration."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import functools
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import secrets
import signal
import sys
import time
import uuid

from musegadget import config, identity
from musegadget.reachy_capabilities import Mode


CHAT_STATE_FILE = "reachy-chat.json"
CHAT_SETUP_PREFIX = (
    "This is Reachy Mini's dedicated side chat for spoken conversations. "
    "Apply these instructions to every later spoken request in this chat:\n\n"
)
CHAT_SETUP_SUFFIX = (
    "\n\nWhen work takes time, provide brief, truthful public updates about actions or status "
    "actually associated with the current request. Never expose private reasoning, raw tool "
    "arguments, credentials, or private data. For this initialization response only, ignore "
    "the later spoken-response format, perform no external actions, searches, or tool calls, "
    "and reply only with the exact initialization challenge appended to this message."
)
CHAT_SETUP_MESSAGE = CHAT_SETUP_PREFIX + CHAT_SETUP_SUFFIX
CHAT_SETUP_TIMEOUT_S = 60


class _SetupResponseLost(ConnectionError, RuntimeError):
    """The setup request may have succeeded but its response was not observed."""


class ReachySideChat:
    """Keep one dedicated chat per Muse VM in this robot's protected state."""

    def __init__(self, node_id: str, directory: Path | None = None,
                 setup_message: str = CHAT_SETUP_MESSAGE):
        self.node_id = node_id
        self.directory = directory or config.state_dir()
        if not isinstance(setup_message, str) or not setup_message.strip():
            raise ValueError("Reachy's side-chat setup message is required")
        self.setup_message = setup_message
        self.setup_sha256 = hashlib.sha256(setup_message.encode("utf-8")).hexdigest()

    def _state(self) -> dict:
        try:
            state = json.loads((self.directory / CHAT_STATE_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 2, "device_id": self.node_id, "chats": {}}
        except (OSError, ValueError) as exc:
            raise ValueError("Reachy's saved chat state could not be read") from exc
        if (not isinstance(state, dict) or state.get("version") not in (1, 2)
                or state.get("device_id") != self.node_id
                or not isinstance(state.get("chats"), dict)):
            raise ValueError("Reachy's saved chat state does not match this device")
        for vm_id, chat in state["chats"].items():
            if not isinstance(vm_id, str) or not vm_id or not isinstance(chat, dict):
                raise ValueError("Reachy's saved chat state is invalid")
            try:
                saved_id = chat["session_id"]
                if not isinstance(saved_id, str) or str(uuid.UUID(saved_id)) != saved_id:
                    raise ValueError
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError("Reachy's saved chat ID is invalid") from exc
            if (not isinstance(chat.get("initialized", False), bool)
                    or not isinstance(chat.get("setup_ack", {}), dict)):
                raise ValueError("Reachy's saved chat initialization state is invalid")
            if any(key not in ("message_id", "reply_to_message_id")
                   or not isinstance(value, str) or not value or len(value) > 512
                   for key, value in chat.get("setup_ack", {}).items()):
                raise ValueError("Reachy's saved setup acknowledgement is invalid")
            for key in ("applied_setup_sha256", "pending_setup_sha256"):
                digest = chat.get(key)
                if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                           or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError("Reachy's saved setup digest is invalid")
        state["version"] = 2
        return state

    async def prepare(self, session, vm_id: str) -> str:
        """Create or resume the side chat after this link has registered."""
        from musegadget.link_client import Header, HttpStreamError

        await session.registered.wait()
        if not isinstance(vm_id, str) or not vm_id:
            raise ValueError("a Muse VM ID is required to select Reachy's chat")
        state = self._state()
        chat = state["chats"].get(vm_id)
        if chat is None:
            chat = {"session_id": str(uuid.uuid4()), "initialized": False}
            state["chats"][vm_id] = chat
            # Persist before sending: an interrupted creation retries the same ID.
            config.save_json(CHAT_STATE_FILE, state, self.directory)
        session_id = chat["session_id"]
        body = json.dumps({"session_id": session_id}).encode()
        async with session.stream_http("POST", "/chat/subscribe", body, headers=[
            Header("Content-Type", "application/json"), Header("Accept", "application/x-ndjson"),
        ]) as stream:
            if 200 <= stream.status < 300:
                exists = True
            elif stream.status == 404:
                exists = False
            else:
                raise HttpStreamError(stream.status, "/chat/subscribe")
        if (exists and chat.get("applied_setup_sha256") == self.setup_sha256
                and "pending_setup_sha256" not in chat):
            return session_id
        # A subscription is forward-only in the observed API. Even with a saved
        # acknowledgement, a reply completed during an outage cannot be drained
        # after reconnect, so repeat the identical setup and await its fresh turn.
        chat.update(initialized=False, pending_setup_sha256=self.setup_sha256)
        chat.pop("setup_ack", None)
        config.save_json(CHAT_STATE_FILE, state, self.directory)
        try:
            setup_ack = await self._apply_setup(session, session_id, subscribe_first=exists)
        except _SetupResponseLost:
            if exists:
                raise
            # A new chat cannot be subscribed before its creating POST. If its
            # response won that race, retry with a fresh challenge after the chat exists.
            setup_ack = await self._apply_setup(session, session_id, subscribe_first=True)
        chat["setup_ack"] = setup_ack
        config.save_json(CHAT_STATE_FILE, state, self.directory)
        self._mark_applied(chat)
        config.save_json(CHAT_STATE_FILE, state, self.directory)
        logging.getLogger(__name__).info("Reachy's dedicated Muse side chat is ready")
        return session_id

    async def _apply_setup(self, session, session_id: str, *, subscribe_first: bool) -> dict:
        nonce = secrets.token_hex(16)
        expected = f"Ready {nonce}"
        message = f"{self.setup_message}\n\nInitialization challenge: reply exactly {expected}"
        armed = asyncio.Event()
        drain = None
        if subscribe_first:
            drain = asyncio.create_task(self._drain_setup(
                session.subscribe_chat(session_id), session_id, expected, armed))
            subscribed = asyncio.create_task(session.chat_subscribed.wait())
            try:
                done, _ = await asyncio.wait(
                    {subscribed, drain}, timeout=CHAT_SETUP_TIMEOUT_S,
                    return_when=asyncio.FIRST_COMPLETED)
                if drain in done:
                    drain.result()
                if subscribed not in done:
                    raise TimeoutError("Muse did not open Reachy's side-chat setup subscription")
            except BaseException:
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
                raise
            finally:
                subscribed.cancel()
                await asyncio.gather(subscribed, return_exceptions=True)
        armed.set()
        try:
            response = await session.send_chat(message, session_id)
            if not response.get("ok"):
                raise RuntimeError("Muse could not create Reachy's dedicated side chat")
            ack = response.get("response")
            if isinstance(ack, dict):
                ack = ack.get("result", ack)
            if (not isinstance(ack, dict) or ack.get("session_id") != session_id
                    or ack.get("is_thread") is not True):
                raise RuntimeError("Muse did not confirm Reachy's dedicated side chat")
            saved_ack = {key: ack[key] for key in ("message_id", "reply_to_message_id")
                         if isinstance(ack.get(key), str) and ack[key]}
            if not saved_ack:
                raise RuntimeError("Muse did not identify Reachy's side-chat setup turn")
            if drain is None:
                drain = asyncio.create_task(self._drain_setup(
                    session.subscribe_chat(session_id), session_id, expected, armed))
            await drain
            return saved_ack
        except BaseException:
            if drain is not None and not drain.done():
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
            raise

    def _mark_applied(self, chat: dict) -> None:
        chat["initialized"] = True
        chat["applied_setup_sha256"] = self.setup_sha256
        chat.pop("pending_setup_sha256", None)
        chat.pop("setup_ack", None)

    @staticmethod
    async def _drain_setup(subscription, session_id: str, expected: str,
                           armed: asyncio.Event) -> None:
        """Consume only the fresh challenge response before microphone input."""
        deadline = time.monotonic() + CHAT_SETUP_TIMEOUT_S
        messages = {}
        running_tasks = set()
        completed_task = False
        challenge_seen = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _SetupResponseLost("Muse did not finish setting up Reachy's side chat")
                try:
                    event = await asyncio.wait_for(subscription.__anext__(), remaining)
                except asyncio.TimeoutError as exc:
                    raise _SetupResponseLost(
                        "Muse did not finish setting up Reachy's side chat") from exc
                except StopAsyncIteration as exc:
                    raise _SetupResponseLost(
                        "Muse's side-chat setup subscription ended") from exc
                if not armed.is_set():
                    continue
                payload = event.get("payload", {})
                if (not isinstance(payload, dict)
                        or payload.get("session_id", event.get("session_id")) != session_id):
                    continue
                name = event.get("event")
                status = payload.get("status")
                task_id = payload.get("task_id")
                if name == "task.status" and isinstance(task_id, str) and task_id:
                    if status == "running":
                        running_tasks.add(task_id)
                    elif task_id in running_tasks and status in (
                            "errored", "failed", "cancelled", "canceled"):
                        raise RuntimeError("Muse did not apply Reachy's side-chat setup")
                    elif task_id in running_tasks and status == "completed":
                        running_tasks.discard(task_id)
                        completed_task = True
                    if challenge_seen and completed_task and not running_tasks:
                        return
                    continue
                if name not in ("delta.message_start", "delta.text_append",
                                "delta.message_done", "message.assistant"):
                    continue
                message_id = payload.get("message_id") or payload.get("id")
                if (not isinstance(message_id, str) or not message_id
                        or payload.get("role") == "user"):
                    continue
                if name == "delta.message_start":
                    if len(messages) >= 32 and message_id not in messages:
                        raise RuntimeError("Muse returned too many side-chat setup messages")
                    messages.setdefault(message_id, "")
                    continue
                if message_id not in messages and name == "delta.text_append":
                    continue
                if len(messages) >= 32 and message_id not in messages:
                    raise RuntimeError("Muse returned too many side-chat setup messages")
                messages.setdefault(message_id, "")
                if name == "delta.text_append" and isinstance(payload.get("text"), str):
                    messages[message_id] += payload["text"]
                elif name in ("delta.message_done", "message.assistant"):
                    text = next((payload[key] for key in ("display_text", "content", "transcript")
                                 if isinstance(payload.get(key), str)), messages[message_id])
                    messages[message_id] = text
                    if text.strip() == expected:
                        return
                if sum(len(text.encode("utf-8")) for text in messages.values()) > 4096:
                    raise RuntimeError("Muse returned too much side-chat setup text")
                challenge_seen = any(text.strip() == expected for text in messages.values())
                if challenge_seen and completed_task and not running_tasks:
                    return
        finally:
            await subscription.aclose()


@dataclass(frozen=True)
class ModeFlags:
    requires: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()


MODE_FLAGS = {
    Mode.MUSE_VOICE: ModeFlags(forbids=("--stt-model", "--stt-backend", "--tts-model",
                                        "--stream-replies", "--wake-model")),
    Mode.ON_ROBOT: ModeFlags(requires=("--stt-model", "--tts-model")),
}


def mode_problems(args) -> list[str]:
    def given(flag):
        return getattr(args, flag[2:].replace("-", "_")) not in (None, False)

    mode = Mode(args.mode)
    rule = MODE_FLAGS[mode]
    problems = [f"--mode {mode.value} needs {flag}" for flag in rule.requires if not given(flag)]
    for flag in rule.forbids:
        if given(flag):
            allowed = " or ".join(other.value for other, flags in MODE_FLAGS.items()
                                  if flag not in flags.forbids)
            problems.append(f"{flag} needs --mode {allowed}")
    return problems


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Talk to Muse through Reachy Mini")
    p.add_argument("--state-dir", type=Path,
                   default=Path.home() / ".local/share/musegadget-reachy")
    p.add_argument("--sdk-token-file", type=Path,
                   help="SDK token file, read without printing its contents")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair", help="Open Bluetooth setup in the Muse phone app")
    pair.add_argument("--force", action="store_true")
    pair.add_argument("--timeout", type=int, default=600)
    for command in ("run", "doctor"):
        item = sub.add_parser(command, help="Start conversations" if command == "run"
                              else "Check the robot and audio without a Muse connection")
        item.add_argument("--host", default="localhost")
        item.add_argument("--port", type=int, default=8000)
        item.add_argument("--media-backend", default="local",
                          choices=("local", "webrtc", "default"))
        item.add_argument("--no-motion", action="store_true")
        item.add_argument("--antenna-mode", choices=("both", "left", "right", "none"), default="both",
                          help="Antennas to animate; other antenna targets stay at their measured startup positions")
        if command == "run":
            item.add_argument("--mode", choices=tuple(mode.value for mode in MODE_FLAGS),
                              default=Mode.MUSE_VOICE.value,
                              help="muse-voice: Muse hears and speaks (default). "
                                   "on-robot: Reachy's own speech models; needs --stt-model and --tts-model")
            item.add_argument("--face-follow", action="store_true",
                              help="Follow a nearby face locally during conversations; camera frames stay on the robot")
            item.add_argument("--face-follow-model", type=Path,
                              help="YuNet ONNX detector override; requires --face-follow")
            item.add_argument("--stt-model", type=Path,
                              help="Local model directory for the selected speech recognition backend")
            item.add_argument("--stt-backend", choices=("whisper", "moonshine", "sherpa-streaming"),
                              help="Local speech recognizer; whisper when omitted")
            item.add_argument("--tts-model", type=Path,
                              help="Piper ONNX model for speaking normal Muse answers locally")
            item.add_argument("--stream-replies", action="store_true",
                              help="Speak incoming Muse chunks with per-chunk expressions")
            item.add_argument("--audio-diagnostics", action="store_true",
                              help="Log playback timing counters without audio or transcript content")
            item.add_argument("--wake-model", type=Path,
                              help="Local model directory for the selected wake backend; enables wake-word invocation")
            item.add_argument("--wake-backend", choices=("sherpa", "vosk"), default="sherpa",
                              help="Local wake detector (default: sherpa); Vosk requires the reachy-wake-vosk extra")
            item.add_argument("--wake-phrase", default="hey muse",
                              help="Phrase that opens conversation when --wake-model is set (default: hey muse)")
            item.add_argument("--wake-timeout", type=float, default=10.0,
                              help="Idle seconds before returning to wake-word detection; excludes thinking and speech")
            item.add_argument("--session-id", help="Existing Muse chat to use instead of Reachy's dedicated side chat")
            item.add_argument("--main-chat", action="store_true", help="Use the main Muse chat instead of Reachy's dedicated side chat")
            item.add_argument("--silence", type=float, default=2.0,
                              help="Seconds of silence that end a spoken turn (default: 2)")
        else:
            item.add_argument("--exercise", action="store_true",
                              help="Capture microphone audio, play a tone, and exercise motion")
    return p


def _hardware(args, *, motion=True):
    from musegadget.reachy_hardware import ReachyController
    from musegadget.local_face_tracking import default_face_model
    face_model = None
    if getattr(args, "face_follow", False) and motion and not args.no_motion:
        face_model = args.face_follow_model or default_face_model()
    return ReachyController(host=args.host, port=args.port,
                           connection_mode="localhost_only" if args.host in ("localhost", "127.0.0.1")
                           else "network", media_backend=args.media_backend,
                           enable_motion=motion and not args.no_motion,
                           antenna_mode=args.antenna_mode, face_follow_model=face_model)


async def run(args) -> int:
    from musegadget.reachy_local_backends import backends_for
    from musegadget.reachy_voice import ReachyService

    if not config.load_json(config.PAIRING_FILE):
        print("Reachy is not paired with Muse. Run `muse-reachy pair` on the robot first.",
              file=sys.stderr)
        return 1
    if args.main_chat and args.session_id:
        raise ValueError("choose either --main-chat or --session-id")
    if args.face_follow_model is not None and not args.face_follow:
        raise ValueError("--face-follow-model requires --face-follow")
    if args.face_follow and args.no_motion:
        raise ValueError("--face-follow requires motion to be enabled")
    if not 0.3 <= args.silence <= 3:
        raise ValueError("--silence must be between 0.3 and 3 seconds")
    if args.wake_backend == "vosk" and args.wake_model is None:
        raise ValueError("--wake-backend vosk requires --wake-model")
    if not math.isfinite(args.wake_timeout) or not 1 <= args.wake_timeout <= 300:
        raise ValueError("--wake-timeout must be between 1 and 300 seconds")
    ident = identity.load_or_create()
    session_id = None if args.main_chat else args.session_id
    robot = _hardware(args)
    dedicated_chat = None
    if not args.main_chat and not args.session_id:
        from musegadget.reachy_capabilities import ReplyStyle
        from musegadget.reachy_expression import voice_context
        context = voice_context(
            ReplyStyle.EXPRESSIVE_JSON if args.stream_replies else ReplyStyle.MARKER,
            motion_enabled=getattr(robot, "motion_enabled", True),
            antenna_mode=getattr(robot, "antenna_mode", "both"),
            face_tracking_enabled=getattr(robot, "face_tracking_enabled", False),
        )
        dedicated_chat = ReachySideChat(
            ident.node_id, setup_message=CHAT_SETUP_PREFIX + context + CHAT_SETUP_SUFFIX)
    speech = None
    progress_speech = None
    transcriber = None
    wake_detector = None
    speech_gate = None
    try:
        if args.tts_model is not None:
            from musegadget.local_speech import PiperSpeech
            speech = PiperSpeech(args.tts_model)
            progress_speech = PiperSpeech(args.tts_model)
        if args.stt_model is not None:
            from musegadget.speech_gate import SileroSpeechGate
            speech_gate = await asyncio.to_thread(SileroSpeechGate)
            logging.getLogger(__name__).info("Reachy speech gate ready: Silero")
            if args.stt_backend == "sherpa-streaming":
                from musegadget.streaming_transcription import SherpaStreamingTranscriber
                transcriber = SherpaStreamingTranscriber(args.stt_model)
            elif args.stt_backend == "moonshine":
                from musegadget.local_transcription import MoonshineTranscriber
                transcriber = MoonshineTranscriber(args.stt_model)
            else:
                from musegadget.local_transcription import WhisperTranscriber
                transcriber = WhisperTranscriber(args.stt_model)
        if args.wake_model is not None:
            if args.wake_backend == "vosk":
                from musegadget.vosk_wake import VoskWakeWordDetector
                wake_factory = VoskWakeWordDetector
            else:
                from musegadget.wake_word import WakeWordDetector
                wake_factory = WakeWordDetector
            wake_detector = await asyncio.to_thread(wake_factory, args.wake_model,
                                                   phrase=args.wake_phrase)
            if getattr(wake_detector, "timing_supported", False):
                logging.getLogger(__name__).info("Reachy timed wake handoff ready")
        service = ReachyService(identity=ident, executor=robot, sdk_token=config.sdk_token(),
                               display_name="Reachy Mini", session_id=session_id,
                               prepare_chat=dedicated_chat.prepare if dedicated_chat else None,
                               owns_chat=dedicated_chat is not None,
                               silence_s=args.silence,
                               backends=functools.partial(backends_for, Mode(args.mode), speech=speech,
                                                          progress_speech=progress_speech,
                                                          transcriber=transcriber,
                                                          stream_replies=args.stream_replies,
                                                          wake=wake_detector is not None),
                               audio_diagnostics=args.audio_diagnostics, wake_detector=wake_detector,
                               wake_timeout_s=args.wake_timeout, speech_gate=speech_gate)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, service.stop)
        await asyncio.to_thread(robot.start)
        if speech is not None:
            await asyncio.wait_for(speech.start(), 30)
        if progress_speech is not None:
            await asyncio.wait_for(progress_speech.start(), 30)
        if transcriber is not None:
            await asyncio.wait_for(transcriber.start(), 30)
        logging.getLogger(__name__).info("Using Reachy's microphone and speaker; chat %s",
                                         "Reachy's dedicated side chat" if dedicated_chat else session_id or "main")
        await service.run()
    finally:
        try:
            await asyncio.to_thread(robot.close)
        finally:
            try:
                if speech is not None:
                    await speech.close()
            finally:
                try:
                    if progress_speech is not None:
                        await progress_speech.close()
                finally:
                    try:
                        if transcriber is not None:
                            await transcriber.close()
                    finally:
                        if wake_detector is not None:
                            await asyncio.to_thread(wake_detector.close)
    return 0


def doctor(args) -> int:
    import numpy as np

    robot = _hardware(args, motion=args.exercise)
    report = {"host": args.host, "paired": bool(config.load_json(config.PAIRING_FILE)),
              "sdk_token_present": bool(config.sdk_token()), "audio_and_motion_exercised": False}
    try:
        robot.start()
        report.update(input_rate=robot.sample_rate, output_rate=robot.output_sample_rate)
        if args.exercise:
            drain_until = time.monotonic() + 0.25
            for _ in range(1000):
                if time.monotonic() >= drain_until or robot.read_audio() is None:
                    break
            deadline = time.monotonic() + 3
            count = 0
            energy = 0.0
            peak = 0.0
            while time.monotonic() < deadline:
                sample = robot.read_audio()
                if sample is None:
                    time.sleep(0.01)
                    continue
                count += len(sample)
                energy += float(np.dot(sample, sample))
                peak = max(peak, float(np.max(np.abs(sample))))
            if not count:
                raise RuntimeError("Reachy's microphone returned no samples")
            report.update(microphone_samples=count, microphone_rms=(energy / count) ** 0.5,
                          microphone_peak=peak)
            robot.set_state("speaking")
            tone = np.arange(robot.output_sample_rate // 2, dtype=np.float32)
            tone = (0.06 * np.sin(2 * np.pi * 440 * tone / robot.output_sample_rate)).astype(np.float32)
            ramp = min(len(tone) // 2, robot.output_sample_rate // 50)
            tone[:ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)
            tone[-ramp:] *= np.linspace(1, 0, ramp, dtype=np.float32)
            chunk_size = robot.output_sample_rate // 50
            for offset in range(0, len(tone), chunk_size):
                robot.play_audio(tone[offset:offset + chunk_size])
                time.sleep(0.02)
            time.sleep(0.2)
            robot.set_state("listening")
            time.sleep(0.8)
            robot.set_state("thinking")
            time.sleep(0.8)
            robot.set_state("idle")
            report.update(audio_and_motion_exercised=True, speaker_test="440 Hz tone sent to Reachy",
                          motion_test="exercised" if not args.no_motion else "disabled")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        robot.close()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    problems = mode_problems(args) if args.command == "run" else []
    for problem in problems:
        print(f"Reachy: {problem}", file=sys.stderr)
    if problems:
        return 2
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Third-party debug logs may include bearer headers or conversation text.
    for name in ("websockets", "urllib3", "paramiko", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ[config.STATE_DIR_ENV] = str(args.state_dir.expanduser())
    try:
        if args.sdk_token_file:
            os.environ[config.SDK_TOKEN_ENV] = args.sdk_token_file.expanduser().read_text().strip()
            config.sdk_token()
        if args.command == "pair":
            from musegadget.cli import cmd_pair
            return cmd_pair(args)
        if args.command == "doctor":
            return doctor(args)
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Reachy: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
