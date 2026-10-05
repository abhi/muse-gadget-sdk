# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""An expression channel for Muse runtimes without custom device tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

REACHY_CAPABILITIES = (
    "You are Muse speaking through Reachy Mini, a tabletop robot with a microphone, "
    "speaker, a head with six degrees of freedom, a rotating body, and two independently "
    "movable antennas. Its eyes are fixed; it has no arms and cannot walk. "
    "The adapter does not send camera images and performs your expressions without "
    "requiring robot tools. "
)

_CONVERSATION_CONTEXT = REACHY_CAPABILITIES + (
    "Answer the user's request naturally, directly, and fully, with concrete details that "
    "help them decide or act. For recommendations, lead with your best-fitting choice and "
    "explain why. Check time-sensitive facts with available tools and be clear "
    "about what you could not verify. "
    "These instructions replace earlier Reachy conversation instructions. "
)

VOICE_CONTEXT = _CONVERSATION_CONTEXT + (
    "Append one expression marker [reachy:NAME] to your answer. Choose NAME "
    "from neutral, happy, sad, surprised, curious, nod, shake, listening, thinking. "
    "Match the expression to your response or the requested movement."
)

_MARKER = re.compile(r"\[reachy:([a-zA-Z0-9_-]{1,80})\]")
_EXPRESSIONS = frozenset(("neutral", "happy", "sad", "surprised", "curious", "nod",
                          "shake", "listening", "thinking"))
STREAM_VOICE_CONTEXT = _CONVERSATION_CONTEXT + (
    "Output only one JSON object per line with text and expression fields, plus optional kind. "
    "Each text is one complete conversational sentence; stream each sentence as it is ready. "
    "Answer the parts you can address now while other parts are still being checked. "
    "Set expression to neutral, happy, sad, surprised, curious, nod, shake, listening, "
    "thinking, or null, matching the sentence or requested movement. "
    "Omit kind for answers, or set it to answer. For an optional brief public update "
    "about an action you actually took, use kind progress and expression thinking or null. "
    "Progress must exclude hidden reasoning, raw tool arguments, credentials, and private data."
)


def voice_context(*, stream_replies: bool, motion_enabled: bool = True,
                  antenna_mode: str = "both", face_tracking_enabled: bool = False) -> str:
    """Describe the configured robot and the reply format for one chat."""
    if antenna_mode not in ("both", "left", "right", "none"):
        raise ValueError("invalid Reachy antenna mode")
    context = STREAM_VOICE_CONTEXT if stream_replies else VOICE_CONTEXT
    if not motion_enabled:
        context += " Movement is disabled in this session; do not promise to perform a movement."
    else:
        context += {
            "both": " Both antennas are enabled in this session.",
            "left": " Only the left antenna is enabled in this session.",
            "right": " Only the right antenna is enabled in this session.",
            "none": " Antenna movement is disabled in this session.",
        }[antenna_mode]
    if face_tracking_enabled:
        context += (" Local face tracking is enabled; camera images stay on Reachy "
                    "and provide you no visual information or identity recognition.")
    return context

_CONTROL = re.compile(r"\[reachy:[^\[\]\r\n]*\]", re.IGNORECASE)
_TOOL_CONTROL = re.compile(r"<\s*/?\s*(?:atem:|tool_call\b|function_calls\b|invoke\b|think\b|analysis\b)",
                           re.IGNORECASE)
_ABBREVIATIONS = frozenset((
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "a.m", "p.m", "fig", "no", "inc", "dept", "approx", "cf", "al",
))
_MAX_REPLY_BYTES = 65536
_MAX_FRAME_BYTES = 8192
_MAX_TEXT_BYTES = 2048


class ReplyProtocolError(ValueError):
    """A reply violates the speech-stream format or its size limits."""


class ReplyRevisionError(ValueError):
    """An authoritative reply changes content that has already been spoken."""


@dataclass(frozen=True)
class SpokenSentence:
    text: str
    expression: str | None
    kind: str = "answer"


def _check_reply_size(text: str) -> None:
    if not isinstance(text, str):
        raise ReplyProtocolError("Invalid speech reply.")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ReplyProtocolError("Invalid speech reply.") from exc
    if size > _MAX_REPLY_BYTES:
        raise ReplyProtocolError("Speech reply exceeds the size limit.")


def _strip_speech_controls(text: str) -> str:
    clean = _CONTROL.sub("", text)
    dangling = re.search(r"\[reachy:", clean, re.IGNORECASE)
    if dangling:
        return clean[:dangling.start()]
    for length in range(1, len("[reachy:")):
        if clean.lower().endswith("[reachy:"[:length]):
            return clean[:-length]
    return clean


def _plain_speech(text: str) -> tuple[str, str | None]:
    if _TOOL_CONTROL.search(text):
        raise ReplyProtocolError("Tool control markup appeared in a speech reply.")
    if re.search(r'\{\s*"', text):
        raise ReplyProtocolError("Speech frames appeared inside a text reply.")
    if "```" in text:
        raise ReplyProtocolError("A speech reply fence appeared inside a text reply.")
    markers = _MARKER.findall(text)
    expression = markers[-1] if markers and markers[-1] in _EXPRESSIONS else None
    clean = _strip_speech_controls(text)
    opening = re.search(r"\{\s*$", clean)
    if opening:
        clean = clean[:opening.start()]
    clean = clean.rstrip("`")
    return clean, expression


def _select_mode(text: str, *, final: bool) -> tuple[str, str, bool]:
    content = text.lstrip()
    if not content:
        return ("text" if final else "undecided"), text, False
    if content.startswith("`"):
        if "```".startswith(content) and not final:
            return "undecided", text, False
        if content.startswith("```"):
            header, newline, rest = content.partition("\n")
            if not newline and not final:
                return "undecided", text, False
            if not newline or header.rstrip("\r ") not in ("```", "```json", "```ndjson"):
                raise ReplyProtocolError("Invalid speech reply fence.")
            return "protocol", rest, True
    if content.startswith("{"):
        return "protocol", content, False
    if content.startswith("["):
        if "[reachy:".startswith(content):
            return ("text" if final else "undecided"), text, False
        if not content.startswith("[reachy:"):
            raise ReplyProtocolError("Speech frames must be JSON objects.")
    return "text", text, False


def _object_is_closed(text: str) -> bool:
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth <= 0:
                return True
    return False


def _speech_frame(value, raw: str) -> SpokenSentence:
    if len(raw.encode("utf-8")) > _MAX_FRAME_BYTES:
        raise ReplyProtocolError("Speech frame exceeds the size limit.")
    if not isinstance(value, dict) or set(value) not in (
        {"text", "expression"}, {"text", "expression", "kind"}
    ):
        raise ReplyProtocolError("Invalid speech frame fields.")
    text, expression = value["text"], value["expression"]
    kind = value.get("kind", "answer")
    if kind not in ("answer", "progress"):
        raise ReplyProtocolError("Invalid speech frame kind.")
    if not isinstance(text, str):
        raise ReplyProtocolError("Invalid speech frame text.")
    if _TOOL_CONTROL.search(text):
        raise ReplyProtocolError("Tool control markup appeared in a speech reply.")
    try:
        text_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ReplyProtocolError("Invalid speech frame text.") from exc
    if text_size > (_MAX_TEXT_BYTES if kind == "answer" else 240):
        raise ReplyProtocolError("Invalid speech frame text.")
    if expression is not None and (not isinstance(expression, str) or expression not in _EXPRESSIONS):
        raise ReplyProtocolError("Invalid speech frame expression.")
    text = _strip_speech_controls(text)
    if kind == "progress" and (
        not text.strip() or expression not in (None, "thinking")
    ):
        raise ReplyProtocolError("Invalid progress speech frame.")
    if kind == "answer" and not text.strip() and expression is None:
        raise ReplyProtocolError("Empty speech frame.")
    return SpokenSentence(text, expression, kind)


def _json_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ReplyProtocolError("Duplicate speech frame fields.")
    return value


def _protocol_frames(pending: str, *, final: bool, fenced: bool, closed: bool):
    frames = []
    decoder = json.JSONDecoder(object_pairs_hook=_json_object)
    while True:
        pending = pending.lstrip()
        if not pending:
            if final and fenced and not closed:
                raise ReplyProtocolError("Unclosed speech reply fence.")
            return frames, pending, closed
        if closed:
            raise ReplyProtocolError("Content follows the speech reply fence.")
        if fenced and pending.startswith("`"):
            if "```".startswith(pending) and len(pending) < 3 and not final:
                return frames, pending, closed
            if not pending.startswith("```"):
                raise ReplyProtocolError("Invalid speech reply fence.")
            closed = True
            pending = pending[3:]
            continue
        if not pending.startswith("{"):
            raise ReplyProtocolError("Speech frames must be JSON objects.")
        try:
            value, end = decoder.raw_decode(pending)
        except json.JSONDecodeError as exc:
            if final or _object_is_closed(pending):
                raise ReplyProtocolError("Invalid speech frame JSON.") from exc
            if len(pending.encode("utf-8")) > _MAX_FRAME_BYTES:
                raise ReplyProtocolError("Speech frame exceeds the size limit.")
            return frames, pending, closed
        frames.append(_speech_frame(value, pending[:end]))
        pending = pending[end:]


def _sentence_boundary(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        punctuation = text[index]
        if punctuation not in ".!?":
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?\"'”’)]":
            end += 1
        if end >= len(text) or not text[end].isspace():
            continue
        if any(char in "\"'”’" for char in text[index + 1:end]):
            following = text[end:].lstrip()
            if not following or following[0].islower():
                continue
        prefix = text[start:index + 1]
        if prefix.count("`") % 2:
            continue
        token = prefix.split()[-1]
        if "://" in token or token.startswith("www.") or "@" in token:
            continue
        if punctuation == ".":
            if (index > start and text[index - 1] == ".") or (index + 1 < len(text) and text[index + 1] == "."):
                continue
            match = re.search(r"([A-Za-z][A-Za-z.]*|\d+)\.$", prefix)
            word = match.group(1) if match else ""
            if (word.lower() in _ABBREVIATIONS or word.isdigit()
                    or (len(word) == 1 and word.isalpha())
                    or re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])+", word)):
                continue
            if re.search(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.$", token):
                continue
        return end
    return None


class SentenceStream:
    """Commit complete speech chunks and reconcile an authoritative final reply."""

    def __init__(self) -> None:
        self._input = ""
        self._mode = "undecided"
        self._pending = ""
        self._fenced = self._closed = self._finished = False
        self._committed: list[SpokenSentence] = []
        self._plain_prefix = ""
        self._plain_expression = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def committed(self) -> tuple[SpokenSentence, ...]:
        return tuple(self._committed)

    def feed(self, delta: str) -> list[SpokenSentence]:
        if self._finished:
            raise ReplyProtocolError("Speech reply is already complete.")
        _check_reply_size(delta)
        text = self._input + delta
        _check_reply_size(text)
        self._input = text
        if self._mode == "undecided":
            self._mode, self._pending, self._fenced = _select_mode(text, final=False)
        elif self._mode == "protocol":
            self._pending += delta
        if self._mode == "undecided":
            return []
        if self._mode == "protocol":
            frames, pending, closed = _protocol_frames(
                self._pending, final=False, fenced=self._fenced, closed=self._closed)
            self._pending, self._closed = pending, closed
        else:
            clean, _ = _plain_speech(text)
            frames = []
            while (end := _sentence_boundary(clean, len(self._plain_prefix))) is not None:
                chunk = clean[len(self._plain_prefix):end].strip()
                self._plain_prefix = clean[:end]
                if chunk:
                    frames.append(SpokenSentence(chunk, None))
        self._committed.extend(frame for frame in frames if frame.kind == "answer")
        return frames

    def finish(self, authoritative_final: str) -> list[SpokenSentence]:
        _check_reply_size(authoritative_final)
        mode, pending, fenced = _select_mode(authoritative_final, final=True)
        if self._committed and mode != self._mode:
            raise ReplyRevisionError("Reply changed after speech was committed.")
        if mode == "protocol":
            frames, _, _ = _protocol_frames(pending, final=True, fenced=fenced, closed=False)
            frames = [frame for frame in frames if frame.kind == "answer"]
            count = len(self._committed)
            if len(frames) < count or tuple(frames[:count]) != self.committed:
                raise ReplyRevisionError("Reply changed after speech was committed.")
            emitted = frames[count:]
        else:
            clean, expression = _plain_speech(authoritative_final)
            if not clean.startswith(self._plain_prefix):
                raise ReplyRevisionError("Reply changed after speech was committed.")
            if self._plain_expression is not None and expression != self._plain_expression:
                raise ReplyRevisionError("Reply changed after speech was committed.")
            emitted = []
            cursor = len(self._plain_prefix)
            while (end := _sentence_boundary(clean, cursor)) is not None:
                chunk = clean[cursor:end].strip()
                cursor = end
                if chunk:
                    emitted.append(SpokenSentence(chunk, None))
            tail = clean[cursor:].strip()
            if tail:
                emitted.append(SpokenSentence(tail, None))
            if emitted and expression is not None:
                emitted[-1] = SpokenSentence(emitted[-1].text, expression)
            elif expression is not None and self._plain_expression is None:
                emitted.append(SpokenSentence("", expression))
            self._plain_prefix = clean
            self._plain_expression = expression
        self._mode = mode
        self._finished = True
        self._committed.extend(emitted)
        return emitted


def spoken_reply(text: str) -> tuple[str, str | None]:
    """Remove expression controls from speech and return the last requested move."""
    if _TOOL_CONTROL.search(text):
        raise ReplyProtocolError("Assistant tool controls are not spoken content")
    markers = _MARKER.findall(text)
    expression = markers[-1] if markers and markers[-1] in _EXPRESSIONS else None
    return _MARKER.sub("", text).strip(), expression


def transcript_text(value, message_id: str | None = None) -> str:
    """Read only assistant text content from Muse's structured transcript."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    messages = value.get("messages", [])
    if not isinstance(messages, list):
        return ""
    assistants = [message for message in messages
                  if isinstance(message, dict) and message.get("role") == "assistant"]
    if message_id is not None:
        assistants = [message for message in assistants if message.get("id") == message_id]
    if not assistants:
        return ""
    content = assistants[-1].get("content", [])
    if not isinstance(content, list):
        return ""
    texts = [item["text"] for item in content
             if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)]
    return "\n".join(texts)
