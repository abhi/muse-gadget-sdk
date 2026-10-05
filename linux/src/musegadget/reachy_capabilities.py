# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np

    from musegadget.reachy_progress import ProgressPlan

    SpeechRole = Literal["ack", "progress", "answer", "notice"]


WAKE_CUE = "Yes?"


class Expression(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    NOD = "nod"
    SHAKE = "shake"
    LISTENING = "listening"
    THINKING = "thinking"

    @classmethod
    def parse(cls, name: object) -> Optional[Expression]:
        """Unknown names mean no gesture; parsing never raises."""
        try:
            return cls(name)
        except ValueError:
            return None


class ReplyStyle(Enum):
    """The Muse prompt a request was sent with, which also picks its reply parser."""

    MUSE_VOICE = "muse_voice"      # Muse speaks; a trailing [reachy:NAME] picks the expression
    MARKER = "marker"              # local speech of a reply with a trailing [reachy:NAME]
    EXPRESSIVE_JSON = "json"       # local speech of one JSON sentence frame per line
    PLAIN_SHORT = "plain"          # one to three plain sentences, condensed and voiced by the companion


class Mode(Enum):
    """How Reachy hears and speaks; each mode is one Backends bundle."""

    MUSE_VOICE = "muse-voice"      # Muse transcribes the turn's audio and speaks its reply
    ON_ROBOT = "on-robot"          # Reachy transcribes and speaks with its own models
    COMPANION = "companion"        # a paired computer hears, speaks and narrates; on-robot is the fallback


@dataclass(frozen=True)
class Partial:
    """One utterance's streaming transcript so far; ``revision`` rises with each change of text."""

    text: str
    revision: int
    utterance_id: int              # a new id for every utterance, even one repeating the last


@dataclass(frozen=True)
class Endpoint:
    """One finished user turn. Without text, Muse receives the audio itself."""

    audio: bytes                   # the turn as a 16 kHz mono PCM16 WAV
    text: Optional[str]            # None: not transcribed here; "": nothing intelligible
    forced: bool = False


@dataclass(frozen=True)
class SpokenLine:
    text: str
    expression: Optional[Expression]
    role: SpeechRole
    muse_message_id: Optional[str] = None


class EndedTurn(Protocol):
    """One user turn whose end the hearing decided; its transcript may still be on the way."""

    truncated: bool                # the turn hit the 60-second recording cap

    def start(self) -> None: ...   # begin recognition now, if it waits for the caller; idempotent
    def cancel(self) -> None: ...
    async def endpoint(self) -> Optional[Endpoint]: ...   # None: its transcription was lost


@dataclass(frozen=True)
class HeardAudio:
    """What one feed told the caller."""

    ended: Optional[EndedTurn] = None   # at most one ended turn per take; feed nothing to drain the next
    failures: int = 0              # recognizer failures; each one owes the user a retry notice
    transcript_lost: bool = False  # an utterance ended without a usable transcript


class HearingTurn(Protocol):
    """One listening window, which may hold several user turns.

    The robot decides when the window opens and closes; the implementation decides where
    each user turn inside it ends. ``feed`` may run off the event loop; ``take`` may not.
    """

    active: bool                   # inside a user turn that has not ended yet
    speech_active: bool
    partial: Optional[Partial]     # the open utterance's transcript so far; never sent to Muse

    def feed(self, samples: np.ndarray) -> None: ...
    def take(self) -> HeardAudio: ...
    def finish_initial_capture(self) -> None: ...
    def abort(self) -> None: ...


class Hearing(Protocol):
    transcribes: bool

    async def start(self) -> None: ...
    def open_turn(self, sample_rate: int, *, silence_s: float, vad: object = None,
                  **options) -> HearingTurn: ...
    async def transcribe(self, wav: bytes) -> str: ...


class PreparedVoice(Protocol):
    """Speech synthesized ahead of playback, as float32 mono at the requested rate."""

    def __aiter__(self) -> AsyncIterator[np.ndarray]: ...
    def cancel(self) -> None: ...
    async def aclose(self) -> None: ...


class Voice(Protocol):
    speaks_text: bool              # False: speaks only Muse's own audio for a Muse message

    async def warm(self, output_rate: int) -> None: ...
    def stream(self, line: SpokenLine, output_rate: int) -> AsyncIterator[np.ndarray]: ...
    def prepare(self, line: SpokenLine, output_rate: int) -> Optional[PreparedVoice]: ...


class Narrator(Protocol):
    """Restates the request, relays public status and condenses the reply; it never adds facts."""

    async def acknowledge(self, request: str) -> Optional[SpokenLine]: ...
    def progress(self, request: str, started: float) -> ProgressPlan: ...
    async def say_progress(self, request: str, status: str,
                           already_said: Tuple[str, ...]) -> Optional[SpokenLine]: ...
    async def lines(self, request: str, reply: str, style: ReplyStyle, *,
                    message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]: ...


class CapabilityUnavailable(Exception):
    """A companion capability could not answer; the failover wrappers answer locally instead.

    ``link_down`` is False when only this operation failed and the link itself is healthy.
    """

    def __init__(self, reason: str, *, link_down: bool = True):
        super().__init__(reason)
        self.link_down = link_down


class LinkHealth(Protocol):
    def begin_turn(self) -> bool: ...    # True: the companion voices and narrates the turn starting now
    def take_outage(self) -> bool: ...   # True once for each outage worth announcing


COMPANION_NOTICE = "My companion computer isn't answering, so I'll use my own voice for now."


@dataclass(frozen=True)
class Backends:
    hearing: Hearing
    voice: Voice
    narrator: Narrator
    local_style: ReplyStyle        # the style of this mode's on-robot parts
    progress_voice: Optional[Voice] = None
    companion: Optional[LinkHealth] = None

    def reply_style(self) -> ReplyStyle:
        """The style for the next request; read once per turn, at its boundary."""
        if self.companion is not None and self.companion.begin_turn():
            return ReplyStyle.PLAIN_SHORT
        return self.local_style

    def take_notice(self) -> Optional[str]:
        """What Reachy must say about its own parts; once per companion outage."""
        if self.companion is not None and self.companion.take_outage():
            return COMPANION_NOTICE
        return None


_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
          "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_VALUES = {**{word: value for value, word in enumerate(_UNITS)},
           **{word: 10 * tens for tens, word in enumerate(_TENS, start=2)}}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
             "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
             "twelfth": "12th", "twentieth": "20th", "thirtieth": "30th"}
_CURRENCY = {"$": "dollars", "€": "euros", "£": "pounds", "¥": "yen"}
_ONE_AS_PRONOUN = frozenset("the this that which each every no any some next last".split())
_DATES = frozenset("monday tuesday wednesday thursday friday saturday sunday january february march april "
                   "june july august september october november december".split())
_ABBREVIATIONS = frozenset("dr mr mrs ms st prof jr sr mt vs".split())
_ALWAYS_KNOWN = frozenset(("muse", "reachy"))
_COMMON_OPENERS = frozenset("""
i you he she it we they me him her us them my your his its our their this that these those there here
what which who whom whose when where why how a an the some any each every all both either neither no not
none many much more most few several other another such and but or so yet for nor if because since
although though while as after before until unless once then also still just only even now well yes oh
okay ok sure great good thanks thank sorry please hello hi hey alright right at in on of to from with
without by about around over under into onto near between through during across along against is are
was were be been being am do does did have has had can could will would shall should may might must let
lets try check ask take bring catch call go get make keep look wear grab remember note expect plan head
see find tell give use wait meet start stop turn maybe perhaps sounds looks seems unfortunately actually
currently first next last
""".split())


def _alt(words) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


_NUMBER = _alt([*_VALUES, *_SCALES])
_AFTER_SCALE = "|".join(f"(?<={scale})" for scale in _SCALES)
_SPAN = re.compile(rf"\b(?:a[ -](?=(?:{_alt(_SCALES)})\b))?(?:{_NUMBER})"
                   rf"(?:(?:(?:{_AFTER_SCALE})[ -]and[ -]|[ -])(?:{_NUMBER}))*\b", re.I)
_DIGIT = _alt(_UNITS[1:10])
_CLOCK = re.compile(rf"\b({_alt(_UNITS[1:13])})[ -](oh[ -](?:{_DIGIT})|(?:twenty|thirty|forty|fifty)"
                    rf"(?:[ -](?:{_DIGIT}))?|{_alt(_UNITS[10:20])})\b(?![ -](?:{_NUMBER}|and)\b)", re.I)
_ORDINAL = re.compile(rf"\b(?:the[ -]second|{_alt(set(_ORDINALS) - {'second'})})\b(?!,)", re.I)
_TIME = re.compile(r"(?<![\d.])\d{1,2}:\d{2}(?: [ap]m)?")
_AMOUNT = re.compile(r"(?<![\d.])\d+(?:\.\d+)? (?:dollars|euros|pounds|yen|cents|percent"
                     r"|degrees(?: fahrenheit| celsius)?)")
_NUMERAL = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SENTENCE_END = re.compile(r"[.!?:]['\")\]]*$")


def _number(match: re.Match) -> str:
    words = [word for word in re.split(r"[ -]", match[0].casefold()) if word not in ("a", "and")]
    if words == ["one"]:
        before = match.string[:match.start()].casefold().split()[-1:]
        after = match.string[match.end():].casefold().split()[:1]
        if before and before[0] in _ONE_AS_PRONOUN or after and after[0].strip(",.;") in ("of", "another"):
            return match[0]
    total = current = 0
    for word in words:
        if word == "hundred":
            current = max(current, 1) * 100
        elif word in _SCALES:
            total, current = total + max(current, 1) * _SCALES[word], 0
        else:
            current += _VALUES[word]
    return str(total + current)


def _clock(match: re.Match) -> str:
    minute = sum(_VALUES[word] for word in re.split(r"[ -]", match[2].casefold()) if word != "oh")
    return f"{_VALUES[match[1].casefold()]}:{minute:02d}"


def _ordinal(match: re.Match) -> str:
    word = match[0].casefold()
    return "the 2nd" if word.endswith("second") else _ORDINALS[word]


def _canonical(text: str) -> str:
    """One spelling for each figure, case kept: digits, "5 dollars", "72 degrees fahrenheit", "3:00 pm"."""
    text = " ".join(text.split())
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"([$€£¥])\s?(\d+(?:\.\d+)?)", lambda m: f"{m[2]} {_CURRENCY[m[1]]}", text)
    text = re.sub(r"\s?%", " percent", text)
    text = re.sub(r"\s?°\s?([fc])\b", lambda m: " degrees " + ("fahrenheit" if m[1] in "fF" else "celsius"),
                  text, flags=re.I)
    text = re.sub(r"\s?°", " degrees", text)
    text = re.sub(r"\b(dollar|euro|pound|cent|degree)s?\b", r"\1s", text, flags=re.I)
    text = _CLOCK.sub(_clock, text)
    text = _ORDINAL.sub(_ordinal, text)
    text = _SPAN.sub(_number, text)
    text = re.sub(r"(?<![\d:.])(\d{1,2}) o['’]clock\b", r"\1:00", text, flags=re.I)
    text = re.sub(r"(\d)\s?([ap])\.?m\b(\.)?", lambda m: f"{m[1]} {m[2].lower()}m{m[3] or ''}", text, flags=re.I)
    return re.sub(r"(?<![\d:.])(\d{1,2}) ([ap]m)\b", r"\1:00 \2", text)


def _bare(word: str) -> str:
    return re.sub(r"['’]s$", "", word.casefold()).strip("'’-").replace("-", "")


def _covered(found, known) -> bool:
    return all(any(fact == seen or seen.startswith(fact + " ") for seen in known) for fact in found)


def grounded(line: str, *sources: str) -> bool:
    """Whether every checkable fact in ``line`` also appears in some source.

    Facts are numbers, amounts and times, however they are spelled; day and month
    names; and capitalized words, including a sentence's first word unless it is a
    common opener. This catches invented figures and names. It does not prove that the
    line is faithful.
    """
    text = _canonical(line)
    source = " ".join(_canonical(s) for s in sources).casefold()
    folded = text.casefold()
    if not (_covered(_TIME.findall(folded), set(_TIME.findall(source)))
            and _covered(_AMOUNT.findall(folded), set(_AMOUNT.findall(source)))
            and set(_NUMERAL.findall(folded)) <= set(_NUMERAL.findall(source))):
        return False
    words = {_bare(word) for word in _WORD.findall(source)}
    initial = True
    for token in text.split():
        word = _WORD.search(token)
        name = word[0] if word is not None else ""
        bare = _bare(name)
        if bare in _DATES or name[:1].isupper():
            known = (bare in words or bare in _ALWAYS_KNOWN or name == "I" or name.startswith(("I'", "I’"))
                     or (initial and bare in _COMMON_OPENERS))
            if not known:
                return False
        abbreviation = bare in _ABBREVIATIONS or (len(name) == 1 and name.isupper())
        initial = bool(_SENTENCE_END.search(token)) and not abbreviation
    return True
