# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""What the user's words mean for Reachy's own replies: follow-ups while Muse works, and small talk.

Pure, stdlib-only rules: no model decides whether a sentence cancels a request.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple


class FollowUp(Enum):
    NEW = "new"                # a separate request, queued behind the running one
    ADD_DETAIL = "add_detail"  # more about the newest request, sent on the same chat
    CANCEL = "cancel"          # drop the newest request
    SILENCE = "silence"        # stop the answer being spoken; a cancel when none is


_SILENCE_PHRASES: Tuple[Tuple[str, ...], ...] = (("stop", "talking"), ("shut", "up"), ("be", "quiet"))
_CANCEL_PHRASES: Tuple[Tuple[str, ...], ...] = (
    ("never", "mind"), ("nevermind",), ("cancel",), ("stop", "stop"), ("stop",),
    ("forget", "it"), ("forget", "that"), ("forget", "about"), ("scratch", "that"),
    ("don't", "bother"), ("dont", "bother"))
# Words that may surround a cancel phrase without turning it into a request ("stop the music" is one).
_CANCEL_LEAD = frozenset(("oh", "okay", "ok", "no", "wait", "actually", "um", "uh", "hmm", "please"))
_CANCEL_TAIL = frozenset(("that", "it", "this", "the", "my", "request", "question", "please",
                          "thanks", "thank", "you", "now"))
_DETAIL_LEADS: Tuple[Tuple[str, ...], ...] = (("oh", "and"), ("also",), ("and",), ("actually",), ("plus",))
_DETAIL_FILLER = frozenset(("oh", "and", "also", "actually", "plus", "please", "um", "uh", "so"))
# After a lead, these open a whole question or request ("and what's the weather in Paris?"),
# not a change to the running one ("and make it vegetarian").
_REQUEST_OPENERS = frozenset((
    "what", "what's", "whats", "when", "when's", "where", "where's", "who", "who's", "whose", "why",
    "how", "how's", "which", "is", "are", "was", "were", "can", "could", "would", "will", "does", "did",
    "should", "shall", "may", "remind", "tell", "set", "call", "text", "send", "email", "message", "play",
    "search", "find", "look", "show", "open", "turn", "start", "schedule", "read", "translate", "explain"))
_TOPIC_LEAD = frozenset((
    "hey", "muse", "can", "could", "would", "will", "you", "please", "tell", "me", "about",
    "what", "what's", "whats", "how", "does", "do", "did", "is", "are", "search", "find", "look",
    "up", "check", "get", "give", "show", "for", "explain", "i", "want", "to", "know"))
TOPIC_WORDS = 6


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.replace("’", "'").casefold())


def classify_follow_up(text: str) -> FollowUp:
    words = _words(text)
    rest = words
    while rest and rest[0] in _CANCEL_LEAD:
        rest = rest[1:]
    for kind, phrases in ((FollowUp.SILENCE, _SILENCE_PHRASES), (FollowUp.CANCEL, _CANCEL_PHRASES)):
        for phrase in phrases:
            if tuple(rest[:len(phrase)]) == phrase and all(word in _CANCEL_TAIL for word in rest[len(phrase):]):
                return kind
    if any(tuple(words[:len(lead)]) == lead for lead in _DETAIL_LEADS):
        rest = words
        while rest and rest[0] in _DETAIL_FILLER:
            rest = rest[1:]
        return FollowUp.NEW if rest and rest[0] in _REQUEST_OPENERS else FollowUp.ADD_DETAIL
    return FollowUp.NEW


def request_topic(request: str) -> str:
    """A few of the request's own words naming what it was about; "" when it has none."""
    tokens = [token.strip(".,!?;:\"()") for token in request.split()]
    tokens = [token for token in tokens if token]
    start = 0
    while start < len(tokens) and tokens[start].replace("’", "'").casefold() in _TOPIC_LEAD:
        start += 1
    return " ".join(tokens[start:][:TOPIC_WORDS] or tokens[:TOPIC_WORDS])


SMALL_TALK_WORDS = 3
_SMALL_TALK = re.compile(
    r"(?:\s*(?:hi|hello|hey|hey there|good (?:morning|afternoon|evening|night)|muse|reachy|there"
    r"|how are you(?: doing)?(?: today)?|how's it going|how is it going|what's up|whats up"
    r"|nice to meet you|thanks|thank you(?: so much| very much)?|you too|i'm (?:fine|good|great|okay|ok)"
    r"|yes|yeah|yep|no|nope|okay|ok|sure|cool|great|bye|goodbye|see you(?: later)?)\b[\s,.!?]*)+")


def is_small_talk(text: str) -> bool:
    """Greetings, thanks, yes or no, or at most ``SMALL_TALK_WORDS`` words: nothing worth acknowledging."""
    words = _words(text)
    return len(words) <= SMALL_TALK_WORDS or _SMALL_TALK.fullmatch(" ".join(words)) is not None
