# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""How Reachy is described to Muse: its body, its session settings and the reply format to use."""

from __future__ import annotations

from musegadget.reachy.replies import ReplyStyle

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


def setup_challenge_reply(nonce: str) -> str:
    """The exact reply that answers the side-chat setup challenge for ``nonce``."""
    return f"Ready {nonce}"


def with_setup_challenge(setup_message: str, nonce: str) -> str:
    """The side-chat setup message with the challenge that Muse answers with setup_challenge_reply(nonce)."""
    return f"{setup_message}\n\nInitialization challenge: reply exactly {setup_challenge_reply(nonce)}"


STOP_REQUEST = "Please stop working on my previous request."

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


PLAIN_VOICE_CONTEXT = REACHY_CAPABILITIES + (
    "Answer in one to three short, plain spoken sentences, leading with the answer. "
    "Use no markup, lists, emoji, JSON or expression markers. "
    "Check time-sensitive facts with available tools and say plainly what you could not verify. "
    "These instructions replace earlier Reachy conversation instructions."
)

_CONTEXTS = {
    ReplyStyle.MUSE_VOICE: VOICE_CONTEXT,
    ReplyStyle.MARKER: VOICE_CONTEXT,
    ReplyStyle.EXPRESSIVE_JSON: STREAM_VOICE_CONTEXT,
    ReplyStyle.PLAIN_SHORT: PLAIN_VOICE_CONTEXT,
}


def voice_context(style: ReplyStyle, *, motion_enabled: bool = True,
                  antenna_mode: str = "both", face_tracking_enabled: bool = False) -> str:
    """Describe the configured robot and the reply format for one chat."""
    if antenna_mode not in ("both", "left", "right", "none"):
        raise ValueError("invalid Reachy antenna mode")
    context = _CONTEXTS[style]
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


def spoken_request(context: str, text: str) -> str:
    """A spoken request that carries the voice context first, for a chat that last heard another style."""
    return context + "\n\nThe user's spoken request is: " + text


SETUP_REQUEST = {
    ReplyStyle.EXPRESSIVE_JSON: (" For setup, return one sentence frame with text 'Ready to talk' and expression "
                                 "'nod'. Keep using this sentence protocol for subsequent spoken messages."),
    ReplyStyle.MARKER: (" For setup, say 'Ready to talk' and append [reachy:nod]. "
                        "Keep using this expression channel for subsequent spoken messages."),
    ReplyStyle.PLAIN_SHORT: " For setup, say 'Ready to talk'. Keep this plain style for subsequent spoken messages.",
}
