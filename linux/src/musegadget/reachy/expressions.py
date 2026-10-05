# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""The expressions Muse can ask Reachy to show."""

from __future__ import annotations

from enum import Enum
from typing import Optional


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
