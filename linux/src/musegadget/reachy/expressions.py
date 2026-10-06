# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
