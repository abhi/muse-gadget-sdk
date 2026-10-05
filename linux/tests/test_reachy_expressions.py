# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import pytest

from musegadget.reachy.expressions import Expression


@pytest.mark.parametrize("name, expected", [
    ("nod", Expression.NOD),
    ("thinking", Expression.THINKING),
    ("wave", None),
    ("NOD", None),
    (None, None),
])
def test_parsing_an_expression_name_never_raises(name, expected):
    assert Expression.parse(name) is expected
