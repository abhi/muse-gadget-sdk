"""The grounded() guard that checks every line a companion narrates before Reachy speaks it."""

import pytest

from musegadget.reachy_capabilities import grounded

REPLY = ("Lisbon is sunny today with a high of 24 degrees. The 7:30 pm tram costs $3.10, "
         "and the drive to Sintra takes 2 hours.")


@pytest.mark.parametrize("line, expected", [
    ("Sunny in Lisbon, 24 degrees.", True),
    ("The drive takes two hours.", True),
    ("Catch the 7:30 pm tram for $3.10.", True),
    ("It's sunny. Bring a hat.", True),
    ("That one leaves tonight.", True),
    ("One tram leaves tonight.", False),
    ("It took 17 minutes.", False),
    ("Sunny in Porto.", False),
    ("The tram costs $3.50.", False),
    ("Catch the 8:30 pm tram.", False),
    ("The drive takes five hours.", False),
    ("It is 2024 degrees.", False),
    ("Ask Maria at the station.", False),
])
def test_a_line_is_grounded_only_when_its_figures_and_names_come_from_a_source(line, expected):
    assert grounded(line, "What's the weather in Lisbon?", REPLY) is expected


@pytest.mark.parametrize("line, source", [
    ("It's 72 degrees Fahrenheit.", "It is 72°F outside."),
    ("It's 72°F outside.", "It is 72 degrees Fahrenheit."),
    ("Meet at 10:30 am.", "Meet at 10:30am."),
    ("Meet at 10:30am.", "Meet at 10:30 a.m. sharp."),
    ("Meet at 3pm.", "Meet at 3 PM."),
    ("It costs 5 dollars.", "It costs $5."),
    ("It costs $200.", "It costs two hundred dollars."),
    ("About 200 people came.", "About two hundred people came."),
    ("About 1,500 people came.", "About fifteen hundred people came."),
    ("The Wi-Fi is down.", "The WiFi is down."),
    ("The WiFi is down.", "The Wi-Fi is down."),
    ("Lunch is at 1 o'clock.", "Lunch is at one o'clock."),
    ("At 12:30 the bus leaves.", "The bus leaves at twelve thirty."),
    ("I asked Muse about the bus.", "When is the bus?"),
    ("Reachy found it on the third shelf.", "It is on the 3rd shelf."),
])
def test_the_same_fact_written_another_way_is_grounded(line, source):
    assert grounded(line, source) is True


@pytest.mark.parametrize("line, source", [
    ("Dr. Smith said yes.", "Dr. Jones said yes."),
    ("It is 3.5 miles away.", "Take exit 3 and drive 5 miles."),
    ("You have one meeting.", "You have three meetings."),
    ("You have a million emails.", "You have 3 emails."),
    ("It is on the first shelf.", "It is on the third shelf."),
    ("Starbucks opens at nine.", "The cafe opens at 9."),
    ("The cafe opens on monday.", "The cafe opens on Tuesday."),
    ("It's 72 degrees Celsius.", "It is 72°F outside."),
    ("It costs 5 euros.", "It costs $5."),
    ("Meet at 11:00.", "Meet at 1:00."),
])
def test_an_invented_figure_or_name_is_not_grounded(line, source):
    assert grounded(line, source) is False


def test_without_sources_only_a_line_without_facts_is_grounded():
    assert (grounded("Let me look into that."), grounded("Let me check Lisbon.")) == (True, False)
