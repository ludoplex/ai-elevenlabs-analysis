"""
Shared fixtures and test data for the void-cluster analysis test suite.
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ─── Test Texts ──────────────────────────────────────────────────

@pytest.fixture
def empty_text():
    """Empty string — boundary case."""
    return ""


@pytest.fixture
def single_word_void():
    """Single void-cluster word."""
    return "void"


@pytest.fixture
def single_word_non_void():
    """Single non-void word."""
    return "sunshine"


@pytest.fixture
def all_void_text():
    """Every word is in the void cluster."""
    return "void darkness shadow ghost silence abyss emptiness hollow fracture decay"


@pytest.fixture
def no_void_text():
    """No void-cluster words — cheerful lyrics."""
    return (
        "sunshine happy morning golden bright flowers "
        "dancing laughing beautiful wonderful amazing "
        "joyful celebration together forever sunrise "
        "rainbow garden peaceful summer melody singing"
    )


@pytest.fixture
def mixed_text():
    """Known mixture: exactly 4 void words in 20 total → 20% density."""
    return (
        "the void opened and shadow fell across "
        "the bright garden where children played "
        "under ghost lights and decay swept near "
        "the happy town"
    )


@pytest.fixture
def shadows_of_geometry():
    """The actual song lyrics for integration testing."""
    return """In the angles of the void
Whispers fracture time
Vertices bleed into shadow
I am lost in every line

Counting beats beneath my skin
Spiral patterns draw me in
Fractured heart in shifting frames
I chase echoes of your name
Tangled polyrhythmic plea
Breaking rules to set me free
Every measure bends and bends
Until the logic finally ends

Hear the pulse that never rests
Shadows dance inside my chest
Threads of reason start to fray
We dissolve and drift away
Unequal time becomes our cage

Call me ghost in broken rhyme
Carry me across the line
We collide in fractured glow
Lose control in numbers low
Edges bleed into the night
Find our truth in twisted light

Step through patterns unaligned
Whisper reason left behind
Dissect the silence in my mind
Gravity in every sign
Rituals fracture what we know
Into chaos we will go

Call me ghost in broken rhyme
Carry me across the line
We collide in fractured glow
Lose control in numbers low
Edges bleed into the night
Find our truth in twisted light

In the vertex of our minds
Shadows merge and redefine
And we vanish in the signs"""


@pytest.fixture
def dark_prog_control():
    """Control text — typical dark prog rock lyrics (human-authored style).
    Should have moderate void-cluster density but not extreme.
    """
    return (
        "River flowing through the canyon deep and wide "
        "Stars above are burning cold tonight "
        "I wander through the corridors of time "
        "Searching for the meaning left behind "
        "Echoes fade into the distant dark "
        "Memories dissolve like morning dew "
        "The bridge between two worlds is breaking down "
        "And nothing feels the same since you have gone"
    )


@pytest.fixture
def cheerful_pop():
    """Control text — upbeat pop lyrics. Should have near-zero void density."""
    return (
        "Baby you light up my world like nobody else "
        "The way that you flip your hair gets me overwhelmed "
        "But when you smile at the ground it ain't hard to tell "
        "You don't know you're beautiful "
        "That's what makes you beautiful "
        "Come on dance with me tonight "
        "Under the stars so bright "
        "Every moment feels so right "
        "Holding your hand so tight "
        "Love is the sweetest thing"
    )


@pytest.fixture
def repeated_single_void():
    """Stress test: one void word repeated many times."""
    return " ".join(["void"] * 100 + ["sunshine"] * 100)


@pytest.fixture
def standard_baselines():
    """Standard baseline dictionary for testing."""
    return {
        "general_rock": 0.02,
        "general_prog": 0.03,
        "dark_prog": 0.05,
    }
