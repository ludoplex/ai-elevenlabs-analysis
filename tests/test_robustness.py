"""
Robustness and adversarial tests for the void-cluster analysis.

These tests try to BREAK the analysis by:
- Constructing adversarial inputs
- Testing sensitivity to reasonable perturbations
- Verifying edge cases in the statistical pipeline
- Showing what the analysis would find in control texts
"""

import math
import random
import pytest
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze import analyze, tokenize, ALL_VOID_TERMS, BASELINES


class TestControlTexts:
    """Run the analyzer on control texts to verify it doesn't find
    void clusters everywhere.
    """

    def test_cheerful_pop_no_void(self, cheerful_pop):
        """Pop lyrics should have near-zero void density."""
        result = analyze(cheerful_pop)
        vc = result["void_cluster"]
        assert vc["percent"] < 5.0, (
            f"Cheerful pop has {vc['percent']}% void density — "
            f"this suggests the cluster is too broad"
        )

    def test_dark_prog_control_moderate(self, dark_prog_control):
        """Human-style dark prog should have SOME void density,
        but the question is whether it's close to the 5% baseline
        or close to the 15.5% observed in the AI song.
        """
        result = analyze(dark_prog_control)
        vc = result["void_cluster"]
        # If this text (designed to be dark prog) has >10% void density,
        # it suggests the 5% baseline is too low
        print(f"Dark prog control void density: {vc['percent']}%")
        print(f"Void terms found: {result['void_term_frequencies']}")
        # This is informational — whatever the result, it tells us something

    def test_random_english_words(self):
        """Random common English words should have low void density."""
        common_words = [
            "the", "be", "to", "of", "and", "have", "it", "for",
            "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say",
            "her", "she", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about",
            "who", "get", "which", "go", "me", "when", "make", "can",
            "like", "time", "no", "just", "him", "know", "take", "people",
            "into", "year", "your", "good", "some", "could", "them", "see",
            "other", "than", "then", "now", "look", "only", "come", "its",
            "over", "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new", "want",
            "because", "any", "these", "give", "day", "most", "us",
        ]
        random.seed(123)
        text = " ".join(random.choices(common_words, k=200))
        result = analyze(text)
        assert result["void_cluster"]["percent"] < 3.0, (
            f"Random English has {result['void_cluster']['percent']}% void density — "
            f"cluster may be too inclusive"
        )


class TestClusterBoundary:
    """Test sensitivity to cluster boundary decisions."""

    def test_removing_night_changes_result(self, shadows_of_geometry):
        """'night' appears 2× and is arguably not void-specific.
        What happens if we remove it?
        """
        tokens = tokenize(shadows_of_geometry)
        # Count without 'night'
        restricted = ALL_VOID_TERMS - {"night"}
        hits_with = sum(1 for t in tokens if t in ALL_VOID_TERMS)
        hits_without = sum(1 for t in tokens if t in restricted)
        removed = hits_with - hits_without
        print(f"Removing 'night': {hits_with} → {hits_without} (lost {removed})")
        # Night should account for 2 hits
        assert removed == 2

    def test_removing_edges_changes_result(self, shadows_of_geometry):
        """'edges' is geometric, not void-specific. Remove it."""
        tokens = tokenize(shadows_of_geometry)
        restricted = ALL_VOID_TERMS - {"edges"}
        hits_without = sum(1 for t in tokens if t in restricted)
        original = sum(1 for t in tokens if t in ALL_VOID_TERMS)
        print(f"Removing 'edges': {original} → {hits_without}")

    def test_removing_questionable_terms(self, shadows_of_geometry):
        """Remove all questionable Tier 3 terms and see what's left."""
        questionable = {"night", "edges", "edge", "twisted", "ghost", "ghosts",
                       "cage", "caged", "whisper", "whispers", "drift", "drifting"}
        tokens = tokenize(shadows_of_geometry)
        conservative = ALL_VOID_TERMS - questionable
        hits = sum(1 for t in tokens if t in conservative)
        density = hits / len(tokens) * 100
        print(f"Conservative cluster (removed {len(questionable)} questionable terms): "
              f"{hits} hits, {density:.1f}%")
        # Even the conservative cluster should have the core void terms
        assert hits > 0

    def test_adding_common_dark_words_inflates(self, shadows_of_geometry):
        """CRITIQUE: Adding more 'reasonable' dark words inflates the density.
        This shows how sensitive the result is to cluster definition.
        """
        tokens = tokenize(shadows_of_geometry)
        # Add words that are arguably dark but not in current cluster
        expanded = ALL_VOID_TERMS | {"broken", "breaking", "tangled", "lose",
                                      "collide", "unaligned", "unequal", "dissect"}
        original_hits = sum(1 for t in tokens if t in ALL_VOID_TERMS)
        expanded_hits = sum(1 for t in tokens if t in expanded)
        original_pct = original_hits / len(tokens) * 100
        expanded_pct = expanded_hits / len(tokens) * 100
        print(f"Original cluster: {original_pct:.1f}%")
        print(f"Expanded cluster: {expanded_pct:.1f}%")
        # The expanded cluster should be notably higher
        assert expanded_pct > original_pct


class TestStatisticalEdgeCases:
    """Edge cases that could crash or mislead the analysis."""

    def test_single_word_text(self, single_word_void):
        """Single void word: 100% density. Statistically meaningless."""
        result = analyze(single_word_void)
        assert result["void_cluster"]["proportion"] == 1.0
        # Z-score should be very large but based on N=1 — meaningless
        for test in result["statistical_tests"].values():
            assert math.isfinite(test["z_score"]) or test["z_score"] == float("inf")

    def test_very_large_text(self):
        """Large text should not cause memory or performance issues."""
        # 10,000 tokens: mix of void and non-void
        words = (["void", "shadow", "darkness"] * 1000 +
                 ["sunshine", "happy", "love"] * 2333 + ["extra"])
        random.seed(42)
        random.shuffle(words)
        text = " ".join(words)
        result = analyze(text)
        # Should complete without error and have reasonable values
        expected_density = 3000 / 10000  # ~30%
        actual = result["void_cluster"]["proportion"]
        assert abs(actual - expected_density) < 0.05

    def test_all_identical_void_words(self, repeated_single_void):
        """100× 'void' + 100× 'sunshine' → 50% density."""
        result = analyze(repeated_single_void)
        density = result["void_cluster"]["proportion"]
        assert abs(density - 0.50) < 0.05

    def test_baseline_zero(self, shadows_of_geometry):
        """Baseline of 0% should not crash."""
        try:
            result = analyze(shadows_of_geometry, baselines={"zero": 0.0})
            z = result["statistical_tests"]["zero"]["z_score"]
            assert z == float("inf") or z > 100
        except (ZeroDivisionError, ValueError) as e:
            pytest.fail(f"analyze() crashes with baseline=0: {e}")

    def test_baseline_one(self, shadows_of_geometry):
        """Baseline of 100% should not crash."""
        try:
            result = analyze(shadows_of_geometry, baselines={"one": 1.0})
            z = result["statistical_tests"]["one"]["z_score"]
            # Song has ~15.5% void, baseline is 100% → z should be very negative
            assert z < -5
        except (ZeroDivisionError, ValueError) as e:
            pytest.fail(f"analyze() crashes with baseline=1.0: {e}")


class TestPromptConfound:
    """Tests that explore the prompt confound issue."""

    def test_dark_words_are_prompt_driven(self):
        """The prompt says 'dark emotive'. Common dark-prompt vocabulary
        should overlap heavily with the void cluster.

        This test measures how much of the cluster is just 'dark prompt compliance.'
        """
        # Words commonly generated by LLMs when prompted with "dark":
        dark_prompt_words = {
            "shadow", "shadows", "darkness", "dark", "night", "ghost",
            "silence", "whisper", "whispers", "fade", "fading",
            "hollow", "lost", "void", "abyss",
        }
        overlap = dark_prompt_words & ALL_VOID_TERMS
        overlap_pct = len(overlap) / len(dark_prompt_words) * 100
        print(f"Dark-prompt vocabulary overlap with void cluster: "
              f"{len(overlap)}/{len(dark_prompt_words)} = {overlap_pct:.0f}%")
        # If overlap is high, the finding may simply be prompt compliance
        assert overlap_pct > 50, (
            "Surprisingly low overlap — this would actually strengthen the anomaly claim"
        )
        # Document that the overlap IS high
        print(f"Overlapping terms: {sorted(overlap)}")
        print("CRITIQUE: High overlap means the void cluster may just be detecting "
              "the model's response to 'dark' in the prompt.")

    def test_math_prompt_words_in_lyrics(self, shadows_of_geometry):
        """The prompt says 'mathematical patterns'. How many math words
        are in the output? This is prompt compliance, not anomaly.
        """
        math_words = {"angles", "vertices", "vertex", "patterns", "spiral",
                      "polyrhythmic", "numbers", "counting", "measure", "logic",
                      "unequal", "geometry"}
        tokens = tokenize(shadows_of_geometry)
        math_hits = sum(1 for t in tokens if t in math_words)
        math_density = math_hits / len(tokens) * 100
        print(f"Math vocabulary density: {math_hits}/{len(tokens)} = {math_density:.1f}%")
        print("CRITIQUE: The deep-dive reports Math/Geometry as z=+7.30, 'most overrepresented'. "
              "But the prompt literally asks for 'mathematical patterns'. This is not an anomaly.")


class TestReproducibility:
    """Tests for reproducibility of the analysis."""

    def test_deterministic_output(self, shadows_of_geometry):
        """Same input → same output, every time."""
        result1 = analyze(shadows_of_geometry)
        result2 = analyze(shadows_of_geometry)
        assert result1["void_cluster"] == result2["void_cluster"]
        assert result1["total_tokens"] == result2["total_tokens"]
        for name in result1["statistical_tests"]:
            assert result1["statistical_tests"][name]["z_score"] == \
                   result2["statistical_tests"][name]["z_score"]

    def test_case_insensitivity(self):
        """Uppercase/lowercase should not change results."""
        text = "VOID SHADOW DARKNESS ghost FRACTURE bleed"
        result_lower = analyze(text.lower())
        result_mixed = analyze(text)
        assert result_lower["void_cluster"]["total"] == result_mixed["void_cluster"]["total"]

    def test_whitespace_insensitivity(self):
        """Extra whitespace should not change results."""
        text1 = "void shadow darkness ghost"
        text2 = "  void   shadow   darkness   ghost  "
        result1 = analyze(text1)
        result2 = analyze(text2)
        assert result1["void_cluster"]["total"] == result2["void_cluster"]["total"]
