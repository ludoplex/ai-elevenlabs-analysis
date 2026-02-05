"""
Tests for statistical functions in analyze.py.

Tests cover:
- z_test_proportion: correctness, edge cases, known values
- chi_squared: correctness, edge cases, assumptions
- cohens_h: correctness, symmetry, boundary values
- Full analyze() pipeline integration

METHODOLOGY CRITIQUE: Several tests explicitly verify the LIMITATIONS of these
tests — e.g., that the z-test produces misleadingly extreme p-values when
independence assumptions are violated.
"""

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze import z_test_proportion, chi_squared, cohens_h, analyze


class TestZTestProportion:
    """Tests for the one-tailed z-test for proportions."""

    def test_perfect_match(self):
        """When observed proportion equals baseline, z ≈ 0."""
        result = z_test_proportion(observed=5, total=100, expected_prop=0.05)
        assert abs(result["z"]) < 0.01, f"Expected z ≈ 0, got {result['z']}"
        assert result["p"] >= 0.49, f"Expected p ≈ 0.5, got {result['p']}"

    def test_clear_overrepresentation(self):
        """50/100 vs 5% baseline → large positive z."""
        result = z_test_proportion(observed=50, total=100, expected_prop=0.05)
        assert result["z"] > 10, f"Expected z >> 0, got {result['z']}"
        assert result["p"] < 0.0001

    def test_clear_underrepresentation(self):
        """0/100 vs 50% baseline → large negative z."""
        result = z_test_proportion(observed=0, total=100, expected_prop=0.50)
        assert result["z"] < -5

    def test_known_value(self):
        """Verify against hand-calculated z-score.
        observed=30, total=194, expected=0.05
        p_hat = 30/194 = 0.15464
        se = sqrt(0.05 * 0.95 / 194) = sqrt(0.000245) = 0.01564
        z = (0.15464 - 0.05) / 0.01564 = 6.69
        """
        result = z_test_proportion(observed=30, total=194, expected_prop=0.05)
        assert abs(result["z"] - 6.69) < 0.1, (
            f"Expected z ≈ 6.69 (per initial report), got {result['z']}"
        )

    def test_zero_expected_proportion(self):
        """Baseline of 0% → se = 0 → should handle gracefully."""
        result = z_test_proportion(observed=5, total=100, expected_prop=0.0)
        assert result["z"] == float("inf") or result["z"] > 1000

    def test_one_expected_proportion(self):
        """Baseline of 100% → se = 0 → should handle gracefully."""
        result = z_test_proportion(observed=5, total=100, expected_prop=1.0)
        # se = sqrt(1.0 * 0.0 / 100) = 0
        assert result["z"] == float("inf") or result["z"] < -1000 or result["z"] == float("-inf")

    def test_small_sample_warning(self):
        """n*p0 < 5 means the normal approximation is unreliable.
        The test still runs but the result is statistically questionable.

        CRITIQUE: The analysis uses n=194, p0=0.02 → np0=3.88 < 5.
        This test documents that the code doesn't warn about this.
        """
        result = z_test_proportion(observed=5, total=100, expected_prop=0.02)
        expected_count = 100 * 0.02  # = 2.0 < 5
        assert expected_count < 5, "Test setup: expected count should be < 5"
        # The function returns a result without warning — this is a known limitation
        assert "z" in result
        assert "p" in result

    def test_p_value_bounds(self):
        """P-values should always be in [0, 1]."""
        for obs in [0, 1, 5, 50, 99, 100]:
            for prop in [0.01, 0.05, 0.10, 0.50, 0.90]:
                result = z_test_proportion(observed=obs, total=100, expected_prop=prop)
                assert 0.0 <= result["p"] <= 1.0, (
                    f"p-value out of bounds: {result['p']} for obs={obs}, prop={prop}"
                )


class TestChiSquared:
    """Tests for chi-squared goodness of fit."""

    def test_perfect_match(self):
        """When observed matches expected, chi² ≈ 0."""
        result = chi_squared(observed=5, total=100, expected_prop=0.05)
        assert result["chi2"] < 0.1, f"Expected χ² ≈ 0, got {result['chi2']}"

    def test_large_deviation(self):
        """50/100 vs 5% → very large chi²."""
        result = chi_squared(observed=50, total=100, expected_prop=0.05)
        assert result["chi2"] > 100

    def test_known_value(self):
        """Verify against the reported value for Shadows of Geometry vs dark prog.
        observed=30, total=194, expected=0.05
        Expected void: 194*0.05 = 9.7
        Expected non-void: 194*0.95 = 184.3
        chi² = (30-9.7)²/9.7 + (164-184.3)²/184.3
             = (20.3)²/9.7 + (-20.3)²/184.3
             = 412.09/9.7 + 412.09/184.3
             = 42.48 + 2.24
             = 44.72
        """
        result = chi_squared(observed=30, total=194, expected_prop=0.05)
        assert abs(result["chi2"] - 44.72) < 1.0, (
            f"Expected χ² ≈ 44.72, got {result['chi2']}"
        )

    def test_zero_expected(self):
        """Expected proportion = 0 → division by zero → should handle."""
        result = chi_squared(observed=5, total=100, expected_prop=0.0)
        assert result["chi2"] == float("inf") or result["chi2"] > 10000

    def test_minimum_expected_frequency_violation(self):
        """CRITIQUE: Chi-squared requires expected frequency ≥ 5.
        With n=194, p0=0.02: expected = 3.88 < 5. Test is invalid.
        Document that the code doesn't enforce this.
        """
        result = chi_squared(observed=30, total=194, expected_prop=0.02)
        expected_count = 194 * 0.02  # = 3.88
        assert expected_count < 5, "Test setup: should violate minimum expected frequency"
        # Code runs without error — this is a limitation to document
        assert result["chi2"] > 0

    def test_chi2_equals_z_squared(self):
        """For 2×1 table (df=1), chi² should approximately equal z².
        This is a known mathematical relationship.
        """
        obs, total, prop = 30, 194, 0.05
        z_result = z_test_proportion(obs, total, prop)
        chi_result = chi_squared(obs, total, prop)
        z_sq = z_result["z"] ** 2
        # Should be approximately equal (not exact due to different formulations)
        assert abs(chi_result["chi2"] - z_sq) < 1.0, (
            f"χ² ({chi_result['chi2']}) should ≈ z² ({z_sq})"
        )


class TestCohensH:
    """Tests for Cohen's h effect size."""

    def test_identical_proportions(self):
        """Same proportion → h = 0."""
        assert cohens_h(0.05, 0.05) == 0.0

    def test_symmetry(self):
        """|h(p1, p2)| = |h(p2, p1)| (absolute value taken)."""
        h1 = cohens_h(0.15, 0.05)
        h2 = cohens_h(0.05, 0.15)
        assert abs(h1 - h2) < 0.001

    def test_known_value(self):
        """Verify against the reported Cohen's h for dark prog baseline.
        h = |2*arcsin(√0.1546) - 2*arcsin(√0.05)|
          = |2*0.4044 - 2*0.2267|
          = |0.8088 - 0.4534|
          = 0.3554
        """
        h = cohens_h(0.1546, 0.05)
        assert abs(h - 0.357) < 0.02, f"Expected h ≈ 0.357, got {h}"

    def test_extreme_difference(self):
        """0% vs 100% → maximum possible h."""
        h = cohens_h(0.0, 1.0)
        # h = |2*arcsin(0) - 2*arcsin(1)| = |0 - π| = π ≈ 3.14
        assert abs(h - math.pi) < 0.01

    def test_zero_vs_small(self):
        """0% vs 1% → small h."""
        h = cohens_h(0.0, 0.01)
        assert h > 0
        assert h < 0.3  # Should be small

    def test_boundary_proportions(self):
        """Proportions at 0 and 1 should work without errors."""
        assert cohens_h(0.0, 0.0) == 0.0
        assert cohens_h(1.0, 1.0) == 0.0
        h = cohens_h(0.0, 1.0)
        assert h > 0

    def test_small_medium_large_thresholds(self):
        """Verify understanding of Cohen's conventions."""
        # These aren't tests of the function per se, but of the interpretation
        h_small = 0.2
        h_medium = 0.5
        h_large = 0.8
        # The reported h of 0.357 for dark prog is "small-to-medium"
        reported_h = cohens_h(0.155, 0.05)
        assert h_small <= reported_h <= h_medium, (
            f"Reported h={reported_h} should be between small ({h_small}) and medium ({h_medium})"
        )


class TestAnalyzePipeline:
    """Integration tests for the full analyze() function."""

    def test_empty_text_handled(self):
        """Empty text should not crash."""
        # tokenize("") returns [], so analyze will have total=0
        # This might raise a ZeroDivisionError — that's a bug to catch
        try:
            result = analyze("")
        except ZeroDivisionError:
            pytest.fail("analyze() crashes on empty text — needs zero-division guard")

    def test_no_void_text(self, no_void_text):
        """Cheerful text should have 0% void density."""
        result = analyze(no_void_text)
        assert result["void_cluster"]["total"] == 0
        assert result["void_cluster"]["proportion"] == 0.0

    def test_all_void_text(self, all_void_text):
        """All-void text should have 100% void density."""
        result = analyze(all_void_text)
        assert result["void_cluster"]["proportion"] == 1.0

    def test_shadows_of_geometry_integration(self, shadows_of_geometry):
        """Full pipeline on the actual song should match reported values."""
        result = analyze(shadows_of_geometry)
        vc = result["void_cluster"]

        # Void density should be approximately 15.5%
        assert 14.0 <= vc["percent"] <= 17.0

        # Total void tokens ≈ 30
        assert 28 <= vc["total"] <= 32

        # Statistical tests should all be significant vs dark prog
        dark_prog = result["statistical_tests"]["dark_prog"]
        assert dark_prog["z_score"] > 5.0, f"Z-score vs dark prog too low: {dark_prog['z_score']}"
        assert dark_prog["z_pvalue"] < 0.001

    def test_custom_baselines(self, shadows_of_geometry):
        """Custom baselines should appear in results."""
        custom = {"my_baseline": 0.10}
        result = analyze(shadows_of_geometry, baselines=custom)
        assert "my_baseline" in result["statistical_tests"]
        # 15.5% vs 10% should still be significant
        assert result["statistical_tests"]["my_baseline"]["z_score"] > 2.0

    def test_high_baseline_not_significant(self, shadows_of_geometry):
        """If baseline is set to 15%, the song should NOT be significantly above it."""
        high = {"ceiling": 0.15}
        result = analyze(shadows_of_geometry, baselines=high)
        z = result["statistical_tests"]["ceiling"]["z_score"]
        # z should be near 0 (density ≈ baseline)
        assert -2.0 <= z <= 2.0, (
            f"Expected non-significant z with 15% baseline, got {z}"
        )

    def test_output_structure(self, shadows_of_geometry):
        """Verify the output dictionary has all expected keys."""
        result = analyze(shadows_of_geometry)
        assert "total_tokens" in result
        assert "unique_words" in result
        assert "void_cluster" in result
        assert "void_term_frequencies" in result
        assert "top_words" in result
        assert "statistical_tests" in result

        vc = result["void_cluster"]
        assert "total" in vc
        assert "proportion" in vc
        assert "percent" in vc
        assert "direct" in vc
        assert "synonyms" in vc
        assert "semantic_neighbors" in vc

    def test_void_term_frequencies_are_counts(self, shadows_of_geometry):
        """Void term frequencies should be positive integers."""
        result = analyze(shadows_of_geometry)
        for term, count in result["void_term_frequencies"].items():
            assert isinstance(count, int), f"{term} count is not int: {type(count)}"
            assert count > 0, f"{term} has non-positive count: {count}"

    def test_void_total_matches_frequencies(self, shadows_of_geometry):
        """Sum of void term frequencies should equal total void count."""
        result = analyze(shadows_of_geometry)
        freq_sum = sum(result["void_term_frequencies"].values())
        assert freq_sum == result["void_cluster"]["total"], (
            f"Frequency sum ({freq_sum}) != total ({result['void_cluster']['total']})"
        )


class TestIndependenceAssumptionViolation:
    """CRITIQUE TESTS: Demonstrate that independence assumptions are violated.

    These tests don't test the code per se — they document the methodological
    weakness by showing that the z-test produces extreme results even when
    the 'signal' is entirely structural.
    """

    def test_repeated_chorus_inflates_significance(self):
        """A single void-dense chorus repeated 10× should NOT produce 10× the z-score,
        but it does under the independence assumption.
        """
        chorus = "ghost shadow void darkness bleed fracture normal word here another"
        # Single chorus: 6 void words in 10 tokens = 60%
        single_result = analyze(chorus, baselines={"test": 0.05})
        # Repeat 10 times: still 60% density but N=100 → z is much larger
        repeated = " ".join([chorus] * 10)
        repeated_result = analyze(repeated, baselines={"test": 0.05})

        z_single = single_result["statistical_tests"]["test"]["z_score"]
        z_repeated = repeated_result["statistical_tests"]["test"]["z_score"]

        # The z-score scales with sqrt(N), so 10× repetition → ~3.16× z-score
        # This is WRONG — the repeated chorus adds no new information
        # But the test documents that the code behaves this way
        assert z_repeated > z_single * 2, (
            "Expected inflation from repeated chorus — "
            "this documents the independence violation"
        )

    def test_shuffled_text_same_result(self):
        """Shuffling word order shouldn't change void density,
        showing the test ignores word order (i.e., treats tokens as IID).
        This is a LIMITATION — real significance depends on structure.
        """
        import random
        random.seed(42)

        text = (
            "void shadow ghost fracture bleed night "
            "happy bright sunshine golden flower warm "
            "the and in of to with"
        )
        original = analyze(text, baselines={"test": 0.05})

        tokens = text.split()
        random.shuffle(tokens)
        shuffled = " ".join(tokens)
        shuffled_result = analyze(shuffled, baselines={"test": 0.05})

        assert original["void_cluster"]["total"] == shuffled_result["void_cluster"]["total"]
        assert original["statistical_tests"]["test"]["z_score"] == shuffled_result["statistical_tests"]["test"]["z_score"]
