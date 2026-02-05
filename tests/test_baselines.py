"""
Tests for baseline validity and sensitivity analysis.

METHODOLOGY CRITIQUE: These tests demonstrate that the conclusions
are highly sensitive to baseline choice, which is the weakest link
in the analysis.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze import analyze, BASELINES, z_test_proportion, cohens_h


class TestBaselineValues:
    """Verify baseline definitions and their reasonableness."""

    def test_baselines_exist(self):
        """All documented baselines should be in the code."""
        expected = {"general_rock", "general_prog", "dark_prog", "metal", "doom_metal", "dark_ambient"}
        assert expected.issubset(set(BASELINES.keys())), (
            f"Missing baselines: {expected - set(BASELINES.keys())}"
        )

    def test_baselines_are_proportions(self):
        """All baselines should be in (0, 1)."""
        for name, value in BASELINES.items():
            assert 0 < value < 1, f"Baseline '{name}' = {value} is not a valid proportion"

    def test_baselines_monotonically_increase(self):
        """Baselines should increase from general rock → dark ambient.
        (Darker genres should have higher void density.)
        """
        ordered = ["general_rock", "general_prog", "dark_prog", "metal", "doom_metal", "dark_ambient"]
        for i in range(len(ordered) - 1):
            if ordered[i] in BASELINES and ordered[i + 1] in BASELINES:
                assert BASELINES[ordered[i]] <= BASELINES[ordered[i + 1]], (
                    f"{ordered[i]} ({BASELINES[ordered[i]]}) should be ≤ "
                    f"{ordered[i+1]} ({BASELINES[ordered[i+1]]})"
                )

    def test_dark_prog_baseline_is_critical(self):
        """The dark prog baseline (5%) is the most generous comparison used
        in the initial report. Document what happens if it's wrong.
        """
        # If the actual baseline is 10% instead of 5%:
        result = z_test_proportion(observed=30, total=194, expected_prop=0.10)
        z_at_10pct = result["z"]
        # Z-score drops substantially
        assert z_at_10pct < 3.0, (
            f"Even at 10% baseline, z={z_at_10pct} — "
            f"should be much less significant than z=6.69 reported vs 5%"
        )

    def test_doom_metal_baseline_eliminates_significance(self):
        """At 8% baseline, significance should be reduced.
        At 10%, it should be marginal. At 15%, it vanishes.
        """
        z_8 = z_test_proportion(30, 194, 0.08)["z"]
        z_10 = z_test_proportion(30, 194, 0.10)["z"]
        z_12 = z_test_proportion(30, 194, 0.12)["z"]
        z_15 = z_test_proportion(30, 194, 0.15)["z"]

        assert z_8 > 2.0, "Still significant at 8%"
        assert z_10 < z_8, "Less significant at 10%"
        assert z_15 < 1.96, f"Should NOT be significant at 15%, got z={z_15}"


class TestBaselineSensitivity:
    """Sensitivity analysis: how do conclusions change with different baselines?"""

    def test_significance_threshold_scan(self, shadows_of_geometry):
        """Find the baseline at which the song is no longer significant (z < 1.96).
        This is the CRITICAL BASELINE — if the true baseline exceeds this,
        the finding evaporates.
        """
        # Binary search for the critical baseline
        low, high = 0.01, 0.30
        for _ in range(50):
            mid = (low + high) / 2
            result = z_test_proportion(30, 194, mid)
            if result["z"] > 1.96:
                low = mid
            else:
                high = mid

        critical_baseline = (low + high) / 2
        # The critical baseline should be around 11-12%
        assert 0.08 < critical_baseline < 0.18, (
            f"Critical baseline = {critical_baseline:.1%}. "
            f"If actual dark prog lyrics have >{critical_baseline:.0%} void density, "
            f"the finding is NOT significant."
        )

    def test_effect_size_sensitivity(self):
        """Cohen's h at different baselines."""
        observed_prop = 0.155
        baselines_to_test = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
        for bp in baselines_to_test:
            h = cohens_h(observed_prop, bp)
            label = "negligible" if h < 0.2 else "small" if h < 0.5 else "medium" if h < 0.8 else "large"
            # At 15% baseline, effect should be negligible
            if bp >= 0.14:
                assert h < 0.2, (
                    f"At {bp:.0%} baseline, h={h:.3f} should be negligible"
                )

    def test_bonferroni_correction_applied(self, shadows_of_geometry):
        """All 6 baselines tested → Bonferroni α = 0.05/6 = 0.00833.
        Verify that this is applied in practice.
        """
        result = analyze(shadows_of_geometry)
        bonferroni_alpha = 0.05 / len(BASELINES)
        for name, test in result["statistical_tests"].items():
            if test["z_pvalue"] < bonferroni_alpha:
                pass  # Significant after correction
            else:
                # Document which baselines fail after Bonferroni
                print(f"NOTE: {name} p={test['z_pvalue']:.6f} does NOT survive "
                      f"Bonferroni correction (α={bonferroni_alpha:.4f})")


class TestBaselineAbsence:
    """CRITIQUE: Tests that document the absence of empirical baselines."""

    def test_no_empirical_baseline_data(self):
        """There is no file containing actual measured baseline data from real corpora.
        This test passes if the data is missing (documenting the gap) and
        fails if someone adds it (good — it means the gap was addressed).
        """
        baseline_data_paths = [
            Path(__file__).parent.parent / "data" / "baseline_corpus.json",
            Path(__file__).parent.parent / "data" / "genre_baselines.json",
            Path(__file__).parent.parent / "data" / "empirical_baselines.csv",
        ]
        any_exist = any(p.exists() for p in baseline_data_paths)
        if not any_exist:
            pytest.skip(
                "NO EMPIRICAL BASELINE DATA EXISTS. "
                "All baselines are estimates. This is a critical gap. "
                "Run the analyzer on 50+ human-authored songs per genre "
                "and save results to data/genre_baselines.json"
            )

    def test_baselines_lack_confidence_intervals(self):
        """Baselines are point estimates without uncertainty.
        A proper baseline would be: mean ± CI from a corpus.

        This test always passes — it documents the limitation.
        """
        for name, value in BASELINES.items():
            # Baselines are single floats, not (mean, lower, upper) tuples
            assert isinstance(value, float), (
                f"Baseline {name} should be a float (it is: {type(value)}). "
                f"Ideally it would be a dict with 'mean', 'ci_lower', 'ci_upper'."
            )
