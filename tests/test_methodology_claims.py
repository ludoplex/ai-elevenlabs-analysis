"""
Tests that directly verify (or challenge) specific claims made in the reports.

Each test corresponds to a specific claim in the analysis documents.
Tests are designed to either confirm the claim or expose its weakness.
"""

import pytest
import sys
import math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze import tokenize, classify_void_tokens, analyze, ALL_VOID_TERMS, VOID_CLUSTER


class TestClaim_TierNarrowingDestroysFinding:
    """CLAIM (methodology.md): 'Narrow cluster (Tier 1+2 only): 0.5% — barely above baseline.'

    This is the single most damning admission in the analysis.
    """

    def test_tier12_only_density(self, shadows_of_geometry):
        """With only Tier 1+2, void density should be ≈ 0.5%."""
        tokens = tokenize(shadows_of_geometry)
        tier12 = VOID_CLUSTER["direct"] | VOID_CLUSTER["synonyms"]
        hits = sum(1 for t in tokens if t in tier12)
        density = hits / len(tokens) * 100
        assert density < 1.5, (
            f"Tier 1+2 density = {density:.1f}%. "
            f"The finding requires Tier 3 — it doesn't exist under strict definitions."
        )

    def test_tier12_not_significant(self, shadows_of_geometry):
        """Tier 1+2 density should NOT be statistically significant vs any baseline."""
        tokens = tokenize(shadows_of_geometry)
        tier12 = VOID_CLUSTER["direct"] | VOID_CLUSTER["synonyms"]
        hits = sum(1 for t in tokens if t in tier12)
        from analyze import z_test_proportion
        for baseline_name, baseline_value in [("rock", 0.02), ("prog", 0.03)]:
            z = z_test_proportion(hits, len(tokens), baseline_value)
            assert z["z"] < 1.96, (
                f"Tier 1+2 is significant vs {baseline_name} (z={z['z']}). "
                f"This would be surprising and contradict the methodology claim."
            )


class TestClaim_ChorusRepetitionRobustness:
    """CLAIM: 'Excluding the repeated chorus still yields ~23/158 = 14.6%'"""

    def test_exclude_chorus2_density(self, shadows_of_geometry):
        """Remove the second chorus and verify density stays high."""
        # The second chorus is an exact repeat of the first
        lines = shadows_of_geometry.strip().split("\n")

        # Find and remove the second occurrence of the chorus block
        chorus = [
            "Call me ghost in broken rhyme",
            "Carry me across the line",
            "We collide in fractured glow",
            "Lose control in numbers low",
            "Edges bleed into the night",
            "Find our truth in twisted light",
        ]

        # Find second occurrence
        text_lines = [l.strip() for l in lines if l.strip()]
        chorus_starts = []
        for i in range(len(text_lines)):
            if text_lines[i] == chorus[0]:
                # Check if full chorus follows
                if i + 5 < len(text_lines) and all(
                    text_lines[i + j] == chorus[j] for j in range(6)
                ):
                    chorus_starts.append(i)

        assert len(chorus_starts) >= 2, "Expected at least 2 chorus instances"

        # Remove the second chorus
        no_chorus2 = [l for i, l in enumerate(text_lines)
                      if not (chorus_starts[1] <= i < chorus_starts[1] + 6)]
        text_no_c2 = "\n".join(no_chorus2)

        tokens = tokenize(text_no_c2)
        result = classify_void_tokens(tokens)
        total_void = len(result["direct"]) + len(result["synonyms"]) + len(result["semantic_neighbors"])
        density = total_void / len(tokens) * 100

        # Report claims ~14.6%
        assert 12.0 <= density <= 17.0, (
            f"Without chorus 2, density = {density:.1f}%. "
            f"Report claims ~14.6%."
        )


class TestClaim_AllSectionsContainVoid:
    """CLAIM: 'Every section contains void-cluster terms — it's not localized.'"""

    def test_void_in_every_section(self, shadows_of_geometry):
        """Each song section should contain at least one void-cluster term."""
        sections = {
            "Intro": "In the angles of the void\nWhispers fracture time\nVertices bleed into shadow\nI am lost in every line",
            "Verse 1": "Counting beats beneath my skin\nSpiral patterns draw me in\nFractured heart in shifting frames\nI chase echoes of your name\nTangled polyrhythmic plea\nBreaking rules to set me free\nEvery measure bends and bends\nUntil the logic finally ends",
            "Pre-Chorus": "Hear the pulse that never rests\nShadows dance inside my chest\nThreads of reason start to fray\nWe dissolve and drift away\nUnequal time becomes our cage",
            "Chorus": "Call me ghost in broken rhyme\nCarry me across the line\nWe collide in fractured glow\nLose control in numbers low\nEdges bleed into the night\nFind our truth in twisted light",
            "Bridge": "Step through patterns unaligned\nWhisper reason left behind\nDissect the silence in my mind\nGravity in every sign\nRituals fracture what we know\nInto chaos we will go",
            "Outro": "In the vertex of our minds\nShadows merge and redefine\nAnd we vanish in the signs",
        }

        for section_name, section_text in sections.items():
            tokens = tokenize(section_text)
            result = classify_void_tokens(tokens)
            total_void = len(result["direct"]) + len(result["synonyms"]) + len(result["semantic_neighbors"])
            assert total_void > 0, (
                f"Section '{section_name}' has NO void-cluster terms! "
                f"This contradicts the claim that all sections contain void words."
            )


class TestClaim_FractureIsExtemeOutlier:
    """CLAIM: '"fracture" family: 5 uses (2.6%) — extreme outlier vs ~0-1 per song'"""

    def test_fracture_count(self, shadows_of_geometry):
        tokens = tokenize(shadows_of_geometry)
        fracture_count = sum(1 for t in tokens if t in {"fracture", "fractured", "fractures", "fracturing"})
        assert fracture_count == 5, f"Expected 5 fracture-family tokens, got {fracture_count}"

    def test_fracture_percentage(self, shadows_of_geometry):
        tokens = tokenize(shadows_of_geometry)
        fracture_count = sum(1 for t in tokens if t in {"fracture", "fractured", "fractures", "fracturing"})
        pct = fracture_count / len(tokens) * 100
        assert abs(pct - 2.6) < 0.5, f"Expected ~2.6%, got {pct:.1f}%"


class TestClaim_DistributionalStrategy:
    """CLAIM: 'The AI avoids repeating "void" itself but saturates the semantic field
    through neighbors. This is a distributional strategy.'

    CRITIQUE: This attributes intentionality to a probabilistic model.
    The simpler explanation is repetition penalty.
    """

    def test_void_appears_only_once(self, shadows_of_geometry):
        tokens = tokenize(shadows_of_geometry)
        assert tokens.count("void") == 1

    def test_many_unique_void_terms(self, shadows_of_geometry):
        """The analysis claims 20 unique void-cluster terms."""
        tokens = tokenize(shadows_of_geometry)
        void_tokens = [t for t in tokens if t in ALL_VOID_TERMS]
        unique_void = set(void_tokens)
        # Report claims 20 unique terms
        assert len(unique_void) >= 18, (
            f"Expected ~20 unique void terms, got {len(unique_void)}: {unique_void}"
        )

    def test_no_void_term_dominates(self, shadows_of_geometry):
        """No single void term should account for >25% of void hits.
        (If one term dominated, it would be repetition, not 'distribution'.)
        """
        tokens = tokenize(shadows_of_geometry)
        void_tokens = [t for t in tokens if t in ALL_VOID_TERMS]
        total_void = len(void_tokens)
        freq = Counter(void_tokens)
        max_single = freq.most_common(1)[0][1]  # Count of most common void term
        assert max_single / total_void < 0.25, (
            f"Most common void term appears {max_single}/{total_void} times "
            f"({max_single/total_void:.0%}). That's too dominant for a 'distributed' strategy."
        )


class TestClaim_ZScoresMatchReport:
    """Verify that the z-scores match the reported values exactly."""

    def test_z_vs_general_rock(self, shadows_of_geometry):
        result = analyze(shadows_of_geometry, baselines={"general_rock": 0.02})
        z = result["statistical_tests"]["general_rock"]["z_score"]
        # Report says +13.40
        assert abs(z - 13.40) < 0.5, f"Expected z ≈ +13.40, got {z}"

    def test_z_vs_general_prog(self, shadows_of_geometry):
        result = analyze(shadows_of_geometry, baselines={"general_prog": 0.03})
        z = result["statistical_tests"]["general_prog"]["z_score"]
        # Report says +10.18
        assert abs(z - 10.18) < 0.5, f"Expected z ≈ +10.18, got {z}"

    def test_z_vs_dark_prog(self, shadows_of_geometry):
        result = analyze(shadows_of_geometry, baselines={"dark_prog": 0.05})
        z = result["statistical_tests"]["dark_prog"]["z_score"]
        # Report says +6.69
        assert abs(z - 6.69) < 0.5, f"Expected z ≈ +6.69, got {z}"


class TestClaim_ContentWordDensity:
    """CLAIM: 'The void cluster alone consumes ~26% of all content words (30 of ~115 non-function words)'"""

    def test_content_word_void_density(self, shadows_of_geometry):
        """Verify that void-cluster proportion of content words is ~26%."""
        function_words = {
            "the", "in", "of", "to", "and", "into", "me", "we", "my", "our",
            "am", "is", "are", "it", "that", "this", "an", "or", "but",
            "every", "until", "never", "becomes", "what", "will",
        }
        tokens = tokenize(shadows_of_geometry)
        content_tokens = [t for t in tokens if t not in function_words]
        void_tokens = [t for t in content_tokens if t in ALL_VOID_TERMS]

        if len(content_tokens) > 0:
            density = len(void_tokens) / len(content_tokens) * 100
            # Report claims ~26%
            assert 20 <= density <= 35, (
                f"Content-word void density = {density:.1f}%. Report claims ~26%."
            )


class TestWordListConsistency:
    """CRITIQUE: The word lists differ between analyze.py and void-cluster-analyzer.c.
    These tests document the discrepancies.
    """

    def test_python_cluster_size(self):
        """Document the Python cluster size."""
        total = len(ALL_VOID_TERMS)
        print(f"Python analyze.py cluster size: {total} terms")
        # Should be ~60 terms per the code
        assert 40 <= total <= 80, f"Unexpected cluster size: {total}"

    def test_c_analyzer_has_more_terms(self):
        """The C analyzer includes ~110 terms vs Python's ~60.
        This is a consistency issue that could produce different results.
        """
        # Read the C source and count DEFAULT_CLUSTER entries
        c_source = Path(__file__).parent.parent / "scripts" / "void-cluster-analyzer.c"
        if not c_source.exists():
            pytest.skip("C analyzer source not found")

        content = c_source.read_text(encoding="utf-8")
        # Count quoted strings in DEFAULT_CLUSTER
        import re
        c_terms = re.findall(r'"([a-z]+)"', content.split("DEFAULT_CLUSTER")[1].split("NULL")[0])
        c_term_set = set(c_terms)

        python_terms = ALL_VOID_TERMS

        extra_in_c = c_term_set - python_terms
        extra_in_python = python_terms - c_term_set

        # Document discrepancies
        if extra_in_c:
            print(f"\nTerms in C but NOT in Python ({len(extra_in_c)}):")
            for t in sorted(extra_in_c):
                print(f"  + {t}")
        if extra_in_python:
            print(f"\nTerms in Python but NOT in C ({len(extra_in_python)}):")
            for t in sorted(extra_in_python):
                print(f"  - {t}")

        # This test is informational — it documents the gap
        # If the sets are very different, something is wrong
        overlap = len(c_term_set & python_terms)
        total_union = len(c_term_set | python_terms)
        jaccard = overlap / total_union if total_union > 0 else 0
        # CRITIQUE FINDING: The Jaccard similarity is expected to be low.
        # This test DOCUMENTS the inconsistency rather than enforcing it.
        # When the word lists are unified, change threshold to 0.9+.
        if jaccard < 0.5:
            pytest.xfail(
                f"Jaccard similarity between C and Python clusters is only {jaccard:.2f}. "
                f"These tools WILL produce different results on the same input. "
                f"RECOMMENDATION: Unify the word lists."
            )
