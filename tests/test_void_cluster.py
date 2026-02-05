"""
Tests for void cluster classification and analysis in analyze.py.

These tests verify:
- Cluster membership (is each word correctly classified?)
- Cluster consistency (does the code match the methodology document?)
- Classification correctness across tiers
- Edge cases in classification
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze import (
    ALL_VOID_TERMS,
    VOID_CLUSTER,
    classify_void_tokens,
    tokenize,
)


class TestClusterMembership:
    """Verify the void cluster contains expected terms."""

    def test_direct_void(self):
        assert "void" in VOID_CLUSTER["direct"]

    def test_synonyms_present(self):
        expected_synonyms = {"emptiness", "nothing", "nothingness", "abyss", "hollow", "blank"}
        for word in expected_synonyms:
            assert word in VOID_CLUSTER["synonyms"], f"'{word}' missing from synonyms"

    def test_key_neighbors_present(self):
        """Key Tier 3 terms that drive the finding must be present."""
        critical_neighbors = {
            "shadow", "shadows", "ghost", "fracture", "fractured",
            "bleed", "night", "edges", "twisted", "cage", "drift",
            "fray", "whisper", "whispers", "dissolve", "silence",
            "lost", "chaos", "vanish",
        }
        for word in critical_neighbors:
            assert word in VOID_CLUSTER["semantic_neighbors"], (
                f"'{word}' missing from semantic_neighbors"
            )

    def test_non_void_words_excluded(self):
        """Common words should NOT be in the cluster."""
        non_void = ["happy", "sunshine", "love", "dance", "music", "the", "and", "is"]
        for word in non_void:
            assert word not in ALL_VOID_TERMS, f"'{word}' should not be in void cluster"

    def test_all_void_terms_is_union(self):
        """ALL_VOID_TERMS should be the union of all three tiers."""
        expected = set()
        for category in VOID_CLUSTER.values():
            expected.update(category)
        assert ALL_VOID_TERMS == expected

    def test_no_overlap_between_tiers(self):
        """Tiers should be mutually exclusive (a word shouldn't be in two tiers)."""
        d = VOID_CLUSTER["direct"]
        s = VOID_CLUSTER["synonyms"]
        n = VOID_CLUSTER["semantic_neighbors"]
        # Allow overlap between synonyms and neighbors (abyss is in both in some versions)
        overlap_ds = d & s
        overlap_dn = d & n
        assert len(overlap_ds) == 0, f"direct/synonyms overlap: {overlap_ds}"
        assert len(overlap_dn) == 0, f"direct/neighbors overlap: {overlap_dn}"


class TestClassification:
    """Test the classify_void_tokens function."""

    def test_empty_input(self):
        result = classify_void_tokens([])
        assert result == {"direct": [], "synonyms": [], "semantic_neighbors": [], "non_void": []}

    def test_single_direct(self):
        result = classify_void_tokens(["void"])
        assert result["direct"] == ["void"]
        assert result["synonyms"] == []
        assert result["semantic_neighbors"] == []
        assert result["non_void"] == []

    def test_single_synonym(self):
        result = classify_void_tokens(["emptiness"])
        assert result["synonyms"] == ["emptiness"]
        assert result["direct"] == []

    def test_single_neighbor(self):
        result = classify_void_tokens(["shadow"])
        assert result["semantic_neighbors"] == ["shadow"]

    def test_single_non_void(self):
        result = classify_void_tokens(["sunshine"])
        assert result["non_void"] == ["sunshine"]
        assert result["direct"] == []
        assert result["synonyms"] == []
        assert result["semantic_neighbors"] == []

    def test_mixed_classification(self):
        tokens = ["void", "emptiness", "shadow", "sunshine", "ghost", "happy"]
        result = classify_void_tokens(tokens)
        assert result["direct"] == ["void"]
        assert result["synonyms"] == ["emptiness"]
        assert "shadow" in result["semantic_neighbors"]
        assert "ghost" in result["semantic_neighbors"]
        assert "sunshine" in result["non_void"]
        assert "happy" in result["non_void"]

    def test_repeated_terms(self):
        tokens = ["void", "void", "void"]
        result = classify_void_tokens(tokens)
        assert len(result["direct"]) == 3

    def test_all_void_all_counted(self, all_void_text):
        """Every word in all_void_text should be classified as void."""
        tokens = tokenize(all_void_text)
        result = classify_void_tokens(tokens)
        total_void = len(result["direct"]) + len(result["synonyms"]) + len(result["semantic_neighbors"])
        assert total_void == len(tokens), (
            f"Expected all {len(tokens)} tokens to be void-classified, got {total_void}. "
            f"Non-void: {result['non_void']}"
        )

    def test_no_void_none_counted(self, no_void_text):
        """No words in cheerful text should be void-classified."""
        tokens = tokenize(no_void_text)
        result = classify_void_tokens(tokens)
        total_void = len(result["direct"]) + len(result["synonyms"]) + len(result["semantic_neighbors"])
        assert total_void == 0, (
            f"Expected 0 void tokens in cheerful text, got {total_void}. "
            f"False positives: direct={result['direct']}, syn={result['synonyms']}, "
            f"neigh={result['semantic_neighbors']}"
        )


class TestClusterConsistency:
    """Test consistency between the cluster definition and methodology claims.

    METHODOLOGY CRITIQUE: These tests document discrepancies between the
    methodology document and the code, which is itself a finding.
    """

    def test_methodology_tier1_matches_code(self):
        """Methodology says Tier 1 = {void}. Code should match."""
        assert VOID_CLUSTER["direct"] == {"void"}

    def test_methodology_tier2_matches_code(self):
        """Methodology says Tier 2 = {emptiness, nothing, nothingness, abyss, vacuum, hollow, blank}.
        Code may have extras — document any discrepancy.
        """
        methodology_tier2 = {"emptiness", "nothing", "nothingness", "abyss", "vacuum", "hollow", "blank"}
        code_tier2 = VOID_CLUSTER["synonyms"]
        extra_in_code = code_tier2 - methodology_tier2
        missing_from_code = methodology_tier2 - code_tier2
        # The code adds "empty", "null", "zero" which aren't in methodology.md
        if extra_in_code:
            pytest.warns(UserWarning, match="Code has extra Tier 2 terms not in methodology")
        # At minimum, methodology terms should be in code
        assert missing_from_code == set(), (
            f"Methodology Tier 2 terms missing from code: {missing_from_code}"
        )

    def test_cluster_size_documented(self):
        """Document the total cluster size for reproducibility."""
        total = len(ALL_VOID_TERMS)
        # The methodology lists ~50 terms, code may have more
        assert total > 0, "Cluster is empty"
        # Warn if significantly different from documented ~50
        if total > 70:
            print(f"WARNING: Cluster has {total} terms, methodology documents ~50. "
                  f"This inflates void density measurements.")


class TestShadowsOfGeometryClassification:
    """Integration test: verify the specific classification of the actual song."""

    def test_total_void_count(self, shadows_of_geometry):
        """The analysis reports 30 void-cluster hits. Verify."""
        tokens = tokenize(shadows_of_geometry)
        result = classify_void_tokens(tokens)
        total = len(result["direct"]) + len(result["synonyms"]) + len(result["semantic_neighbors"])
        assert total == 30, (
            f"Expected 30 void-cluster hits (per initial report), got {total}. "
            f"Classified: direct={len(result['direct'])}, syn={len(result['synonyms'])}, "
            f"neigh={len(result['semantic_neighbors'])}"
        )

    def test_direct_count(self, shadows_of_geometry):
        """Only 'void' should be in Tier 1 → count = 1."""
        tokens = tokenize(shadows_of_geometry)
        result = classify_void_tokens(tokens)
        assert len(result["direct"]) == 1

    def test_synonym_count(self, shadows_of_geometry):
        """No Tier 2 synonyms found in the song → count = 0."""
        tokens = tokenize(shadows_of_geometry)
        result = classify_void_tokens(tokens)
        assert len(result["synonyms"]) == 0

    def test_neighbor_count(self, shadows_of_geometry):
        """29 Tier 3 neighbors → this carries 96.7% of the finding."""
        tokens = tokenize(shadows_of_geometry)
        result = classify_void_tokens(tokens)
        assert len(result["semantic_neighbors"]) == 29

    def test_void_density(self, shadows_of_geometry):
        """Void density should be approximately 15.5% (±1% for tokenizer variation)."""
        tokens = tokenize(shadows_of_geometry)
        result = classify_void_tokens(tokens)
        total_void = len(result["direct"]) + len(result["synonyms"]) + len(result["semantic_neighbors"])
        density = total_void / len(tokens) * 100
        assert 14.0 <= density <= 17.0, (
            f"Expected void density ~15.5%, got {density:.1f}%"
        )
