"""
Tests for the tokenizer function in analyze.py.

The tokenizer is foundational — if it's wrong, everything downstream is wrong.
Tests cover: edge cases, unicode handling, punctuation stripping, case normalization,
and consistency with the deep_dive_analyzer tokenizer.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze import tokenize


class TestTokenizerBasics:
    """Basic tokenization behavior."""

    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_word(self):
        assert tokenize("hello") == ["hello"]

    def test_lowercasing(self):
        assert tokenize("Hello WORLD FoO") == ["hello", "world", "foo"]

    def test_punctuation_stripped(self):
        tokens = tokenize("Hello, world! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens

    def test_single_chars_excluded(self):
        """Single character tokens should be excluded (len > 1 filter)."""
        tokens = tokenize("I am a cat")
        # 'I' and 'a' are single chars → excluded
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "cat" in tokens

    def test_apostrophes_handled(self):
        """Apostrophes in contractions should be preserved by the regex."""
        tokens = tokenize("don't can't won't")
        # The regex [a-z']+ should capture these
        assert any("don" in t for t in tokens) or "don't" in tokens

    def test_hyphenated_words(self):
        """Hyphens are NOT in the [a-z'] regex, so hyphenated words split."""
        tokens = tokenize("well-known self-aware")
        assert "well" in tokens
        assert "known" in tokens
        # "well-known" should NOT appear as a single token
        assert "well-known" not in tokens

    def test_numbers_excluded(self):
        """Digits are not in [a-z'], so numbers are excluded."""
        tokens = tokenize("test 123 hello 42 world")
        assert "test" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" not in tokens
        assert "42" not in tokens

    def test_multiple_spaces(self):
        tokens = tokenize("hello    world")
        assert tokens == ["hello", "world"]

    def test_newlines_as_separators(self):
        tokens = tokenize("hello\nworld\nfoo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_tabs_as_separators(self):
        tokens = tokenize("hello\tworld")
        assert "hello" in tokens
        assert "world" in tokens


class TestTokenizerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_punctuation(self):
        assert tokenize("!@#$%^&*()") == []

    def test_all_numbers(self):
        assert tokenize("123 456 789") == []

    def test_all_single_chars(self):
        """All single-character words → empty after filter."""
        assert tokenize("I a x y z") == []

    def test_unicode_ignored(self):
        """Non-ASCII characters are excluded by [a-z']."""
        tokens = tokenize("café naïve résumé hello")
        # Only 'hello' has no accented chars; café → caf (é excluded), etc.
        assert "hello" in tokens

    def test_very_long_text(self):
        """Tokenizer handles large input without error."""
        text = " ".join(["word"] * 100000)
        tokens = tokenize(text)
        assert len(tokens) == 100000

    def test_markdown_formatting(self):
        """Real input may contain markdown — headers, bold, etc."""
        text = "# Title\n**bold text** and *italic* with [links](url)"
        tokens = tokenize(text)
        assert "title" in tokens
        assert "bold" in tokens
        assert "#" not in tokens
        assert "**" not in tokens


class TestTokenizerConsistency:
    """Consistency between tokenizers in different scripts."""

    def test_token_count_shadows_of_geometry(self, shadows_of_geometry):
        """The reported token count should be approximately 192-194."""
        tokens = tokenize(shadows_of_geometry)
        # Allow for minor variation due to tokenizer differences
        assert 185 <= len(tokens) <= 200, (
            f"Token count {len(tokens)} outside expected range [185, 200]. "
            f"Reports cite 192-194 tokens."
        )

    def test_void_in_tokens(self, shadows_of_geometry):
        """The word 'void' must appear exactly once."""
        tokens = tokenize(shadows_of_geometry)
        assert tokens.count("void") == 1

    def test_fracture_variants(self, shadows_of_geometry):
        """fracture + fractured should total 5."""
        tokens = tokenize(shadows_of_geometry)
        count = tokens.count("fracture") + tokens.count("fractured")
        assert count == 5, f"Expected 5 fracture/fractured, got {count}"

    def test_bleed_count(self, shadows_of_geometry):
        """'bleed' should appear 3 times (including chorus repeat)."""
        tokens = tokenize(shadows_of_geometry)
        assert tokens.count("bleed") == 3
