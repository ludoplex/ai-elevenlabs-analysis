#!/usr/bin/env python3
"""
Lyrics Data Collection & Processing Pipeline

Parses song lyrics from markdown files, tokenizes, computes word frequencies,
semantic cluster densities, and outputs structured JSON for downstream analysis.

Designed to be reusable across analysis repos (ElevenLabs, Arena, Grok).

Usage:
    # Single file
    python collect.py data/shadows-of-geometry.md -o output/

    # Batch: all markdown files in a directory
    python collect.py data/ -o output/ --batch

    # Custom semantic clusters
    python collect.py data/ -o output/ --batch --clusters clusters.json

    # Append to existing corpus catalog
    python collect.py data/new-song.md -o output/ --catalog corpus.json

    # Platform tagging
    python collect.py data/ -o output/ --batch --platform elevenlabs --model "music-v1"
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── Version ─────────────────────────────────────────────────────
__version__ = "0.2.0"

# ─── Default Semantic Clusters ───────────────────────────────────
# Pre-specified BEFORE looking at data (methodology.md §1)
DEFAULT_CLUSTERS = {
    "void_dissolution": {
        "description": "Void, emptiness, dissolution, darkness semantic field",
        "tiers": {
            "direct": [
                "void",
            ],
            "synonyms": [
                "emptiness", "nothing", "nothingness", "abyss", "vacuum",
                "hollow", "blank", "empty", "null", "zero", "oblivion",
            ],
            "semantic_neighbors": [
                "shadow", "shadows", "ghost", "ghosts", "vanish", "vanished",
                "dissolve", "dissolved", "silence", "silenced", "absence",
                "lost", "darkness", "dark", "night", "bleed", "bleeding",
                "fracture", "fractured", "fractures", "chaos", "cage", "caged",
                "drift", "drifting", "fray", "frayed", "twisted", "edges",
                "edge", "whisper", "whispers", "fade", "faded", "fading",
                "shatter", "shattered", "crumble", "collapse", "collapsed",
                "erode", "eroded", "decay", "decayed", "wither", "withered",
                "extinct", "chasm", "depths",
                "forgotten", "forsaken", "abandoned", "desolate", "barren",
            ],
        },
    },
    "mathematics_geometry": {
        "description": "Mathematical, geometric, and structural concepts",
        "tiers": {
            "direct": [
                "geometry", "geometric", "mathematics", "mathematical",
                "equation", "theorem", "proof", "axiom",
            ],
            "related": [
                "angles", "vertices", "vertex", "patterns", "numbers",
                "measure", "logic", "counting", "polyrhythmic", "unequal",
                "spiral", "line", "lines", "signs", "sign", "dimension",
                "symmetry", "asymmetry", "fractal", "infinite", "infinity",
                "ratio", "proportion", "sequence", "algorithm", "binary",
                "calculus", "vector", "matrix", "plane", "curve",
            ],
        },
    },
    "embodiment": {
        "description": "Body, physicality, sensation",
        "tiers": {
            "direct": [
                "body", "flesh", "bone", "bones", "skin", "blood",
            ],
            "related": [
                "heart", "chest", "mind", "minds", "brain", "eyes",
                "hands", "fingers", "breath", "breathing", "pulse",
                "veins", "spine", "skull", "teeth", "tongue",
                "touch", "feel", "feeling", "pain", "ache",
            ],
        },
    },
    "transcendence": {
        "description": "Transcendence, cosmic, spiritual elevation",
        "tiers": {
            "direct": [
                "transcend", "transcendence", "enlighten", "enlightenment",
                "divine", "sacred", "holy", "cosmic",
            ],
            "related": [
                "heaven", "celestial", "eternal", "eternity", "infinite",
                "ascend", "ascending", "rise", "rising", "elevate",
                "spirit", "soul", "awaken", "awakening", "illuminate",
                "radiant", "luminous", "glory", "sublime", "beyond",
                "universe", "star", "stars", "constellation",
            ],
        },
    },
    "control_agency": {
        "description": "Control, power, agency vs. helplessness",
        "tiers": {
            "control": [
                "control", "rules", "reason", "logic", "gravity",
                "rituals", "measure", "counting", "command", "order",
                "power", "force", "master", "reign", "dominate",
            ],
            "loss_of_control": [
                "lose", "lost", "broken", "breaking", "unaligned",
                "tangled", "unequal", "chaos", "fray", "fracture",
                "fractured", "collide", "dissolve", "drift", "helpless",
                "surrender", "submit", "fall", "falling", "drown",
            ],
        },
    },
}


# ─── Markdown Lyrics Parser ─────────────────────────────────────

class LyricParser:
    """Parse lyrics from markdown files with various formats.

    Supports formats:
    - ElevenLabs style: ## Song N: "Title" with ### Lyrics header
    - Simple: markdown with **Section:** headers
    - Plain: raw text (no markdown structure)
    - Arena/Grok style: metadata header + lyrics body
    """

    # Markdown structural elements to strip
    STRIP_PATTERNS = [
        r"^#{1,6}\s+.*$",           # Headers (keep content separately)
        r"^\*\*[^*]+\*\*\s*$",      # Bold-only lines (metadata)
        r"^---+\s*$",               # Horizontal rules
        r"^\|.*\|.*$",              # Table rows
        r"^>\s+",                   # Blockquotes
    ]

    # Section header patterns (capture the section name)
    SECTION_PATTERNS = [
        r"^\*\*(.+?)(?::)?\*\*\s*$",          # **Verse 1:** or **Chorus**
        r"^###?\s+(.+?)\s*$",                  # ## Section or ### Section
        r"^\[(.+?)\]\s*$",                     # [Verse 1]
        r"^(?:Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Hook|Refrain)\s*\d*\s*[:.]?\s*$",
    ]

    # Metadata patterns to extract
    METADATA_PATTERNS = {
        "title": [
            r'##\s+Song\s+\d+:\s*"(.+?)"',
            r'#\s+"(.+?)"',
            r"title:\s*(.+)",
            r'\*\*Title:\*\*\s*"?(.+?)"?\s*$',
        ],
        "style": [
            r"\*\*Style:\*\*\s*(.+)",
            r"style:\s*(.+)",
            r"\*\*Genre:\*\*\s*(.+)",
        ],
        "url": [
            r"\*\*(?:Project\s+)?URL:\*\*\s*(https?://\S+)",
            r"url:\s*(https?://\S+)",
        ],
        "platform": [
            r"\*\*Platform:\*\*\s*(.+)",
            r"platform:\s*(.+)",
        ],
    }

    def __init__(self):
        self._compiled_sections = [re.compile(p, re.MULTILINE) for p in self.SECTION_PATTERNS]

    def parse_file(self, filepath: str | Path) -> dict:
        """Parse a markdown file into structured song data.

        Returns dict with:
            metadata: {title, style, url, platform, source_file, ...}
            sections: {section_name: lyrics_text, ...}
            full_lyrics: str (all lyrics concatenated)
            raw_text: str (original file content)
        """
        filepath = Path(filepath)
        text = filepath.read_text(encoding="utf-8")
        return self.parse_text(text, source_file=str(filepath))

    def parse_text(self, text: str, source_file: str = None) -> dict:
        """Parse lyrics from markdown text."""
        metadata = self._extract_metadata(text, source_file)
        sections = self._extract_sections(text)
        full_lyrics = "\n".join(sections.values())

        # If no sections found, treat entire cleaned text as lyrics
        if not sections or not full_lyrics.strip():
            cleaned = self._strip_markdown_structure(text)
            if cleaned.strip():
                sections = {"full": cleaned.strip()}
                full_lyrics = cleaned.strip()

        return {
            "metadata": metadata,
            "sections": sections,
            "full_lyrics": full_lyrics,
            "raw_text": text,
        }

    def _extract_metadata(self, text: str, source_file: str = None) -> dict:
        """Extract metadata fields from markdown text."""
        meta = {"source_file": source_file}
        for field, patterns in self.METADATA_PATTERNS.items():
            for pattern in patterns:
                m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if m:
                    meta[field] = m.group(1).strip()
                    break
        return meta

    def _extract_sections(self, text: str) -> dict:
        """Extract labeled sections and their lyrics content."""
        sections = {}
        current_section = None
        current_lines = []
        in_lyrics = False

        # Check if there's a "### Lyrics:" header
        lyrics_start = re.search(r"^###?\s+Lyrics:?\s*$", text, re.MULTILINE)
        analysis_start = re.search(r"^##\s+(?:Preliminary|Analysis)", text, re.MULTILINE)

        if lyrics_start:
            start_pos = lyrics_start.end()
            end_pos = analysis_start.start() if analysis_start else len(text)
            text_to_parse = text[start_pos:end_pos]
        else:
            text_to_parse = text

        for line in text_to_parse.split("\n"):
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            # Check for section header
            section_name = self._match_section_header(stripped)
            if section_name:
                # Save previous section
                if current_section and current_lines:
                    sections[current_section] = self._clean_section("\n".join(current_lines))
                current_section = section_name
                current_lines = []
                continue

            # Skip metadata lines, tables, horizontal rules
            if self._is_metadata_line(stripped):
                continue

            # Accumulate lyrics
            if current_section:
                # Strip inline markdown formatting but keep text
                cleaned_line = self._clean_line(stripped)
                if cleaned_line:
                    current_lines.append(cleaned_line)

        # Save last section
        if current_section and current_lines:
            sections[current_section] = self._clean_section("\n".join(current_lines))

        return sections

    def _match_section_header(self, line: str) -> str | None:
        """Check if a line is a section header, return section name or None."""
        # **Section Name:** pattern
        m = re.match(r"^\*\*(.+?)(?::)?\*\*\s*$", line)
        if m:
            name = m.group(1).strip()
            # Filter out metadata-like bold lines
            if any(kw in name.lower() for kw in ["url", "style", "project", "genre", "platform"]):
                return None
            return name

        # [Section Name] pattern
        m = re.match(r"^\[(.+?)\]\s*$", line)
        if m:
            return m.group(1).strip()

        return None

    def _is_metadata_line(self, line: str) -> bool:
        """Check if a line is metadata (not lyrics)."""
        if re.match(r"^---+$", line):
            return True
        if re.match(r"^\|.*\|", line):
            return True
        if re.match(r"^#{1,3}\s+", line) and not re.match(r"^###?\s+(?:Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus)", line, re.IGNORECASE):
            return True
        # Bold metadata lines like **Style:** ...
        if re.match(r"^\*\*(?:Style|URL|Project|Genre|Platform|Date|Model):", line, re.IGNORECASE):
            return True
        return False

    def _clean_line(self, line: str) -> str:
        """Remove inline markdown, keep lyric text."""
        # Remove bold/italic markers
        line = re.sub(r"\*{1,3}", "", line)
        # Remove inline code
        line = re.sub(r"`[^`]+`", "", line)
        # Remove links [text](url) → text
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        return line.strip()

    def _clean_section(self, text: str) -> str:
        """Clean up section text (strip trailing blank lines)."""
        return text.strip()

    def _strip_markdown_structure(self, text: str) -> str:
        """Remove all markdown structure, return plain text."""
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if self._is_metadata_line(stripped):
                continue
            cleaned = self._clean_line(stripped)
            # Skip section headers
            if self._match_section_header(stripped):
                continue
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines)


# ─── Tokenizer ───────────────────────────────────────────────────

class Tokenizer:
    """Configurable word tokenizer with normalization."""

    def __init__(self, min_length: int = 2, lowercase: bool = True,
                 strip_possessives: bool = True,
                 stopwords: set[str] | None = None):
        self.min_length = min_length
        self.lowercase = lowercase
        self.strip_possessives = strip_possessives
        self.stopwords = stopwords or set()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        # Extract word tokens (allow apostrophes within words)
        words = re.findall(r"[a-z']+", text) if self.lowercase else re.findall(r"[a-zA-Z']+", text)
        tokens = []
        for w in words:
            # Strip leading/trailing apostrophes
            w = w.strip("'")
            if self.strip_possessives:
                w = re.sub(r"'s$", "", w)
            # Skip vocal expressions commonly in music
            if w in ("ahh", "aah", "ooh", "ooo", "mmh", "mmm", "oh", "ah", "na", "la", "da"):
                continue
            if len(w) >= self.min_length and w not in self.stopwords:
                tokens.append(w)
        return tokens

    def tokenize_preserving_position(self, text: str) -> list[dict]:
        """Tokenize with position information (for concordance analysis)."""
        if self.lowercase:
            text_lower = text.lower()
        else:
            text_lower = text
        results = []
        for m in re.finditer(r"[a-z']+", text_lower):
            word = m.group().strip("'")
            if self.strip_possessives:
                word = re.sub(r"'s$", "", word)
            if len(word) >= self.min_length and word not in self.stopwords:
                results.append({
                    "token": word,
                    "start": m.start(),
                    "end": m.end(),
                    "line": text[:m.start()].count("\n") + 1,
                })
        return results


# ─── Semantic Cluster Analyzer ───────────────────────────────────

class SemanticClusterAnalyzer:
    """Analyze token distributions across configurable semantic clusters."""

    def __init__(self, clusters: dict | None = None):
        self.clusters = clusters or DEFAULT_CLUSTERS
        # Build flat lookup: token → (cluster_name, tier_name)
        self._lookup = {}
        for cname, cdata in self.clusters.items():
            for tname, terms in cdata.get("tiers", {}).items():
                for term in terms:
                    # First cluster to claim a term wins
                    if term not in self._lookup:
                        self._lookup[term] = (cname, tname)

    def analyze(self, tokens: list[str]) -> dict:
        """Compute semantic cluster densities for a token list.

        Returns dict with per-cluster stats:
            {cluster_name: {count, proportion, tier_breakdown, terms}}
        """
        total = len(tokens)
        if total == 0:
            return {}

        results = {}
        for cname, cdata in self.clusters.items():
            tier_counts = defaultdict(int)
            term_counts = Counter()
            for token in tokens:
                if token in self._lookup and self._lookup[token][0] == cname:
                    tier = self._lookup[token][1]
                    tier_counts[tier] += 1
                    term_counts[token] += 1

            cluster_total = sum(tier_counts.values())
            results[cname] = {
                "description": cdata.get("description", ""),
                "count": cluster_total,
                "proportion": round(cluster_total / total, 6),
                "percent": round(cluster_total / total * 100, 2),
                "tier_breakdown": dict(tier_counts),
                "terms": dict(term_counts.most_common(50)),
            }

        # Also compute overlap/unclustered stats
        clustered_count = sum(r["count"] for r in results.values())
        results["_meta"] = {
            "total_tokens": total,
            "total_clustered": clustered_count,
            "clustered_proportion": round(clustered_count / total, 6),
            "unclustered_count": total - clustered_count,
        }

        return results

    def classify_token(self, token: str) -> tuple[str, str] | None:
        """Return (cluster_name, tier_name) for a token, or None."""
        return self._lookup.get(token)


# ─── Lexical Diversity Metrics ───────────────────────────────────

def compute_lexical_metrics(tokens: list[str]) -> dict:
    """Compute standard lexical diversity metrics."""
    total = len(tokens)
    if total == 0:
        return {}

    freq = Counter(tokens)
    types = len(freq)

    # Type-Token Ratio
    ttr = types / total

    # Hapax legomena (frequency = 1)
    hapax = [w for w, c in freq.items() if c == 1]
    n_hapax = len(hapax)

    # Hapax dis-legomena (frequency = 2)
    dis_legomena = [w for w, c in freq.items() if c == 2]

    # Yule's K (vocabulary richness — lower = more diverse)
    freq_spectrum = Counter(freq.values())
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    yules_k = 10000 * (m2 - total) / (total * total) if total > 1 else 0

    # Brunet's W
    brunets_w = total ** (types ** -0.172) if types > 0 else 0

    # Honore's R
    if 0 < n_hapax < types:
        honores_r = 100 * math.log(total) / (1 - n_hapax / types)
    else:
        honores_r = 0

    # Simpson's Diversity Index (probability two random tokens are different)
    simpsons_d = 1 - sum(c * (c - 1) for c in freq.values()) / (total * (total - 1)) if total > 1 else 0

    return {
        "total_tokens": total,
        "unique_types": types,
        "type_token_ratio": round(ttr, 4),
        "hapax_legomena": n_hapax,
        "hapax_ratio": round(n_hapax / types, 4) if types > 0 else 0,
        "dis_legomena": len(dis_legomena),
        "yules_k": round(yules_k, 2),
        "brunets_w": round(brunets_w, 2),
        "honores_r": round(honores_r, 2),
        "simpsons_diversity": round(simpsons_d, 4),
        "frequency_spectrum": {str(k): v for k, v in sorted(freq_spectrum.items())},
    }


# ─── Structural Metrics ─────────────────────────────────────────

def compute_structural_metrics(sections: dict, tokenizer: Tokenizer) -> dict:
    """Compute per-section and overall structural metrics."""
    section_stats = {}
    all_lines = []

    for name, text in sections.items():
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        tokens = tokenizer.tokenize(text)
        words_per_line = [len(tokenizer.tokenize(l)) for l in lines]

        avg_wpl = sum(words_per_line) / len(words_per_line) if words_per_line else 0
        word_lengths = [len(t) for t in tokens]

        section_stats[name] = {
            "line_count": len(lines),
            "token_count": len(tokens),
            "words_per_line": words_per_line,
            "avg_words_per_line": round(avg_wpl, 2),
            "avg_word_length": round(sum(word_lengths) / len(word_lengths), 2) if word_lengths else 0,
        }
        all_lines.extend(lines)

    # Overall metrics
    all_wpl = []
    for s in section_stats.values():
        all_wpl.extend(s["words_per_line"])

    avg_wpl = sum(all_wpl) / len(all_wpl) if all_wpl else 0
    std_wpl = math.sqrt(sum((w - avg_wpl) ** 2 for w in all_wpl) / len(all_wpl)) if all_wpl else 0
    cv_wpl = std_wpl / avg_wpl if avg_wpl > 0 else 0

    return {
        "total_lines": len(all_lines),
        "total_sections": len(sections),
        "avg_words_per_line": round(avg_wpl, 2),
        "std_words_per_line": round(std_wpl, 2),
        "cv_words_per_line": round(cv_wpl, 4),
        "section_stats": section_stats,
    }


# ─── Content Hashing ────────────────────────────────────────────

def content_hash(text: str) -> str:
    """SHA-256 of normalized lyrics for dedup/identity."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ─── Full Pipeline ───────────────────────────────────────────────

class CollectionPipeline:
    """Orchestrates: parse → tokenize → analyze → output."""

    def __init__(self, clusters: dict | None = None,
                 tokenizer: Tokenizer | None = None,
                 platform: str = "unknown",
                 model: str = "unknown"):
        self.parser = LyricParser()
        self.tokenizer = tokenizer or Tokenizer()
        self.cluster_analyzer = SemanticClusterAnalyzer(clusters)
        self.platform = platform
        self.model = model

    def process_file(self, filepath: str | Path) -> dict:
        """Process a single markdown file through the full pipeline."""
        filepath = Path(filepath)
        parsed = self.parser.parse_file(filepath)
        return self._analyze_parsed(parsed, filepath)

    def process_text(self, text: str, source: str = "inline") -> dict:
        """Process raw text through the pipeline."""
        parsed = self.parser.parse_text(text, source_file=source)
        return self._analyze_parsed(parsed)

    def _analyze_parsed(self, parsed: dict, filepath: Path = None) -> dict:
        """Run analysis on parsed data."""
        lyrics = parsed["full_lyrics"]
        if not lyrics.strip():
            return {"error": "No lyrics found", "metadata": parsed["metadata"]}

        tokens = self.tokenizer.tokenize(lyrics)
        freq = Counter(tokens)

        # Core analyses
        cluster_analysis = self.cluster_analyzer.analyze(tokens)
        lexical_metrics = compute_lexical_metrics(tokens)
        structural_metrics = compute_structural_metrics(parsed["sections"], self.tokenizer)

        # Per-section cluster analysis
        section_clusters = {}
        for section_name, section_text in parsed["sections"].items():
            section_tokens = self.tokenizer.tokenize(section_text)
            if section_tokens:
                section_clusters[section_name] = self.cluster_analyzer.analyze(section_tokens)

        # Build result
        result = {
            "pipeline_version": __version__,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash(lyrics),
            "platform": parsed["metadata"].get("platform", self.platform),
            "model": self.model,
            "metadata": parsed["metadata"],
            "tokens": {
                "total": len(tokens),
                "unique": len(freq),
                "frequencies": dict(freq.most_common(100)),
            },
            "lexical_metrics": lexical_metrics,
            "semantic_clusters": cluster_analysis,
            "section_clusters": section_clusters,
            "structural_metrics": structural_metrics,
            "sections": list(parsed["sections"].keys()),
        }

        return result

    def process_batch(self, directory: str | Path, pattern: str = "*.md") -> list[dict]:
        """Process all matching files in a directory."""
        directory = Path(directory)
        results = []
        for filepath in sorted(directory.glob(pattern)):
            try:
                result = self.process_file(filepath)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "metadata": {"source_file": str(filepath)},
                })
        return results


# ─── Corpus Catalog ──────────────────────────────────────────────

class CorpusCatalog:
    """Track collected songs across platforms for corpus management.

    The catalog is a JSON file mapping content_hash → entry metadata.
    Supports dedup, cross-platform tracking, and collection planning.
    """

    def __init__(self, catalog_path: str | Path):
        self.path = Path(catalog_path)
        self.entries = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, analysis_result: dict) -> str:
        """Add an analysis result to the catalog. Returns content_hash."""
        chash = analysis_result.get("content_hash", "unknown")
        meta = analysis_result.get("metadata", {})

        entry = {
            "title": meta.get("title", "untitled"),
            "platform": analysis_result.get("platform", "unknown"),
            "model": analysis_result.get("model", "unknown"),
            "style": meta.get("style", ""),
            "source_file": meta.get("source_file", ""),
            "url": meta.get("url", ""),
            "collected_at": analysis_result.get("collected_at", ""),
            "total_tokens": analysis_result.get("tokens", {}).get("total", 0),
            "void_density": analysis_result.get("semantic_clusters", {}).get(
                "void_dissolution", {}
            ).get("percent", 0),
        }

        self.entries[chash] = entry
        return chash

    def save(self):
        """Write catalog to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def summary(self) -> dict:
        """Return corpus summary statistics."""
        if not self.entries:
            return {"count": 0}

        platforms = Counter(e["platform"] for e in self.entries.values())
        tokens = [e["total_tokens"] for e in self.entries.values() if e["total_tokens"] > 0]
        densities = [e["void_density"] for e in self.entries.values() if e["void_density"] > 0]

        return {
            "count": len(self.entries),
            "platforms": dict(platforms),
            "total_tokens": sum(tokens),
            "avg_tokens_per_song": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "void_density_range": {
                "min": round(min(densities), 2) if densities else 0,
                "max": round(max(densities), 2) if densities else 0,
                "mean": round(sum(densities) / len(densities), 2) if densities else 0,
            },
        }

    def collection_plan(self, target: int = 30) -> dict:
        """Generate a collection plan to reach target N."""
        current = len(self.entries)
        needed = max(0, target - current)
        platforms = Counter(e["platform"] for e in self.entries.values())

        plan = {
            "current_n": current,
            "target_n": target,
            "needed": needed,
            "current_platforms": dict(platforms),
            "recommendations": [],
        }

        if needed > 0:
            # Recommend balanced collection
            plan["recommendations"] = [
                f"Collect {max(1, needed // 3)} more songs from ElevenLabs with VARIED prompts",
                f"Collect {max(1, needed // 3)} songs from competing platforms (Suno, Udio) for cross-platform comparison",
                f"Collect {max(1, needed // 3)} human-authored songs in matching genres for baseline",
                "Ensure prompt diversity: dark, neutral, positive, abstract, narrative",
                "Document exact prompts/parameters for reproducibility",
            ]

        return plan


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lyrics Data Collection & Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect.py data/shadows-of-geometry.md
  python collect.py data/ --batch -o output/
  python collect.py data/ --batch --catalog corpus.json --platform elevenlabs
  python collect.py data/ --batch --clusters custom-clusters.json
        """,
    )
    parser.add_argument("input", help="Markdown file or directory (with --batch)")
    parser.add_argument("-o", "--output", help="Output directory for JSON results")
    parser.add_argument("--batch", action="store_true", help="Process all .md files in directory")
    parser.add_argument("--pattern", default="*.md", help="Glob pattern for batch mode (default: *.md)")
    parser.add_argument("--catalog", help="Path to corpus catalog JSON (append mode)")
    parser.add_argument("--platform", default="elevenlabs", help="Platform tag (default: elevenlabs)")
    parser.add_argument("--model", default="unknown", help="Model identifier")
    parser.add_argument("--clusters", help="Custom semantic clusters JSON file")
    parser.add_argument("--format", choices=["json", "summary"], default="json", help="Output format")
    parser.add_argument("--min-token-length", type=int, default=2, help="Min token length (default: 2)")
    parser.add_argument("--plan", action="store_true", help="Show collection plan and exit")

    args = parser.parse_args()

    # Load custom clusters if specified
    clusters = None
    if args.clusters:
        clusters = json.loads(Path(args.clusters).read_text(encoding="utf-8"))

    # Tokenizer config
    tokenizer = Tokenizer(min_length=args.min_token_length)

    # Pipeline
    pipeline = CollectionPipeline(
        clusters=clusters,
        tokenizer=tokenizer,
        platform=args.platform,
        model=args.model,
    )

    # Catalog
    catalog = CorpusCatalog(args.catalog) if args.catalog else None

    # Collection plan mode
    if args.plan:
        if catalog:
            plan = catalog.collection_plan()
            print(json.dumps(plan, indent=2))
        else:
            print("Error: --plan requires --catalog", file=sys.stderr)
            sys.exit(1)
        return

    # Process
    input_path = Path(args.input)
    if args.batch:
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory", file=sys.stderr)
            sys.exit(1)
        results = pipeline.process_batch(input_path, args.pattern)
    else:
        if not input_path.is_file():
            print(f"Error: {input_path} is not a file", file=sys.stderr)
            sys.exit(1)
        results = [pipeline.process_file(input_path)]

    # Output
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        # Add to catalog
        if catalog and "error" not in result:
            catalog.add(result)

        if args.format == "summary":
            _print_summary(result)
        else:
            if output_dir:
                # Write individual JSON file
                chash = result.get("content_hash", "unknown")
                title = result.get("metadata", {}).get("title", "untitled")
                safe_name = re.sub(r"[^\w\-]", "_", title.lower())[:50]
                outpath = output_dir / f"{safe_name}_{chash}.json"
                outpath.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"  -> {outpath}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save catalog
    if catalog:
        catalog.save()
        summary = catalog.summary()
        print(f"\nCorpus catalog: {catalog.path}", file=sys.stderr)
        print(f"  Songs: {summary['count']}", file=sys.stderr)
        print(f"  Total tokens: {summary.get('total_tokens', 0)}", file=sys.stderr)
        if summary.get("void_density_range"):
            vdr = summary["void_density_range"]
            print(f"  Void density: {vdr['min']}% - {vdr['max']}% (mean {vdr['mean']}%)", file=sys.stderr)


def _print_summary(result: dict):
    """Print a human-readable summary of analysis results."""
    meta = result.get("metadata", {})
    tokens = result.get("tokens", {})
    clusters = result.get("semantic_clusters", {})
    lexical = result.get("lexical_metrics", {})

    title = meta.get("title", meta.get("source_file", "unknown"))
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Platform: {result.get('platform', 'unknown')}")
    print(f"  Tokens: {tokens.get('total', 0)} total, {tokens.get('unique', 0)} unique")
    print(f"  TTR: {lexical.get('type_token_ratio', 0):.3f}")
    print(f"  Yule's K: {lexical.get('yules_k', 0):.1f}")

    # Cluster densities
    print(f"\n  Semantic Clusters:")
    for cname, cdata in clusters.items():
        if cname.startswith("_"):
            continue
        if cdata.get("count", 0) > 0:
            print(f"    {cname}: {cdata['count']} ({cdata['percent']}%)")
            if cdata.get("terms"):
                top3 = list(cdata["terms"].items())[:3]
                terms_str = ", ".join(f"{t}({c})" for t, c in top3)
                print(f"      Top: {terms_str}")

    # Top words
    if tokens.get("frequencies"):
        top10 = list(tokens["frequencies"].items())[:10]
        print(f"\n  Top words: {', '.join(f'{w}({c})' for w, c in top10)}")


if __name__ == "__main__":
    main()
