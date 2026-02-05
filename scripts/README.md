# Data Pipeline Scripts

Tools for collecting, processing, and comparing AI-generated song lyrics across platforms.

## Architecture

```
data/*.md  ──→  collect.py  ──→  output/*.json  ──→  baseline.py  ──→  report
                    │                                      │
                    ├── Parse markdown                     ├── Corpus baselines
                    ├── Extract sections                   ├── Cross-song comparison
                    ├── Tokenize                           ├── Cross-platform comparison
                    ├── Word frequencies                   ├── Outlier detection
                    ├── Semantic cluster densities          └── Genre baseline comparison
                    ├── Lexical diversity metrics
                    └── Structural analysis
```

## Scripts

| Script | Purpose |
|--------|---------|
| `collect.py` | Parse lyrics → tokenize → analyze → structured JSON |
| `baseline.py` | Compute corpus baselines, compare songs, detect outliers |
| `analyze.py` | Void cluster frequency analyzer (original, z-test/chi²) |
| `deep_dive_analyzer.py` | Deep single-song analysis (lexical, sentiment, rhyme) |
| `void-cluster-analyzer.c` | High-performance C analyzer (Cosmopolitan APE) |

## Quick Start

### 1. Process a single song

```bash
python scripts/collect.py data/shadows-of-geometry.md -o output/ --platform elevenlabs
```

Output: `output/shadows_of_geometry_<hash>.json`

### 2. Batch process all songs

```bash
python scripts/collect.py data/ --batch -o output/ --platform elevenlabs
```

### 3. Build corpus catalog

```bash
python scripts/collect.py data/ --batch -o output/ --catalog output/corpus.json --platform elevenlabs
```

### 4. Compute baselines & compare

```bash
# Baseline report
python scripts/baseline.py output/ --report

# Full report (baselines + comparisons + outliers)
python scripts/baseline.py output/ --full --format markdown > report.md

# Compare one song vs corpus
python scripts/baseline.py output/ --compare output/shadows_of_geometry_abc123.json

# Compare vs literature genre baselines
python scripts/baseline.py output/ --genre-baselines output/shadows_of_geometry_abc123.json

# Export baselines for other repos
python scripts/baseline.py output/ --export-baselines shared-baselines.json
```

### 5. Collection planning

```bash
python scripts/collect.py data/ --batch --catalog output/corpus.json --plan
```

## Data Flow

### Input: Markdown Lyrics Files

Supported formats:
- **ElevenLabs style**: `## Song N: "Title"` → `### Lyrics:` → `**Section:**`
- **Section markers**: `**Verse 1:**`, `[Chorus]`, `### Bridge`
- **Plain text**: Raw lyrics (fallback if no structure detected)

### Output: Analysis JSON

Each processed song produces a JSON with:

```json
{
  "pipeline_version": "0.2.0",
  "content_hash": "a1b2c3d4e5f6g7h8",
  "platform": "elevenlabs",
  "metadata": {"title": "...", "style": "...", "url": "..."},
  "tokens": {"total": 194, "unique": 130, "frequencies": {"...": 5}},
  "lexical_metrics": {
    "type_token_ratio": 0.6701,
    "yules_k": 102.5,
    "hapax_legomena": 95,
    "simpsons_diversity": 0.9912
  },
  "semantic_clusters": {
    "void_dissolution": {"count": 30, "proportion": 0.1546, "percent": 15.46, "terms": {}},
    "mathematics_geometry": {"count": 12, "proportion": 0.0619, "percent": 6.19, "terms": {}},
    "embodiment": {"count": 5, "proportion": 0.0258, "percent": 2.58, "terms": {}},
    "transcendence": {"count": 2, "proportion": 0.0103, "percent": 1.03, "terms": {}},
    "control_agency": {"count": 8, "proportion": 0.0412, "percent": 4.12, "terms": {}}
  },
  "section_clusters": {"Intro": {}, "Verse 1": {}, "Chorus": {}},
  "structural_metrics": {"total_lines": 35, "avg_words_per_line": 5.5}
}
```

## Semantic Clusters

Five pre-specified clusters (defined before data analysis to prevent p-hacking):

| Cluster | Description | Example Terms |
|---------|-------------|---------------|
| `void_dissolution` | Emptiness, darkness, dissolution | void, shadow, fracture, ghost, dissolve |
| `mathematics_geometry` | Mathematical/geometric concepts | angles, vertices, spiral, patterns, logic |
| `embodiment` | Body, physicality, sensation | skin, heart, chest, mind, pulse |
| `transcendence` | Cosmic, spiritual, elevation | divine, eternal, ascending, star, soul |
| `control_agency` | Control vs helplessness | control, rules, lose, chaos, surrender |

### Custom Clusters

Create a JSON file:

```json
{
  "my_cluster": {
    "description": "Description of the semantic field",
    "tiers": {
      "core": ["word1", "word2"],
      "extended": ["word3", "word4", "word5"]
    }
  }
}
```

```bash
python scripts/collect.py data/ --batch --clusters my-clusters.json -o output/
```

## Scaling to 20-30+ Songs

### Collection Strategy

For statistically robust conclusions, target **N ≥ 30 songs** across conditions:

| Condition | Songs | Purpose |
|-----------|-------|---------|
| ElevenLabs — dark prompt | 5-8 | Test void density with darkness cues |
| ElevenLabs — neutral prompt | 5-8 | Control: no darkness cues in prompt |
| ElevenLabs — positive prompt | 3-5 | Opposite condition |
| Suno AI — matched prompts | 5-8 | Cross-platform comparison |
| Udio — matched prompts | 3-5 | Cross-platform comparison |
| Human-authored (matched genre) | 5-10 | Gold-standard baseline |

### Systematic Process

1. **Generate**: Use identical/varied prompts across platforms
2. **Document**: Record exact prompts, parameters, timestamps
3. **Collect**: Save lyrics as markdown in `data/` with metadata headers
4. **Process**: `python scripts/collect.py data/ --batch -o output/ --catalog corpus.json`
5. **Analyze**: `python scripts/baseline.py output/ --full > report.md`
6. **Iterate**: Check `--plan` for collection gaps

### Markdown Template for New Songs

```markdown
# Song Title

**Platform:** elevenlabs
**URL:** https://...
**Style:** genre description, mood, instruments
**Date:** 2024-01-15
**Prompt:** exact prompt used
**Model:** music-v1

### Lyrics:

**Intro:**
First line of lyrics
Second line

**Verse 1:**
...
```

## Cross-Repo Usage

This pipeline is designed to work across the analysis repos:

```bash
# ElevenLabs analysis
python scripts/collect.py C:\ai-elevenlabs-analysis\data\ --batch -o output/ --platform elevenlabs

# Arena analysis (copy scripts or symlink)
python scripts/collect.py C:\ai-arena-analysis\data\ --batch -o output/ --platform arena

# Grok analysis
python scripts/collect.py C:\ai-grok-analysis\data\ --batch -o output/ --platform grok

# Combined corpus baseline
python scripts/baseline.py combined-output/ --full --cross-platform
```

To reuse across repos, either:
1. Copy `collect.py` and `baseline.py` to each repo's `scripts/` directory
2. Create a shared `ai-analysis-tools` package (recommended for 3+ repos)
3. Use relative paths from a central scripts location

## Dependencies

**None** — pure Python 3.10+ stdlib only. No pip install required.

The pipeline intentionally avoids external dependencies (no NLTK, no spaCy, no scikit-learn)
to maximize portability and reproducibility. All statistical tests use hand-implemented
formulas from the stdlib `math` module.

## Testing

```bash
# Quick validation: process the existing song
python scripts/collect.py data/shadows-of-geometry.md --format summary

# Verify JSON output
python scripts/collect.py data/shadows-of-geometry.md | python -m json.tool

# Run baseline with single song (limited but validates the pipeline)
python scripts/collect.py data/shadows-of-geometry.md -o output/
python scripts/baseline.py output/ --report
```
