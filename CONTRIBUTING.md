# Contributing to AI ElevenLabs Music Analysis

Thank you for your interest in contributing! This project analyzes ElevenLabs AI-generated music lyrics for statistical anomalies in semantic clustering — specifically, overrepresentation of void/dissolution/emptiness themes.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/ludoplex/ai-elevenlabs-analysis.git
cd ai-elevenlabs-analysis

# Install dev tools (optional, for linting)
pip install ruff mypy

# Run the full analysis pipeline
make all

# Or run individual steps
make validate      # Check data file format
make analyze       # Run void cluster analysis
make deep-dive     # Run deep dive analyzer
make lint          # Lint Python scripts
```

## Project Structure

```
ai-elevenlabs-analysis/
├── data/                    # Song lyrics and metadata (one file per song)
│   └── shadows-of-geometry.md
├── scripts/                 # Analysis tools
│   ├── analyze.py           # Core void cluster frequency analyzer
│   ├── deep_dive_analyzer.py # Extended analysis (lexical, sentiment, etc.)
│   └── void-cluster-analyzer.c  # C implementation (performance)
├── analysis/                # Reports and findings
│   ├── initial-report.md
│   ├── elevenlabs-deep-dive.md
│   └── methodology.md
├── .github/
│   ├── workflows/
│   │   ├── analyze.yml      # CI: runs analysis on push/PR
│   │   └── lint.yml         # CI: lints all scripts
│   └── dependabot.yml       # Keeps dependencies updated
├── Makefile                 # Local dev commands
├── CONTRIBUTING.md          # ← You are here
└── README.md
```

## How to Contribute

### 1. Add New Song Data

This is the **most valuable contribution** — we need more data points.

**Format:** Create a markdown file in `data/` with this structure:

```markdown
# ElevenLabs Music - Generated Songs Data

## Song N: "Song Title"
**Project URL:** https://elevenlabs.io/app/music/project/YOUR_PROJECT_ID
**Style:** [paste the exact style/prompt used]
**Generated:** YYYY-MM-DD

### Lyrics:

**Intro:**
[lyrics here]

**Verse 1:**
[lyrics here]

**Chorus:**
[lyrics here]

[... etc ...]
```

**Requirements:**
- One file per song: `data/song-name-kebab-case.md`
- Include the **exact prompt/style tags** used (critical for confound analysis)
- Include the **project URL** for verification
- Include the **generation date** if known
- Use UTF-8 encoding
- Preserve the original section structure (Intro, Verse, Chorus, etc.)

**Especially needed:**
- Songs generated with **neutral prompts** (no darkness cues) — for control comparison
- Songs from **different genres** — to test cross-genre void clustering
- **Multiple generations with identical prompts** — to test reproducibility
- Songs from **other AI platforms** (Suno, Udio) — for cross-platform comparison

### 2. Improve Analysis Scripts

The analysis pipeline lives in `scripts/`. Key areas for improvement:

- **More robust tokenization** — current regex-based tokenizer is simple
- **Additional statistical tests** — bootstrap confidence intervals, permutation tests
- **Semantic embedding analysis** — compare void cluster using word2vec/GloVe distances
- **Cross-song aggregation** — combined statistics across multiple songs
- **Visualization** — generate plots (matplotlib/plotly) in CI

Before modifying `analyze.py`, read `analysis/methodology.md` to understand the pre-specified void cluster and why it was defined before data analysis.

### 3. Add Baseline Corpora

We need reference data from human-authored lyrics:

- **Prog rock:** Yes, Genesis, Rush, Dream Theater, Tool
- **Dark prog:** Porcupine Tree, dark-era Pink Floyd
- **Metal/doom:** My Dying Bride, Shape of Despair
- **General rock:** broad sample for normalization

Place baseline data in `data/baselines/` with clear attribution and licensing info.

### 4. Improve Documentation

- Expand the methodology document
- Add interpretation guides
- Write up findings for specific songs
- Add statistical background for non-specialists

## Running the Pipeline

### Local (Makefile)

```bash
make help              # Show all available commands

# Analysis
make analyze           # Run on all data files (markdown output)
make analyze-json      # JSON output (for scripting)
make analyze-check     # Check z-scores against threshold
make deep-dive         # Extended analysis

# Quality
make validate          # Verify data file format
make lint              # Python linting (ruff)
make format            # Auto-format Python code
make typecheck         # Type checking (mypy)

# Other
make compile-c         # Compile C tools
make word-count        # Token counts across data
make clean             # Remove build artifacts
make setup             # Install dev dependencies
```

**Custom baselines and thresholds:**

```bash
make analyze BASELINE=0.03         # Compare against general prog (3%)
make analyze-check Z_THRESHOLD=2.5 # Lower alert threshold
```

### CI/CD (Automatic)

The GitHub Actions pipeline runs automatically on:

| Trigger | What runs |
|---------|-----------|
| Push to `master`/`main` (data/scripts/analysis changes) | Full analysis + outlier alerting |
| Pull request | Analysis + validation (no alerts) |
| Manual dispatch | Analysis with custom baseline/threshold |

**What the CI does:**

1. **Validate** — checks data files exist, are valid UTF-8, contain lyrics
2. **Analyze** — runs `analyze.py` on every data file, extracts z-scores
3. **Alert** — if any z-score > 3.0, creates/updates a GitHub Issue with label `outlier-alert`
4. **Lint** — checks Python (ruff), C (gcc -Wall), Markdown, and YAML
5. **Artifacts** — uploads analysis results for 90-day retention

**Manual workflow dispatch:** Go to Actions → "Void Cluster Analysis" → Run workflow. You can override the baseline and z-threshold.

## Code Style

- **Python:** Formatted with [ruff](https://docs.astral.sh/ruff/). Run `make format` before committing.
- **C:** Standard C11, `gcc -Wall -Wextra -Wpedantic` clean.
- **Markdown:** Standard GFM. Tables for structured data.
- **Commits:** Conventional-ish — `feat:`, `fix:`, `data:`, `analysis:`, `ci:`.

## Statistical Methodology

**Please read `analysis/methodology.md` before contributing analysis code.**

Key principles:

1. **Pre-specified hypotheses** — The void cluster was defined before any data analysis. Don't add terms after seeing results (that's p-hacking).
2. **Multiple comparison correction** — We test against 6 baselines. Bonferroni α = 0.05/6 = 0.0083.
3. **Effect sizes matter** — Report Cohen's h alongside p-values. Statistical significance ≠ practical significance.
4. **Document limitations** — Every analysis should note sample size, confounds, and assumptions.
5. **Reproducibility** — All analysis must be runnable from the scripts. No hand-calculated numbers in reports.

## Pull Request Checklist

- [ ] Data files follow the format in [§1 above](#1-add-new-song-data)
- [ ] `make validate` passes
- [ ] `make analyze` runs without errors
- [ ] `make lint` passes (or explain why not)
- [ ] New analysis code has docstrings
- [ ] Methodology changes documented in `analysis/methodology.md`
- [ ] Commit messages are descriptive

## Questions?

Open an issue or start a discussion. We're particularly interested in hearing from:

- **Linguists** — is our semantic cluster well-defined?
- **NLP researchers** — better methods for semantic field analysis?
- **Music theorists** — what's the right baseline for "expected darkness" in prog rock?
- **AI researchers** — why might transformer models over-produce void semantics?

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
