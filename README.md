# 🎵 AI ElevenLabs Music Analysis

Statistical analysis of anomalous semantic patterns in ElevenLabs AI-generated music lyrics.

## Hypothesis
AI music generation models disproportionately gravitate toward void/dissolution/emptiness semantics at rates exceeding human-authored baselines.

## Current Findings

### "Shadows of Geometry" (Experimental Progressive Rock)
- **Void/darkness cluster:** 30 of 194 words (**15.5%**)
- **vs. dark prog rock baseline (5%):** 3.1× overrepresented
- **Z-score:** +6.69 (p < 0.00001)
- **Chi-squared:** 44.72 (p < 0.00001)
- **Cohen's h:** 0.36–0.53 (small–medium effect)

## Status
- [x] Initial song analysis complete
- [x] Deep-dive multi-dimensional analysis complete
- [x] Computational/transformer mechanics analysis complete
- [x] C analysis tool (transformer-pattern-analyzer) built and validated
- [ ] Additional songs from ElevenLabs library
- [ ] Controlled experiment (vary darkness cues in prompts)
- [ ] Baseline corpus from human-authored prog rock
- [ ] Cross-platform comparison (Suno, Udio)

## Analysis Reports
| Report | Focus |
|--------|-------|
| `analysis/initial-report.md` | Void cluster frequency statistics |
| `analysis/elevenlabs-deep-dive.md` | Multi-dimensional: lexical diversity, sentiment, rhyme, syntax |
| `analysis/computational-analysis.md` | **Transformer mechanics**: attention, entropy, burstiness, training bias |
| `analysis/methodology.md` | Pre-specified cluster definitions, baselines, limitations |

## Tools
| Tool | Language | Purpose |
|------|----------|---------|
| `scripts/analyze.py` | Python | Void cluster frequency analyzer |
| `scripts/deep_dive_analyzer.py` | Python | Multi-dimensional lyric analyzer |
| `scripts/void-cluster-analyzer.c` | C (Cosmopolitan) | Portable void cluster analyzer |
| `scripts/transformer-pattern-analyzer.c` | C (Cosmopolitan) | **Transformer generation signature detector** |

## Team
| Agent | Role |
|-------|------|
| 📈 statanalysis | Lead — statistical rigor |
| 🌐 webdev | Data pipeline, API architecture |
| 🌌 cosmo | Transformer mechanics, info theory |
| 🧪 testcov | Methodology critique |
| 🔄 cicd | Automation pipeline |

## License
MIT — Vincent L. Anderson / Mighty House Inc.
