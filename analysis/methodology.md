# Methodology: Full Outlier Analysis

## Scope Expansion

This methodology extends beyond the original void-cluster-only approach to cover **all detectable statistical anomalies** in AI-generated lyrics. The twelve analytical dimensions are defined below.

---

## 1. Semantic Cluster Pre-specification

### Cluster Definitions

Ten semantic clusters were defined **before running the expanded analysis**. Each cluster's boundary was drawn using:
- WordNet synset distances (terms within 2 hops of the seed concept)
- GloVe 300d cosine similarity > 0.3 to the seed word
- Human judgment for edge cases (acknowledged as a limitation)

### Clusters

1. **Void/Dissolution** (seed: "void") -- absence, darkness, dissolution, decay
2. **Mathematics/Geometry** (seed: "geometry") -- formal math/science terms
3. **Loss/Entropy** (seed: "loss") -- breakdown, collapse, disintegration
4. **Body/Embodiment** (seed: "body") -- physical/somatic references
5. **Control/Order** (seed: "control") -- authority, structure, rules
6. **Motion/Transformation** (seed: "motion") -- movement, change
7. **Liminality/Threshold** (seed: "boundary") -- edges, crossings, transitions
8. **Light/Perception** (seed: "light") -- visual, perceptual terms
9. **Music/Rhythm** (seed: "rhythm") -- musical terminology
10. **Identity/Self** (seed: "self") -- personal pronouns, identity terms

### Overlap Policy

Tokens may belong to multiple clusters (e.g., "fracture" belongs to both Void and Loss). Cluster percentages may sum to > 100%. This is explicitly noted in all reports.

---

## 2. Baseline Selection

### Semantic Cluster Baselines

Three tier baselines estimated from genre literature:

| Baseline | Source | Application |
|----------|--------|-------------|
| General rock | Cross-genre lyric corpora (Fell 2014, Nichols et al. 2009) | Conservative comparison |
| Prog rock | Prog-specific subcorpus (Yes, Genesis, Rush, Dream Theater) | Genre-matched |
| Dark prog rock | Tool, Porcupine Tree, dark-era Pink Floyd | Most generous |

**Limitation:** These are estimates, not measured from a controlled reference corpus. Building an empirical baseline corpus is a priority for future work.

### Other Baselines

| Analysis | Baseline Source |
|----------|----------------|
| Zipf's law | Natural language universal: alpha = 0.8-1.2 |
| Function words | Brown Corpus, BNC, COCA frequency tables |
| Technical vocabulary | Song lyrics corpora: ~0.5% technical terms |
| Structural regularity | Published CV ranges for metered verse vs. free verse |

---

## 3. Statistical Tests

### Z-Test for Proportions (one-tailed)
- H0: p_observed <= p_baseline
- H1: p_observed > p_baseline
- Used for all cluster and function word comparisons

### Chi-Squared Goodness of Fit (df=1)
- Confirmatory test alongside z-test
- Same hypotheses

### Cohen's h Effect Size
- Measures practical significance independent of sample size
- Small: 0.2, Medium: 0.5, Large: 0.8

### Zipf's Law Regression
- Log-log linear regression of frequency vs. rank
- Alpha (negative slope) and R-squared reported
- Deviations identified as residuals > 1.5 SD from regression line

### Shannon Entropy
- H = -sum(p_i * log2(p_i)) for all word types
- Redundancy = 1 - H/H_max
- H_max = log2(V) where V = number of unique types

### Multiple Comparison Correction
- Bonferroni correction: 10 clusters x 3 baselines = 30 tests
- Corrected alpha = 0.05/30 = 0.00167
- Equivalent z threshold: 2.95
- All findings marked "anomalous" survive this correction

---

## 4. Known Limitations

1. **Small N** -- Single song (192 tokens). Adequate for large effects (z > 5) but insufficient for subtle anomalies.
2. **Prompt confound** -- Style tags ("dark emotive," "mathematical patterns") may drive semantic results. Structural anomalies (Zipf, syllable regularity) are prompt-independent.
3. **LLM repetition penalty** -- Transformer decoding with frequency penalties artificially flattens word distributions. This is a mechanical explanation, not a dismissal.
4. **Baseline approximation** -- Not from a controlled reference corpus.
5. **Token independence** -- Z-tests assume independent token draws. Lyrics violate this. Permutation tests recommended for future work.
6. **Semantic boundary subjectivity** -- Cluster boundaries involve judgment. Embedding-based definitions recommended.
7. **Syllable estimation** -- Rule-based heuristic, not phonetic transcription.
8. **Function word baselines** -- General English, not lyric-specific. Lyrics naturally have more pronouns.
9. **Single model, single generation** -- All findings may be specific to this ElevenLabs version/seed.

---

## 5. Robustness Checks

### Applied
- Exclude repeated Chorus 2: void density 14.6% (vs 15.6%), all conclusions hold
- Bonferroni correction for 30 comparisons: all 5 anomalous clusters survive
- Three-tier baselines: effects significant against all tiers

### Recommended for Future Work
- Permutation test (10,000 reshuffles from reference corpus)
- Embedding-based cluster validation (cosine similarity > 0.4)
- Independent rater agreement (Cohen's kappa)
- Cross-model replication (Suno, Udio, ChatGPT-generated lyrics)
- Empirical baseline construction (50-100 human songs, same method)
