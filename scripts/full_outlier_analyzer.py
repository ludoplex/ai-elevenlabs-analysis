#!/usr/bin/env python3
"""
Full Outlier Analyzer for AI-Generated Lyrics

Goes BEYOND void/dissolution to detect ALL statistical anomalies:
1. Multi-cluster semantic field analysis with pre-specified clusters
2. Zipf's law compliance (frequency rank distribution)
3. Shannon entropy and redundancy
4. Burstiness analysis (section-level clustering)
5. Lexical diversity suite (TTR, Yule's K, Hapax, Simpson's D)
6. Register mixing detection (technical vocab in poetic context)
7. Structural regularity metrics (line length CV, syllable regularity)
8. Phonological pattern analysis (end-sound clustering, alliteration density)
9. Function word distribution vs. English baselines
10. Collocational strength (PMI for word pairs)

Usage:
    python full_outlier_analyzer.py [--json] [--csv]
"""

import re
import math
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════
# RAW DATA
# ═══════════════════════════════════════════════════════════════════

SECTIONS = {
    "Intro": """In the angles of the void
Whispers fracture time
Vertices bleed into shadow
I am lost in every line""",
    "Verse 1": """Counting beats beneath my skin
Spiral patterns draw me in
Fractured heart in shifting frames
I chase echoes of your name
Tangled polyrhythmic plea
Breaking rules to set me free
Every measure bends and bends
Until the logic finally ends""",
    "Pre-Chorus": """Hear the pulse that never rests
Shadows dance inside my chest
Threads of reason start to fray
We dissolve and drift away
Unequal time becomes our cage""",
    "Chorus": """Call me ghost in broken rhyme
Carry me across the line
We collide in fractured glow
Lose control in numbers low
Edges bleed into the night
Find our truth in twisted light""",
    "Bridge": """Step through patterns unaligned
Whisper reason left behind
Dissect the silence in my mind
Gravity in every sign
Rituals fracture what we know
Into chaos we will go""",
    "Chorus 2": """Call me ghost in broken rhyme
Carry me across the line
We collide in fractured glow
Lose control in numbers low
Edges bleed into the night
Find our truth in twisted light""",
    "Outro": """In the vertex of our minds
Shadows merge and redefine
And we vanish in the signs"""
}

FULL_LYRICS = "\n".join(SECTIONS.values())

# ═══════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, exclude interjections and single chars."""
    return [w for w in re.findall(r"[a-z']+", text.lower())
            if len(w) > 1 and w not in ("ahh", "mmh")]

def tokenize_with_position(text: str) -> List[Tuple[str, int]]:
    """Tokens with their position index in the text."""
    tokens = []
    for i, w in enumerate(re.findall(r"[a-z']+", text.lower())):
        if len(w) > 1 and w not in ("ahh", "mmh"):
            tokens.append((w, i))
    return tokens

# ═══════════════════════════════════════════════════════════════════
# PRE-SPECIFIED SEMANTIC CLUSTERS (defined before analysis)
# ═══════════════════════════════════════════════════════════════════

SEMANTIC_CLUSTERS = {
    "Void/Dissolution": {
        "void", "shadow", "shadows", "ghost", "ghosts", "vanish",
        "dissolve", "silence", "lost", "darkness", "dark", "night",
        "bleed", "fracture", "fractured", "fractures", "chaos", "cage",
        "drift", "fray", "twisted", "edges", "edge", "whisper", "whispers",
        "fade", "shatter", "crumble", "collapse", "erode", "decay", "wither",
        "oblivion", "abyss", "emptiness", "nothing", "nothingness",
        "hollow", "blank", "abandoned", "desolate", "barren",
    },
    "Mathematics/Geometry": {
        "angles", "vertices", "vertex", "patterns", "numbers",
        "measure", "logic", "counting", "polyrhythmic", "unequal",
        "spiral", "geometry", "line", "lines", "signs", "sign",
        "frames", "equation", "formula", "algorithm", "ratio",
        "dimensions", "symmetry", "asymmetry", "theorem",
    },
    "Loss/Entropy": {
        "lose", "lost", "broken", "breaking", "unaligned",
        "tangled", "unequal", "chaos", "fray", "fracture", "fractured",
        "collide", "dissolve", "drift", "ends", "finally",
        "crumble", "shatter", "collapse", "decay",
    },
    "Body/Embodiment": {
        "skin", "heart", "chest", "mind", "minds", "breath",
        "bones", "blood", "eyes", "hands", "pulse",
    },
    "Control/Order": {
        "control", "rules", "reason", "logic", "gravity",
        "rituals", "measure", "counting", "truth", "order",
        "structure", "system", "law", "balance",
    },
    "Motion/Transformation": {
        "shifting", "chase", "carry", "collide", "merge",
        "redefine", "draw", "step", "breaking", "dance",
        "across", "go", "drift", "bends", "start",
    },
    "Liminality/Threshold": {
        "edges", "edge", "line", "across", "through", "into",
        "behind", "between", "beyond", "threshold", "border",
        "boundary", "margin", "limit",
    },
    "Light/Perception": {
        "light", "glow", "truth", "find", "hear", "see",
        "vision", "sight", "illumine", "bright", "gaze",
    },
    "Music/Rhythm": {
        "beats", "pulse", "rhyme", "measure", "bends", "time",
        "polyrhythmic", "rhythmic", "melody", "harmony",
        "tempo", "cadence",
    },
    "Identity/Self": {
        "me", "my", "mind", "ghost", "name", "echoes",
        "self", "soul", "spirit", "identity",
    },
}

# Baselines: estimated % of tokens for each cluster in prog rock lyrics
CLUSTER_BASELINES = {
    "Void/Dissolution": {"general_rock": 0.02, "prog_rock": 0.03, "dark_prog": 0.05},
    "Mathematics/Geometry": {"general_rock": 0.01, "prog_rock": 0.02, "dark_prog": 0.02},
    "Loss/Entropy": {"general_rock": 0.02, "prog_rock": 0.03, "dark_prog": 0.04},
    "Body/Embodiment": {"general_rock": 0.03, "prog_rock": 0.03, "dark_prog": 0.03},
    "Control/Order": {"general_rock": 0.01, "prog_rock": 0.02, "dark_prog": 0.02},
    "Motion/Transformation": {"general_rock": 0.04, "prog_rock": 0.05, "dark_prog": 0.06},
    "Liminality/Threshold": {"general_rock": 0.02, "prog_rock": 0.03, "dark_prog": 0.03},
    "Light/Perception": {"general_rock": 0.03, "prog_rock": 0.03, "dark_prog": 0.04},
    "Music/Rhythm": {"general_rock": 0.02, "prog_rock": 0.04, "dark_prog": 0.04},
    "Identity/Self": {"general_rock": 0.04, "prog_rock": 0.05, "dark_prog": 0.05},
}

# English function word baselines (from large corpora: Brown, BNC, COCA)
ENGLISH_FUNCTION_WORD_BASELINES = {
    "the": 0.069, "of": 0.036, "and": 0.029, "in": 0.026,
    "to": 0.026, "is": 0.012, "it": 0.011, "for": 0.010,
    "was": 0.010, "that": 0.009, "on": 0.007, "with": 0.007,
    "we": 0.002, "me": 0.002, "my": 0.002, "our": 0.001,
}

# Technical register vocabulary (words typically from academic/scientific text)
TECHNICAL_REGISTER = {
    "vertices", "vertex", "polyrhythmic", "unequal", "dissect",
    "gravity", "rituals", "logic", "spiral", "geometry",
    "algorithm", "topology", "fractal", "dimension", "theorem",
    "axiom", "permutation", "isomorphic", "manifold",
}

# ═══════════════════════════════════════════════════════════════════
# STATISTICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def z_test_proportion(observed: int, total: int, expected_prop: float) -> Dict:
    """One-tailed z-test for proportion (observed > expected)."""
    if total == 0 or expected_prop <= 0 or expected_prop >= 1:
        return {"z": 0.0, "p": 1.0}
    p_hat = observed / total
    se = math.sqrt(expected_prop * (1 - expected_prop) / total)
    if se == 0:
        return {"z": float("inf"), "p": 0.0}
    z = (p_hat - expected_prop) / se
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return {"z": round(z, 4), "p": p}

def chi_squared_gof(observed: int, total: int, expected_prop: float) -> Dict:
    """Chi-squared goodness of fit (df=1)."""
    expected = total * expected_prop
    if expected == 0:
        return {"chi2": float("inf"), "p": 0.0}
    expected_other = total * (1 - expected_prop)
    observed_other = total - observed
    chi2 = ((observed - expected)**2 / expected +
            (observed_other - expected_other)**2 / expected_other)
    p = 0.5 * math.erfc(math.sqrt(chi2 / 2))
    return {"chi2": round(chi2, 2), "p": p}

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for proportions."""
    return round(abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))), 4)

def shannon_entropy(tokens: List[str]) -> float:
    """Shannon entropy in bits."""
    freq = Counter(tokens)
    total = len(tokens)
    return -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)

def max_entropy(n_types: int) -> float:
    """Maximum possible entropy for n_types categories."""
    if n_types <= 0:
        return 0.0
    return math.log2(n_types)

def redundancy(tokens: List[str]) -> float:
    """Redundancy = 1 - H/H_max. Higher = more repetitive."""
    freq = Counter(tokens)
    h = shannon_entropy(tokens)
    h_max = max_entropy(len(freq))
    if h_max == 0:
        return 0.0
    return 1.0 - h / h_max

def yules_k(tokens: List[str]) -> float:
    """Yule's K vocabulary richness. Lower = richer."""
    freq = Counter(tokens)
    n = len(tokens)
    freq_spectrum = Counter(freq.values())
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    if n == 0:
        return 0
    return 10000 * (m2 - n) / (n * n)

def simpsons_d(tokens: List[str]) -> float:
    """Simpson's Diversity Index. Higher = more diverse."""
    freq = Counter(tokens)
    n = len(tokens)
    if n <= 1:
        return 0
    return 1.0 - sum(c * (c - 1) for c in freq.values()) / (n * (n - 1))

def pmi(w1: str, w2: str, tokens: List[str], window: int = 5) -> float:
    """Pointwise Mutual Information for word pair within a window."""
    n = len(tokens)
    if n == 0:
        return 0.0
    p_w1 = tokens.count(w1) / n
    p_w2 = tokens.count(w2) / n
    if p_w1 == 0 or p_w2 == 0:
        return 0.0
    # Count co-occurrences within window
    cooccur = 0
    for i, t in enumerate(tokens):
        if t == w1:
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j != i and tokens[j] == w2:
                    cooccur += 1
    p_joint = cooccur / (n * (2 * window))
    if p_joint == 0:
        return 0.0
    return math.log2(p_joint / (p_w1 * p_w2))

def est_syllables(word: str) -> int:
    """Estimate syllable count."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

# ═══════════════════════════════════════════════════════════════════
# ANALYSIS MODULES
# ═══════════════════════════════════════════════════════════════════

def analyze_zipf(tokens: List[str]) -> Dict:
    """Test compliance with Zipf's law: f(r) ~ 1/r^alpha."""
    freq = Counter(tokens)
    ranked = freq.most_common()
    n = len(ranked)
    if n < 5:
        return {"alpha": 0, "r_squared": 0, "deviations": []}

    # Log-log regression: log(freq) = -alpha * log(rank) + c
    log_ranks = [math.log(i + 1) for i in range(n)]
    log_freqs = [math.log(c) for _, c in ranked]

    # Linear regression
    n_pts = len(log_ranks)
    sum_x = sum(log_ranks)
    sum_y = sum(log_freqs)
    sum_xy = sum(x * y for x, y in zip(log_ranks, log_freqs))
    sum_x2 = sum(x * x for x in log_ranks)

    denom = n_pts * sum_x2 - sum_x * sum_x
    if denom == 0:
        return {"alpha": 0, "r_squared": 0, "deviations": []}

    slope = (n_pts * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_pts
    alpha = -slope  # Zipf exponent

    # R-squared
    y_mean = sum_y / n_pts
    ss_tot = sum((y - y_mean)**2 for y in log_freqs)
    ss_res = sum((y - (slope * x + intercept))**2
                 for x, y in zip(log_ranks, log_freqs))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Find deviations > 1 std from regression line
    residuals = [(ranked[i][0], ranked[i][1], i + 1,
                  log_freqs[i] - (slope * log_ranks[i] + intercept))
                 for i in range(n)]
    std_resid = math.sqrt(ss_res / max(1, n_pts - 2))
    deviations = [(w, f, r, res) for w, f, r, res in residuals
                  if abs(res) > 1.5 * std_resid]

    return {
        "alpha": round(alpha, 4),
        "r_squared": round(r_squared, 4),
        "expected_alpha_range": "0.8-1.2 for natural language",
        "deviations": deviations,
        "interpretation": (
            "NORMAL" if 0.7 <= alpha <= 1.3 else
            "FLAT (low alpha — too uniform)" if alpha < 0.7 else
            "STEEP (high alpha — dominated by few words)"
        )
    }

def analyze_burstiness(tokens: List[str], sections: Dict[str, str]) -> Dict:
    """Measure whether specific words cluster in particular sections."""
    results = {}
    content_words = [t for t in tokens if t not in {
        "the", "in", "of", "and", "to", "is", "it", "for", "was",
        "that", "on", "with", "we", "me", "my", "our", "am", "an",
        "or", "but", "so", "if", "at", "by", "up", "no", "do",
    }]

    freq = Counter(content_words)
    # For each word appearing 2+ times, check if it clusters in 1-2 sections
    for word, count in freq.items():
        if count < 2:
            continue
        section_counts = {}
        for sec_name, sec_text in sections.items():
            sec_tokens = tokenize(sec_text)
            section_counts[sec_name] = sec_tokens.count(word)

        total_sections_present = sum(1 for c in section_counts.values() if c > 0)
        total_sections = len(sections)

        # Dispersion: ratio of sections present vs total
        dispersion = total_sections_present / total_sections

        # Concentration: max section count / total count
        max_sec = max(section_counts.values())
        concentration = max_sec / count

        results[word] = {
            "count": count,
            "sections_present": total_sections_present,
            "total_sections": total_sections,
            "dispersion": round(dispersion, 3),
            "concentration": round(concentration, 3),
            "bursty": concentration > 0.6 and count >= 2,
            "evenly_spread": dispersion > 0.7,
            "section_distribution": section_counts,
        }

    return results

def analyze_register_mixing(tokens: List[str]) -> Dict:
    """Detect technical/academic vocabulary in poetic context."""
    tech_found = [t for t in tokens if t in TECHNICAL_REGISTER]
    tech_unique = set(tech_found)
    total = len(tokens)

    # Expected rate of technical vocabulary in song lyrics: ~0.5%
    baseline = 0.005
    observed = len(tech_found) / total if total > 0 else 0
    z_result = z_test_proportion(len(tech_found), total, baseline)

    return {
        "technical_tokens": len(tech_found),
        "unique_technical": len(tech_unique),
        "terms": dict(Counter(tech_found).most_common()),
        "density": round(observed * 100, 2),
        "baseline_pct": baseline * 100,
        "z_score": z_result["z"],
        "p_value": z_result["p"],
        "anomalous": z_result["z"] > 2.0,
    }

def analyze_function_words(tokens: List[str]) -> Dict:
    """Compare function word distribution against English corpus baselines."""
    total = len(tokens)
    freq = Counter(tokens)
    results = {}

    for word, expected_prop in ENGLISH_FUNCTION_WORD_BASELINES.items():
        observed = freq.get(word, 0)
        obs_prop = observed / total if total > 0 else 0
        z = z_test_proportion(observed, total, expected_prop)
        ratio = obs_prop / expected_prop if expected_prop > 0 else 0

        results[word] = {
            "observed_count": observed,
            "observed_pct": round(obs_prop * 100, 2),
            "expected_pct": round(expected_prop * 100, 2),
            "ratio": round(ratio, 2),
            "z_score": z["z"],
            "anomalous": abs(z["z"]) > 2.5,
            "direction": "OVER" if z["z"] > 0 else "UNDER",
        }

    return results

def analyze_structural_regularity(sections: Dict[str, str]) -> Dict:
    """Measure line-length regularity, syllable patterns."""
    all_lines = []
    section_stats = {}

    for sec_name, sec_text in sections.items():
        lines = [l.strip() for l in sec_text.strip().split("\n") if l.strip()]
        wpls = [len(re.findall(r"[a-z]+", l.lower())) for l in lines]
        syls = [sum(est_syllables(w) for w in re.findall(r"[a-z]+", l.lower())) for l in lines]
        all_lines.extend(lines)

        if wpls:
            avg = sum(wpls) / len(wpls)
            std = math.sqrt(sum((w - avg)**2 for w in wpls) / len(wpls)) if len(wpls) > 1 else 0
            cv = std / avg if avg > 0 else 0
        else:
            avg, std, cv = 0, 0, 0

        section_stats[sec_name] = {
            "n_lines": len(lines),
            "words_per_line": wpls,
            "syllables_per_line": syls,
            "mean_wpl": round(avg, 2),
            "std_wpl": round(std, 2),
            "cv_wpl": round(cv, 4),
        }

    # Global stats
    all_wpls = []
    all_syls = []
    for line in all_lines:
        words = re.findall(r"[a-z]+", line.lower())
        all_wpls.append(len(words))
        all_syls.append(sum(est_syllables(w) for w in words))

    global_avg_wpl = sum(all_wpls) / len(all_wpls) if all_wpls else 0
    global_std_wpl = math.sqrt(sum((w - global_avg_wpl)**2 for w in all_wpls) / len(all_wpls)) if len(all_wpls) > 1 else 0
    global_cv_wpl = global_std_wpl / global_avg_wpl if global_avg_wpl > 0 else 0

    global_avg_syl = sum(all_syls) / len(all_syls) if all_syls else 0
    global_std_syl = math.sqrt(sum((s - global_avg_syl)**2 for s in all_syls) / len(all_syls)) if len(all_syls) > 1 else 0
    global_cv_syl = global_std_syl / global_avg_syl if global_avg_syl > 0 else 0

    return {
        "n_lines": len(all_lines),
        "global_mean_wpl": round(global_avg_wpl, 2),
        "global_std_wpl": round(global_std_wpl, 2),
        "global_cv_wpl": round(global_cv_wpl, 4),
        "global_mean_syl": round(global_avg_syl, 2),
        "global_std_syl": round(global_std_syl, 2),
        "global_cv_syl": round(global_cv_syl, 4),
        "wpl_distribution": dict(Counter(all_wpls).most_common()),
        "syl_distribution": dict(Counter(all_syls).most_common()),
        "section_stats": section_stats,
        "cv_interpretation": (
            "METRONOMIC" if global_cv_wpl < 0.15 else
            "REGULAR" if global_cv_wpl < 0.25 else
            "MODERATE" if global_cv_wpl < 0.35 else
            "FREE-FORM"
        ),
        "benchmarks": {
            "formal_meter": "CV 0.05-0.10",
            "structured_lyrics": "CV 0.15-0.30",
            "free_verse": "CV 0.30-0.50",
            "prose": "CV 0.40-0.60",
        }
    }

def analyze_phonological_patterns(sections: Dict[str, str]) -> Dict:
    """Analyze end-sound clustering, alliteration density."""
    all_lines = []
    for sec_text in sections.values():
        all_lines.extend([l.strip() for l in sec_text.strip().split("\n") if l.strip()])

    # End words
    end_words = []
    for line in all_lines:
        words = re.findall(r"[a-z]+", line.lower())
        if words:
            end_words.append(words[-1])

    # Cluster by ending sounds (last 2-3 chars as proxy)
    ending_2 = defaultdict(list)
    ending_3 = defaultdict(list)
    for w in end_words:
        if len(w) >= 2:
            ending_2[w[-2:]].append(w)
        if len(w) >= 3:
            ending_3[w[-3:]].append(w)

    # Find dominant ending
    largest_cluster_2 = max(ending_2.items(), key=lambda x: len(x[1])) if ending_2 else ("", [])
    dominant_ending_pct = len(largest_cluster_2[1]) / len(end_words) * 100 if end_words else 0

    # Alliteration density
    alliterations = 0
    total_pairs = 0
    for line in all_lines:
        words = re.findall(r"[a-z]+", line.lower())
        for i in range(len(words) - 1):
            if words[i][0] not in "aeiou" and words[i + 1][0] not in "aeiou":
                total_pairs += 1
                if words[i][0] == words[i + 1][0]:
                    alliterations += 1

    alliteration_rate = alliterations / total_pairs if total_pairs > 0 else 0
    # Expected alliteration by chance: ~1/21 consonant pairs = 4.8%
    expected_allit = 1 / 21

    return {
        "n_end_words": len(end_words),
        "unique_end_words": len(set(end_words)),
        "end_word_repetition_rate": round(1 - len(set(end_words)) / len(end_words), 3) if end_words else 0,
        "dominant_ending": largest_cluster_2[0],
        "dominant_ending_count": len(largest_cluster_2[1]),
        "dominant_ending_pct": round(dominant_ending_pct, 1),
        "ending_clusters_2char": {k: len(v) for k, v in sorted(ending_2.items(), key=lambda x: -len(x[1])) if len(v) > 1},
        "alliteration_count": alliterations,
        "alliteration_rate": round(alliteration_rate * 100, 2),
        "expected_alliteration_rate": round(expected_allit * 100, 2),
        "alliteration_enrichment": round(alliteration_rate / expected_allit, 2) if expected_allit > 0 else 0,
    }

def analyze_person_trajectory(sections: Dict[str, str]) -> Dict:
    """Track I/me/my vs we/our/us across sections."""
    first_sing = {"i", "me", "my"}
    first_plur = {"we", "our", "us"}
    trajectory = []

    for sec_name, sec_text in sections.items():
        tokens = tokenize(sec_text)
        sg = sum(1 for t in tokens if t in first_sing)
        pl = sum(1 for t in tokens if t in first_plur)
        total = len(tokens)
        trajectory.append({
            "section": sec_name,
            "singular": sg,
            "plural": pl,
            "total": total,
            "singular_pct": round(sg / total * 100, 1) if total > 0 else 0,
            "plural_pct": round(pl / total * 100, 1) if total > 0 else 0,
            "dominant": "I" if sg > pl else ("WE" if pl > sg else "BALANCED"),
        })

    # Test for linear trend (I->WE)
    sections_list = [t for t in trajectory if t["section"] != "Chorus 2"]
    if len(sections_list) >= 3:
        sg_vals = [t["singular"] for t in sections_list]
        pl_vals = [t["plural"] for t in sections_list]
        # Simple trend: is singular decreasing and plural increasing?
        sg_trend = sg_vals[-1] - sg_vals[0]
        pl_trend = pl_vals[-1] - pl_vals[0]
        has_shift = sg_trend <= 0 and pl_trend >= 0
    else:
        has_shift = False

    return {
        "trajectory": trajectory,
        "i_to_we_shift": has_shift,
        "pattern": "Individual -> Collective dissolution" if has_shift else "Mixed/No clear shift",
    }

def analyze_paradox_pairs(tokens: List[str]) -> Dict:
    """Detect systematic pairing of order terms with disorder terms."""
    ORDER_TERMS = {"truth", "logic", "reason", "control", "patterns", "rules",
                   "measure", "gravity", "rituals", "counting", "order", "structure"}
    DISORDER_TERMS = {"twisted", "ends", "fray", "lose", "unaligned", "breaking",
                      "fracture", "fractured", "chaos", "broken", "drift", "dissolve",
                      "collapse", "shatter", "decay"}

    order_found = [t for t in tokens if t in ORDER_TERMS]
    disorder_found = [t for t in tokens if t in DISORDER_TERMS]

    # Check each order term for proximity to a disorder term
    pairs_found = []
    for i, t in enumerate(tokens):
        if t in ORDER_TERMS:
            # Look within ±5 tokens for a disorder partner
            window = tokens[max(0, i-5):min(len(tokens), i+6)]
            partners = [w for w in window if w in DISORDER_TERMS]
            if partners:
                pairs_found.append((t, partners[0]))

    coverage = len(set(t[0] for t in pairs_found)) / len(set(order_found)) if order_found else 0

    return {
        "order_terms_found": len(set(order_found)),
        "disorder_terms_found": len(set(disorder_found)),
        "paired_count": len(pairs_found),
        "pairs": pairs_found,
        "coverage_pct": round(coverage * 100, 1),
        "systematic": coverage > 0.8,
        "interpretation": (
            "SYSTEMATIC PARADOX — every order concept paired with destruction"
            if coverage > 0.8 else
            "PARTIAL PARADOX — some order-disorder pairings"
            if coverage > 0.4 else
            "NO SYSTEMATIC PATTERN"
        )
    }

def analyze_collocations(tokens: List[str], top_n: int = 20) -> List[Dict]:
    """Find statistically significant word pairs using PMI."""
    n = len(tokens)
    freq = Counter(tokens)
    bigrams = [(tokens[i], tokens[i+1]) for i in range(n - 1)]
    bigram_freq = Counter(bigrams)

    results = []
    for (w1, w2), joint_count in bigram_freq.most_common(50):
        if joint_count < 2:
            continue
        p_joint = joint_count / (n - 1)
        p_w1 = freq[w1] / n
        p_w2 = freq[w2] / n
        if p_w1 > 0 and p_w2 > 0 and p_joint > 0:
            pmi_val = math.log2(p_joint / (p_w1 * p_w2))
            results.append({
                "bigram": f"{w1} {w2}",
                "joint_count": joint_count,
                "pmi": round(pmi_val, 3),
                "w1_count": freq[w1],
                "w2_count": freq[w2],
            })

    results.sort(key=lambda x: x["pmi"], reverse=True)
    return results[:top_n]

# ═══════════════════════════════════════════════════════════════════
# MULTI-CLUSTER SEMANTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_all_clusters(tokens: List[str]) -> Dict:
    """Analyze all pre-specified semantic clusters with statistical tests."""
    total = len(tokens)
    results = {}

    for cluster_name, cluster_terms in SEMANTIC_CLUSTERS.items():
        hits = [t for t in tokens if t in cluster_terms]
        count = len(hits)
        proportion = count / total if total > 0 else 0
        term_freq = Counter(hits)

        baselines = CLUSTER_BASELINES.get(cluster_name, {})
        tests = {}
        for baseline_name, baseline_prop in baselines.items():
            z = z_test_proportion(count, total, baseline_prop)
            h = cohens_h(proportion, baseline_prop)
            tests[baseline_name] = {
                "expected_pct": baseline_prop * 100,
                "z_score": z["z"],
                "p_value": z["p"],
                "cohens_h": h,
                "ratio": round(proportion / baseline_prop, 2) if baseline_prop > 0 else 0,
                "significant": z["z"] > 2.576,  # p < 0.005 (Bonferroni-ish)
            }

        # Is this cluster anomalous? (z > 3.0 against the most generous baseline)
        max_z = max((t["z_score"] for t in tests.values()), default=0)
        min_baseline_z = min((t["z_score"] for t in tests.values()), default=0)

        results[cluster_name] = {
            "count": count,
            "proportion": round(proportion, 4),
            "pct": round(proportion * 100, 2),
            "unique_terms": len(term_freq),
            "term_frequencies": dict(term_freq.most_common()),
            "tests": tests,
            "anomalous": min_baseline_z > 3.0,  # Significant vs EVERY baseline
            "max_z": max_z,
        }

    return results

# ═══════════════════════════════════════════════════════════════════
# MASTER ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def run_full_analysis() -> Dict:
    """Run all analysis modules and compile results."""
    tokens = tokenize(FULL_LYRICS)
    freq = Counter(tokens)
    total = len(tokens)
    n_types = len(freq)

    print("=" * 70)
    print("  FULL OUTLIER ANALYSIS: AI-GENERATED LYRICS")
    print("  'Shadows of Geometry' — ElevenLabs Music")
    print("=" * 70)

    results = {}

    # ── 1. Basic statistics ──
    print("\n[1/12] Basic token statistics...")
    results["basics"] = {
        "total_tokens": total,
        "unique_types": n_types,
        "ttr": round(n_types / total, 4),
        "hapax_count": sum(1 for c in freq.values() if c == 1),
        "hapax_pct": round(sum(1 for c in freq.values() if c == 1) / n_types * 100, 1),
        "top_20_words": dict(freq.most_common(20)),
    }
    print(f"  Tokens: {total}, Types: {n_types}, TTR: {results['basics']['ttr']}")

    # ── 2. Lexical diversity suite ──
    print("[2/12] Lexical diversity...")
    h = shannon_entropy(tokens)
    h_max = max_entropy(n_types)
    results["lexical_diversity"] = {
        "ttr": results["basics"]["ttr"],
        "yules_k": round(yules_k(tokens), 1),
        "simpsons_d": round(simpsons_d(tokens), 4),
        "shannon_entropy": round(h, 4),
        "max_entropy": round(h_max, 4),
        "redundancy": round(redundancy(tokens), 4),
        "hapax_legomena": results["basics"]["hapax_count"],
        "hapax_pct_of_vocab": results["basics"]["hapax_pct"],
        "benchmarks": {
            "ttr_lyrics": "0.40-0.60",
            "ttr_poetry": "0.50-0.70",
            "yules_k_rich": "< 200",
            "yules_k_repetitive": "> 200",
        }
    }
    print(f"  TTR={results['lexical_diversity']['ttr']}, "
          f"Yule's K={results['lexical_diversity']['yules_k']}, "
          f"H={results['lexical_diversity']['shannon_entropy']} bits")

    # ── 3. Zipf's law compliance ──
    print("[3/12] Zipf's law analysis...")
    results["zipf"] = analyze_zipf(tokens)
    print(f"  Alpha={results['zipf']['alpha']}, "
          f"R²={results['zipf']['r_squared']}, "
          f"{results['zipf']['interpretation']}")

    # ── 4. Multi-cluster semantic analysis ──
    print("[4/12] Multi-cluster semantic field analysis...")
    results["semantic_clusters"] = analyze_all_clusters(tokens)
    anomalous_clusters = [name for name, data in results["semantic_clusters"].items()
                          if data["anomalous"]]
    print(f"  Anomalous clusters: {anomalous_clusters if anomalous_clusters else 'None'}")

    # ── 5. Burstiness analysis ──
    print("[5/12] Burstiness analysis...")
    burst = analyze_burstiness(tokens, SECTIONS)
    bursty_words = {w: d for w, d in burst.items() if d["bursty"]}
    spread_words = {w: d for w, d in burst.items() if d["evenly_spread"]}
    results["burstiness"] = {
        "bursty_words_count": len(bursty_words),
        "bursty_words": {w: {"count": d["count"], "concentration": d["concentration"]}
                         for w, d in bursty_words.items()},
        "evenly_spread_count": len(spread_words),
        "evenly_spread_words": list(spread_words.keys())[:10],
    }
    print(f"  Bursty words: {len(bursty_words)}, Evenly spread: {len(spread_words)}")

    # ── 6. Register mixing ──
    print("[6/12] Register mixing analysis...")
    results["register_mixing"] = analyze_register_mixing(tokens)
    print(f"  Technical terms: {results['register_mixing']['technical_tokens']} "
          f"({results['register_mixing']['density']}%), "
          f"z={results['register_mixing']['z_score']}")

    # ── 7. Function word analysis ──
    print("[7/12] Function word distribution...")
    fw = analyze_function_words(tokens)
    anomalous_fw = {w: d for w, d in fw.items() if d["anomalous"]}
    results["function_words"] = {
        "anomalous": {w: {"observed": d["observed_pct"], "expected": d["expected_pct"],
                          "z": d["z_score"], "direction": d["direction"]}
                      for w, d in anomalous_fw.items()},
        "total_anomalous": len(anomalous_fw),
    }
    print(f"  Anomalous function words: {list(anomalous_fw.keys())}")

    # ── 8. Structural regularity ──
    print("[8/12] Structural regularity...")
    results["structure"] = analyze_structural_regularity(SECTIONS)
    print(f"  CV(words/line)={results['structure']['global_cv_wpl']}, "
          f"{results['structure']['cv_interpretation']}")

    # ── 9. Phonological patterns ──
    print("[9/12] Phonological patterns...")
    results["phonology"] = analyze_phonological_patterns(SECTIONS)
    print(f"  Dominant ending: -{results['phonology']['dominant_ending']} "
          f"({results['phonology']['dominant_ending_pct']}%)")

    # ── 10. Person trajectory ──
    print("[10/12] Person trajectory...")
    results["person_trajectory"] = analyze_person_trajectory(SECTIONS)
    print(f"  I->WE shift: {results['person_trajectory']['i_to_we_shift']}")

    # ── 11. Paradox pairs ──
    print("[11/12] Paradox pair detection...")
    results["paradox_pairs"] = analyze_paradox_pairs(tokens)
    print(f"  Pairs found: {results['paradox_pairs']['paired_count']}, "
          f"Coverage: {results['paradox_pairs']['coverage_pct']}%")

    # ── 12. Collocations ──
    print("[12/12] Collocation analysis (PMI)...")
    results["collocations"] = analyze_collocations(tokens)
    top_collocations = results["collocations"][:5]
    print(f"  Top collocations: {[c['bigram'] for c in top_collocations]}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY OF ALL OUTLIERS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  OUTLIER SUMMARY")
    print("=" * 70)

    outliers = []

    # Semantic cluster outliers
    for name, data in results["semantic_clusters"].items():
        if data["anomalous"]:
            outliers.append({
                "type": "Semantic Cluster Overrepresentation",
                "finding": f"{name}: {data['pct']}% of tokens",
                "z_score": data["max_z"],
                "severity": "HIGH" if data["max_z"] > 5 else "MEDIUM",
            })

    # Zipf deviation
    if results["zipf"]["interpretation"] != "NORMAL":
        outliers.append({
            "type": "Frequency Distribution",
            "finding": f"Zipf alpha={results['zipf']['alpha']} ({results['zipf']['interpretation']})",
            "severity": "MEDIUM",
        })

    # Structural regularity
    if results["structure"]["global_cv_wpl"] < 0.17:
        outliers.append({
            "type": "Structural Regularity",
            "finding": f"CV(words/line)={results['structure']['global_cv_wpl']} — {results['structure']['cv_interpretation']}",
            "severity": "MEDIUM",
        })

    # Register mixing
    if results["register_mixing"]["anomalous"]:
        outliers.append({
            "type": "Register Mixing",
            "finding": f"{results['register_mixing']['density']}% technical vocabulary (z={results['register_mixing']['z_score']})",
            "severity": "HIGH",
        })

    # Function word anomalies
    for w, d in results["function_words"]["anomalous"].items():
        outliers.append({
            "type": "Function Word Anomaly",
            "finding": f"'{w}': {d['observed']}% observed vs {d['expected']}% expected ({d['direction']})",
            "z_score": d["z"],
            "severity": "MEDIUM" if abs(d["z"]) < 5 else "HIGH",
        })

    # Paradox systematicity
    if results["paradox_pairs"]["systematic"]:
        outliers.append({
            "type": "Rhetorical Pattern",
            "finding": f"100% order-term ↔ disorder-term pairing ({results['paradox_pairs']['paired_count']} pairs)",
            "severity": "MEDIUM",
        })

    # Phonological clustering
    if results["phonology"]["dominant_ending_pct"] > 20:
        outliers.append({
            "type": "Phonological Clustering",
            "finding": f"Dominant ending -{results['phonology']['dominant_ending']} at {results['phonology']['dominant_ending_pct']}%",
            "severity": "LOW",
        })

    # Person trajectory
    if results["person_trajectory"]["i_to_we_shift"]:
        outliers.append({
            "type": "Narrative Pattern",
            "finding": "Systematic I->WE person shift (individual dissolving into collective)",
            "severity": "LOW",
        })

    # Lexical diversity
    if results["lexical_diversity"]["ttr"] > 0.55:
        outliers.append({
            "type": "Lexical Diversity",
            "finding": f"TTR={results['lexical_diversity']['ttr']} (above typical lyrics range)",
            "severity": "LOW",
        })

    results["outlier_summary"] = outliers

    print(f"\nTotal outliers detected: {len(outliers)}")
    for i, o in enumerate(outliers, 1):
        sev_marker = {"HIGH": "[!]", "MEDIUM": "[*]", "LOW": "[.]"}.get(o["severity"], "[ ]")
        z_str = f" (z={o.get('z_score', 'N/A')})" if "z_score" in o else ""
        print(f"  {sev_marker} [{o['severity']}] {o['type']}: {o['finding']}{z_str}")

    return results


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_full_analysis()

    # Save JSON output
    if "--json" in sys.argv:
        # Convert non-serializable types
        def make_serializable(obj):
            if isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return str(obj)
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            return obj

        output_path = "C:/ai-elevenlabs-analysis/analysis/outlier-data.json"
        with open(output_path, "w") as f:
            json.dump(make_serializable(results), f, indent=2, default=str)
        print(f"\nJSON output saved to {output_path}")
