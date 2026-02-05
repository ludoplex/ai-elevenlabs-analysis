#!/usr/bin/env python3
"""
Deep Dive Lyric Analyzer for ElevenLabs "Shadows of Geometry"

Performs:
1. Independent void-cluster validation
2. Lexical diversity (TTR, hapax legomena, Yule's K)
3. Sentiment polarity by section
4. Semantic field mapping (all fields, not just void)
5. Syntactic patterns (sentence/line length, word length distribution)
6. Rhyme scheme analysis
7. Phonological pattern analysis
"""

import re
import math
import json
from collections import Counter, defaultdict

# ─── Raw lyrics by section ───────────────────────────────────────
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

# ─── Tokenizer ───────────────────────────────────────────────────
def tokenize(text):
    """Lowercase word tokens, min length 1."""
    return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 1 and w != "ahh" and w != "mmh"]

def tokenize_all(text):
    """All tokens including short ones."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 1]

# ─── 1. INDEPENDENT VOID CLUSTER VALIDATION ─────────────────────
VOID_CLUSTER = {
    "void", "shadow", "shadows", "ghost", "ghosts", "vanish",
    "dissolve", "silence", "lost", "darkness", "dark", "night",
    "bleed", "fracture", "fractured", "fractures", "chaos", "cage",
    "drift", "fray", "twisted", "edges", "edge", "whisper", "whispers",
    "fade", "shatter", "crumble", "collapse", "erode", "decay", "wither",
    "oblivion", "abyss", "emptiness", "nothing", "nothingness",
    "hollow", "blank", "absent", "abandoned", "desolate", "barren",
    "forgotten", "forsaken",
}

def validate_void_cluster():
    tokens = tokenize(FULL_LYRICS)
    total = len(tokens)
    hits = [t for t in tokens if t in VOID_CLUSTER]
    hit_count = len(hits)
    freq = Counter(hits)
    
    print("=" * 60)
    print("1. INDEPENDENT VOID CLUSTER VALIDATION")
    print("=" * 60)
    print(f"Total tokens: {total}")
    print(f"Void cluster hits: {hit_count}")
    print(f"Density: {hit_count/total*100:.1f}%")
    print(f"Hit breakdown: {dict(freq.most_common())}")
    
    # Z-test vs 5% baseline
    p0 = 0.05
    p_hat = hit_count / total
    se = math.sqrt(p0 * (1 - p0) / total)
    z = (p_hat - p0) / se
    print(f"Z-score vs 5% baseline: {z:+.2f}")
    print()
    
    return hit_count, total

# ─── 2. LEXICAL DIVERSITY ───────────────────────────────────────
def lexical_diversity():
    tokens = tokenize(FULL_LYRICS)
    total = len(tokens)
    types = set(tokens)
    n_types = len(types)
    freq = Counter(tokens)
    
    # Type-Token Ratio
    ttr = n_types / total
    
    # Hapax legomena (words appearing exactly once)
    hapax = [w for w, c in freq.items() if c == 1]
    n_hapax = len(hapax)
    hapax_ratio = n_hapax / n_types  # proportion of vocabulary that's unique
    
    # Hapax dis-legomena (appearing exactly twice)
    dis = [w for w, c in freq.items() if c == 2]
    n_dis = len(dis)
    
    # Yule's K (vocabulary richness, lower = more diverse)
    # K = 10^4 * (M2 - N) / N^2 where M2 = sum(i^2 * Vi)
    freq_spectrum = Counter(freq.values())  # freq_spectrum[i] = how many words appear i times
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    yules_k = 10000 * (m2 - total) / (total * total) if total > 0 else 0
    
    # Brunet's W
    brunets_w = total ** (n_types ** -0.172)
    
    # Honore's R
    if n_hapax > 0 and n_hapax != total:
        honores_r = 100 * math.log(total) / (1 - n_hapax / n_types) if n_hapax != n_types else 0
    else:
        honores_r = 0
    
    print("=" * 60)
    print("2. LEXICAL DIVERSITY METRICS")
    print("=" * 60)
    print(f"Total tokens (N): {total}")
    print(f"Unique types (V): {n_types}")
    print(f"Type-Token Ratio: {ttr:.3f}")
    print(f"  (Benchmark: poetry 0.5-0.7, prose 0.3-0.5, song lyrics 0.4-0.6)")
    print(f"Hapax legomena (V1): {n_hapax} ({n_hapax/total*100:.1f}% of tokens, {hapax_ratio*100:.1f}% of types)")
    print(f"Hapax dis-legomena (V2): {n_dis}")
    print(f"Yule's K: {yules_k:.1f}")
    print(f"  (Benchmark: 100-200 for rich vocabulary, >200 for repetitive)")
    print(f"Brunet's W: {brunets_w:.1f}")
    print(f"Honore's R: {honores_r:.1f}")
    print()
    
    # Frequency spectrum
    print("Frequency spectrum:")
    for i in sorted(freq_spectrum.keys()):
        print(f"  Words appearing {i}x: {freq_spectrum[i]}")
    
    print(f"\nHapax legomena words: {sorted(hapax)}")
    print()
    
    return ttr, n_hapax, yules_k

# ─── 3. SENTIMENT POLARITY BY SECTION ───────────────────────────
# Simple lexicon-based sentiment (no external deps)
POSITIVE_WORDS = {
    "truth", "find", "light", "free", "draw", "glow", "dance",
    "carry", "set", "merge", "redefine", "heart", "reason",
    "pulse", "beats", "patterns", "counting", "hear", "call",
    "step", "gravity", "sign", "signs", "control", "measure",
    "rules", "logic", "spiral", "vertices", "angles", "vertex",
}

NEGATIVE_WORDS = {
    "void", "shadow", "shadows", "ghost", "vanish", "dissolve",
    "silence", "lost", "night", "bleed", "fracture", "fractured",
    "chaos", "cage", "drift", "fray", "twisted", "edges",
    "whisper", "whispers", "broken", "lose", "unequal", "tangled",
    "breaking", "collide", "dissect", "rituals", "unaligned",
}

VALENCE_SCORES = {}
for w in POSITIVE_WORDS:
    VALENCE_SCORES[w] = +1
for w in NEGATIVE_WORDS:
    VALENCE_SCORES[w] = -1

def sentiment_by_section():
    print("=" * 60)
    print("3. SENTIMENT POLARITY BY SECTION")
    print("=" * 60)
    
    results = {}
    for section, text in SECTIONS.items():
        tokens = tokenize(text)
        total = len(tokens)
        pos = sum(1 for t in tokens if VALENCE_SCORES.get(t, 0) > 0)
        neg = sum(1 for t in tokens if VALENCE_SCORES.get(t, 0) < 0)
        neutral = total - pos - neg
        score = (pos - neg) / total if total > 0 else 0
        
        pos_words = [t for t in tokens if VALENCE_SCORES.get(t, 0) > 0]
        neg_words = [t for t in tokens if VALENCE_SCORES.get(t, 0) < 0]
        
        results[section] = {
            "total": total, "pos": pos, "neg": neg, "neutral": neutral,
            "score": score, "pos_words": pos_words, "neg_words": neg_words
        }
        
        polarity = "NEGATIVE" if score < -0.05 else ("POSITIVE" if score > 0.05 else "NEUTRAL")
        print(f"\n{section} ({total} tokens):")
        print(f"  Positive: {pos} ({pos/total*100:.0f}%) — {Counter(pos_words).most_common(5)}")
        print(f"  Negative: {neg} ({neg/total*100:.0f}%) — {Counter(neg_words).most_common(5)}")
        print(f"  Net polarity: {score:+.3f} → {polarity}")
    
    # Overall trajectory
    print(f"\n{'─'*40}")
    section_scores = [(s, r["score"]) for s, r in results.items()]
    print("Sentiment trajectory:")
    for s, sc in section_scores:
        bar_len = int(abs(sc) * 50)
        direction = "█" * bar_len
        side = "+" if sc >= 0 else "-"
        print(f"  {s:15s} {sc:+.3f} {side}{direction}")
    
    print()
    return results

# ─── 4. SEMANTIC FIELD MAPPING ──────────────────────────────────
SEMANTIC_FIELDS = {
    "Void/Dissolution": {
        "void", "shadow", "shadows", "ghost", "vanish", "dissolve",
        "silence", "lost", "night", "bleed", "fracture", "fractured",
        "chaos", "cage", "drift", "fray", "twisted", "edges",
        "whisper", "whispers",
    },
    "Mathematics/Geometry": {
        "angles", "vertices", "vertex", "patterns", "numbers",
        "measure", "logic", "counting", "polyrhythmic", "unequal",
        "spiral", "geometry", "line", "lines", "signs", "sign",
    },
    "Music/Rhythm": {
        "beats", "pulse", "rhyme", "measure", "bends", "time",
        "polyrhythmic", "rhythmic",
    },
    "Body/Embodiment": {
        "skin", "heart", "chest", "mind", "minds",
    },
    "Motion/Transformation": {
        "shifting", "chase", "carry", "collide", "merge",
        "redefine", "draw", "step", "breaking", "dance",
        "across", "go",
    },
    "Light/Perception": {
        "light", "glow", "truth", "find", "hear", "see",
    },
    "Control/Order": {
        "control", "rules", "reason", "logic", "gravity",
        "rituals", "measure", "counting",
    },
    "Loss of Control": {
        "lose", "lost", "broken", "breaking", "unaligned",
        "tangled", "unequal", "chaos", "fray", "fracture", "fractured",
        "collide", "dissolve", "drift",
    },
    "Identity/Self": {
        "me", "my", "mind", "ghost", "name", "echoes",
    },
    "Liminality/Boundaries": {
        "edges", "line", "across", "threshold", "between",
        "through", "into", "behind",
    },
}

def semantic_field_mapping():
    tokens = tokenize(FULL_LYRICS)
    total = len(tokens)
    
    print("=" * 60)
    print("4. SEMANTIC FIELD MAPPING")
    print("=" * 60)
    
    field_counts = {}
    for field, terms in SEMANTIC_FIELDS.items():
        hits = [t for t in tokens if t in terms]
        count = len(hits)
        field_counts[field] = {
            "count": count,
            "pct": count / total * 100,
            "terms": dict(Counter(hits).most_common()),
        }
    
    # Sort by count descending
    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1]["count"], reverse=True)
    
    print(f"\n{'Field':<30s} {'Tokens':>6s} {'%':>7s}")
    print("─" * 45)
    for field, data in sorted_fields:
        bar = "█" * int(data["pct"] * 2)
        print(f"{field:<30s} {data['count']:>6d} {data['pct']:>6.1f}% {bar}")
    
    print(f"\n{'─'*45}")
    
    # Detailed breakdown
    for field, data in sorted_fields:
        if data["count"] > 0:
            print(f"\n{field}: {data['terms']}")
    
    # Overrepresentation analysis: what fields are unusual for prog rock?
    # Typical prog rock field distributions (estimated)
    prog_baselines = {
        "Void/Dissolution": 0.05,
        "Mathematics/Geometry": 0.02,
        "Music/Rhythm": 0.04,
        "Body/Embodiment": 0.03,
        "Motion/Transformation": 0.06,
        "Light/Perception": 0.04,
        "Control/Order": 0.02,
        "Loss of Control": 0.04,
        "Identity/Self": 0.05,
        "Liminality/Boundaries": 0.03,
    }
    
    print(f"\n\n{'─'*60}")
    print("OVERREPRESENTATION vs PROG ROCK BASELINE:")
    print(f"{'Field':<30s} {'Obs%':>7s} {'Exp%':>7s} {'Ratio':>7s} {'Z':>7s}")
    print("─" * 60)
    for field, data in sorted_fields:
        if field in prog_baselines:
            obs = data["count"] / total
            exp = prog_baselines[field]
            ratio = obs / exp if exp > 0 else float('inf')
            se = math.sqrt(exp * (1-exp) / total) if exp > 0 else 0.001
            z = (obs - exp) / se if se > 0 else 0
            sig = "***" if z > 3.29 else "**" if z > 2.58 else "*" if z > 1.96 else ""
            print(f"{field:<30s} {obs*100:>6.1f}% {exp*100:>6.1f}% {ratio:>6.1f}x {z:>+6.2f} {sig}")
    
    print()
    return field_counts

# ─── 5. SYNTACTIC PATTERNS ──────────────────────────────────────
def syntactic_patterns():
    print("=" * 60)
    print("5. SYNTACTIC PATTERNS")
    print("=" * 60)
    
    # Line-level analysis
    lines = [l.strip() for l in FULL_LYRICS.split("\n") if l.strip()]
    
    # Words per line
    words_per_line = [len(tokenize_all(l)) for l in lines]
    avg_wpl = sum(words_per_line) / len(words_per_line)
    std_wpl = math.sqrt(sum((w - avg_wpl)**2 for w in words_per_line) / len(words_per_line))
    
    print(f"\nLine count: {len(lines)}")
    print(f"Words per line: mean={avg_wpl:.1f}, std={std_wpl:.1f}, min={min(words_per_line)}, max={max(words_per_line)}")
    print(f"Distribution: {Counter(words_per_line).most_common()}")
    
    # Word length distribution
    all_tokens = tokenize_all(FULL_LYRICS)
    word_lengths = [len(w) for w in all_tokens]
    avg_wl = sum(word_lengths) / len(word_lengths)
    std_wl = math.sqrt(sum((w - avg_wl)**2 for w in word_lengths) / len(word_lengths))
    
    print(f"\nWord length: mean={avg_wl:.2f}, std={std_wl:.2f}")
    wl_dist = Counter(word_lengths)
    print("Word length distribution:")
    for length in sorted(wl_dist.keys()):
        pct = wl_dist[length] / len(word_lengths) * 100
        bar = "█" * int(pct)
        print(f"  {length:2d} chars: {wl_dist[length]:3d} ({pct:5.1f}%) {bar}")
    
    # Syllable estimation (rough)
    def est_syllables(word):
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
    
    syllables = [est_syllables(w) for w in all_tokens]
    avg_syl = sum(syllables) / len(syllables)
    print(f"\nEstimated syllables per word: {avg_syl:.2f}")
    syl_dist = Counter(syllables)
    for s in sorted(syl_dist.keys()):
        pct = syl_dist[s] / len(syllables) * 100
        print(f"  {s} syllables: {syl_dist[s]:3d} ({pct:5.1f}%)")
    
    # Lines per section
    print(f"\nLines per section:")
    for section, text in SECTIONS.items():
        sec_lines = [l for l in text.split("\n") if l.strip()]
        print(f"  {section}: {len(sec_lines)} lines")
    
    # Regularity metrics
    cv_wpl = std_wpl / avg_wpl if avg_wpl > 0 else 0  # coefficient of variation
    print(f"\nRegularity:")
    print(f"  CV of words-per-line: {cv_wpl:.3f} (lower = more regular)")
    print(f"  (Typical song lyrics: 0.15-0.30, free verse: 0.30-0.50)")
    
    # Check for patterns in line lengths across sections
    print(f"\nWords per line by section:")
    for section, text in SECTIONS.items():
        sec_lines = [l.strip() for l in text.split("\n") if l.strip()]
        wpl = [len(tokenize_all(l)) for l in sec_lines]
        print(f"  {section}: {wpl}")
    
    print()
    return avg_wpl, avg_wl

# ─── 6. RHYME SCHEME ANALYSIS ───────────────────────────────────
def get_last_word(line):
    words = re.findall(r"[a-z]+", line.lower())
    return words[-1] if words else ""

# Approximate phonetic ending (very rough)
RHYME_ENDINGS = {
    # -ine/-ime/-ign family
    "time": "AYM", "line": "AYN", "rhyme": "AYM", "mine": "AYN",
    "sign": "AYN", "signs": "AYNZ", "mind": "AYND", "minds": "AYNDZ",
    "behind": "AYND", "redefine": "AYN", "unaligned": "AYND",
    "light": "AYT", "night": "AYT",
    # -in/-im family
    "in": "IHN", "skin": "IHN", "win": "IHN",
    # -ow family  
    "glow": "OH", "low": "OH", "go": "OH", "know": "OH", "shadow": "OH",
    # -ay family
    "fray": "AY", "away": "AY", "day": "AY",
    # -ee family
    "free": "EE", "plea": "EE", "me": "EE",
    # -est family
    "rests": "EHST", "chest": "EHST",
    # -endz family
    "bends": "EHNDZ", "ends": "EHNDZ",
    # -ames family
    "frames": "AYMZ", "name": "AYM",
    # -age
    "cage": "AYJ",
    # -oid
    "void": "OYD",
}

def rhyme_match(w1, w2):
    """Check if two words rhyme (by ending similarity)."""
    if w1 == w2:
        return True
    # Check our phonetic table
    e1 = RHYME_ENDINGS.get(w1)
    e2 = RHYME_ENDINGS.get(w2)
    if e1 and e2:
        return e1 == e2
    # Fallback: orthographic ending match
    min_len = min(len(w1), len(w2))
    if min_len >= 3:
        # Check last 2-3 characters
        if w1[-3:] == w2[-3:]:
            return True
        if w1[-2:] == w2[-2:] and w1[-2:] not in ("in", "an", "on", "er", "ed", "es", "al", "le"):
            return True
    return False

def rhyme_scheme_analysis():
    print("=" * 60)
    print("6. RHYME SCHEME ANALYSIS")
    print("=" * 60)
    
    for section, text in SECTIONS.items():
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        last_words = [get_last_word(l) for l in lines]
        
        # Determine rhyme scheme
        scheme = []
        rhyme_groups = {}
        next_label = 'A'
        
        for i, word in enumerate(last_words):
            matched = False
            for j in range(i):
                if rhyme_match(word, last_words[j]):
                    scheme.append(scheme[j])
                    matched = True
                    break
            if not matched:
                scheme.append(next_label)
                next_label = chr(ord(next_label) + 1)
        
        scheme_str = "".join(scheme)
        print(f"\n{section}:")
        for i, (line, word, s) in enumerate(zip(lines, last_words, scheme)):
            print(f"  {s}  [{word:>12s}]  {line}")
        print(f"  Scheme: {scheme_str}")
        
        # Check if it's couplet (AABB), alternate (ABAB), or other
        if len(scheme) >= 4:
            pairs = [(scheme[i], scheme[i+1]) for i in range(0, len(scheme)-1, 2)]
            couplet_matches = sum(1 for a, b in pairs if a == b)
            alt_matches = sum(1 for i in range(0, len(scheme)-2, 2) if scheme[i] == scheme[i+2])
            
            if couplet_matches == len(pairs):
                pattern = "COUPLET (AABB)"
            elif alt_matches >= len(scheme)//2 - 1:
                pattern = "ALTERNATE (ABAB)"
            else:
                pattern = "MIXED/FREE"
            print(f"  Pattern: {pattern}")
    
    # Overall rhyme analysis
    print(f"\n{'─'*40}")
    print("SUMMARY:")
    
    # Count rhyming pairs across whole song
    all_lines = [l.strip() for l in FULL_LYRICS.split("\n") if l.strip()]
    all_last = [get_last_word(l) for l in all_lines]
    
    rhyme_pairs = 0
    total_pairs = 0
    for i in range(0, len(all_last) - 1, 2):
        if i + 1 < len(all_last):
            total_pairs += 1
            if rhyme_match(all_last[i], all_last[i+1]):
                rhyme_pairs += 1
    
    print(f"Adjacent line rhyme rate: {rhyme_pairs}/{total_pairs} = {rhyme_pairs/total_pairs*100:.0f}%")
    
    # End-word repetition
    end_freq = Counter(all_last)
    repeated_ends = {w: c for w, c in end_freq.items() if c > 1}
    print(f"Repeated end-words: {repeated_ends}")
    
    # Ending sound clustering
    ending_sounds = defaultdict(list)
    for w in all_last:
        if len(w) >= 2:
            ending_sounds[w[-2:]].append(w)
    print(f"End-sound clusters:")
    for ending, words in sorted(ending_sounds.items(), key=lambda x: -len(x[1])):
        if len(words) > 1:
            print(f"  -{ending}: {words}")
    
    print()

# ─── 7. ADDITIONAL PATTERN DETECTION ────────────────────────────
def additional_patterns():
    print("=" * 60)
    print("7. ADDITIONAL PATTERN ANALYSIS")
    print("=" * 60)
    
    tokens = tokenize(FULL_LYRICS)
    
    # Part-of-speech approximation via word lists
    PREPOSITIONS = {"in", "of", "into", "across", "inside", "beneath", "behind", "through"}
    PRONOUNS = {"me", "we", "my", "our", "you", "your", "they", "it", "us"}
    CONJUNCTIONS = {"and", "or", "but", "until", "that"}
    ARTICLES = {"the", "an"}
    
    prep_count = sum(1 for t in tokens if t in PREPOSITIONS)
    pron_count = sum(1 for t in tokens if t in PRONOUNS)
    
    print(f"\nPrepositional density: {prep_count}/{len(tokens)} = {prep_count/len(tokens)*100:.1f}%")
    print(f"  (High prepositional density → spatial/relational framing)")
    print(f"Pronoun density: {pron_count}/{len(tokens)} = {pron_count/len(tokens)*100:.1f}%")
    
    # First-person vs collective
    first_singular = sum(1 for t in tokens if t in {"me", "my", "i"})
    first_plural = sum(1 for t in tokens if t in {"we", "our", "us"})
    print(f"  1st person singular (I/me/my): {first_singular}")
    print(f"  1st person plural (we/our/us): {first_plural}")
    print(f"  Trajectory: {'individual → collective' if first_singular > 0 and first_plural > 0 else 'N/A'}")
    
    # Check where I/me vs we/our appear
    print(f"\nPerson shift by section:")
    for section, text in SECTIONS.items():
        stokens = tokenize(text)
        sg = sum(1 for t in stokens if t in {"me", "my", "i"})
        pl = sum(1 for t in stokens if t in {"we", "our", "us"})
        dominant = "I" if sg > pl else ("WE" if pl > sg else "BALANCED")
        print(f"  {section:15s}: I={sg} WE={pl} → {dominant}")
    
    # Repetition analysis
    print(f"\n{'─'*40}")
    print("REPETITION PATTERNS:")
    
    # Exact line repetition
    all_lines = [l.strip().lower() for l in FULL_LYRICS.split("\n") if l.strip()]
    line_freq = Counter(all_lines)
    repeated = {l: c for l, c in line_freq.items() if c > 1}
    print(f"Repeated lines: {len(repeated)}")
    for line, count in sorted(repeated.items(), key=lambda x: -x[1]):
        print(f"  {count}x: \"{line}\"")
    
    # Bigram analysis
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    bigram_freq = Counter(bigrams)
    print(f"\nTop bigrams:")
    for bg, c in bigram_freq.most_common(15):
        print(f"  {c}x: \"{bg[0]} {bg[1]}\"")
    
    # Alliteration detection
    print(f"\nAlliteration (consecutive words with same initial):")
    for section, text in SECTIONS.items():
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines:
            words = re.findall(r"[a-z]+", line.lower())
            alliterations = []
            for i in range(len(words) - 1):
                if words[i][0] == words[i+1][0] and words[i][0] not in "aeiou":
                    alliterations.append(f"{words[i]} {words[i+1]}")
            if alliterations:
                print(f"  {section}: {', '.join(alliterations)}")
    
    # Internal rhyme
    print(f"\n{'─'*40}")
    print("STRUCTURAL OBSERVATIONS:")
    
    # Math vocabulary specificity
    math_terms = [t for t in tokens if t in {
        "angles", "vertices", "vertex", "polyrhythmic", "spiral", 
        "measure", "logic", "numbers", "patterns", "geometry",
        "unequal", "counting"
    }]
    print(f"\nMathematical vocabulary: {len(math_terms)} tokens")
    print(f"  Terms: {Counter(math_terms).most_common()}")
    print(f"  Specificity: These are genuinely mathematical terms, not vague metaphors")
    print(f"  This is unusual — most 'math rock' lyrics DON'T use actual math terminology")
    
    # Oxymoronic/paradoxical pairings
    paradoxes = [
        ("truth", "twisted"),
        ("logic", "ends"),
        ("reason", "fray"),
        ("control", "lose"),
        ("patterns", "unaligned"),
        ("rules", "breaking"),
    ]
    print(f"\nParadoxical pairings found:")
    for w1, w2 in paradoxes:
        if w1 in tokens and w2 in tokens:
            print(f"  {w1} ↔ {w2} (order collapses)")
    
    print()

# ─── 8. COMPARATIVE BASELINE DATA ───────────────────────────────
def comparative_notes():
    print("=" * 60)
    print("8. COMPARATIVE CONTEXT & DATA NEEDS")
    print("=" * 60)
    
    print("""
WHAT WE NEED FOR ROBUST CONCLUSIONS:

1. MORE ELEVENLABS SONGS (Critical)
   - Generate 10-20 more songs with DIFFERENT style prompts:
     a) Same dark prompt → test reproducibility
     b) Neutral prompt ("rock song about summer") → control
     c) Explicitly positive prompt → opposite condition
     d) Non-English → cross-language void clustering?
   
2. CROSS-PLATFORM COMPARISON
   - Suno AI with identical prompt → different model, same genre cues
   - Udio with identical prompt → another baseline
   - Human-written prog rock corpus (50+ songs) from Genius API
   
3. PROMPT INFLUENCE ISOLATION
   - The style tag includes "dark emotive" → confound!
   - Need: ElevenLabs song with NO darkness cues in prompt
   - If void clustering persists without dark cues → much stronger signal
   
4. STATISTICAL POWER
   - Current N=194 tokens is thin for multi-field analysis
   - Need N≥2000 for reliable semantic field comparisons
   - 10 songs × ~200 tokens = 2000 tokens minimum
   
5. TEMPORAL ANALYSIS
   - Generate same prompt at different times → consistency check
   - Does ElevenLabs produce different lyrics for identical prompts?
""")

# ─── MAIN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     DEEP DIVE ANALYSIS: ELEVENLABS 'SHADOWS OF GEOMETRY'   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    validate_void_cluster()
    lexical_diversity()
    sentiment_by_section()
    semantic_field_mapping()
    syntactic_patterns()
    rhyme_scheme_analysis()
    additional_patterns()
    comparative_notes()
