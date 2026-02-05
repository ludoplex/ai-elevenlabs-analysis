/*
 * transformer-pattern-analyzer.c
 * Detects distributional signatures of transformer-generated text
 *
 * Build: cosmocc -O2 -o transformer-pattern-analyzer.com transformer-pattern-analyzer.c -lm
 * Usage: transformer-pattern-analyzer [options] [file ...]
 *        echo "text" | transformer-pattern-analyzer
 *
 * Analyzes text for computational signatures of autoregressive transformer
 * generation: entropy, burstiness, self-reinforcement, compression ratio,
 * and repetition penalty artifacts.
 *
 * Designed as an Actually Portable Executable (APE) via Cosmopolitan libc.
 *
 * Part of: github.com/ludoplex/ai-elevenlabs-analysis
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Configuration ───────────────────────────────────────────────── */

#define MAX_WORD_LEN    64
#define MAX_TOKENS      100000
#define MAX_TYPES       8192
#define HASH_SIZE       16384   /* must be power of 2 */
#define MAX_CLUSTER     512
#define MAX_POSITIONS   100000
#define BIGRAM_HASH     65536   /* must be power of 2 */

/* ── Default void/dissolution cluster ────────────────────────────── */

static const char *DEFAULT_CLUSTER[] = {
    "void", "abyss", "nothing", "nothingness", "emptiness",
    "vacuum", "hollow", "blank", "oblivion",
    "dark", "darkness", "shadow", "shadows", "night", "black",
    "blackness", "dim", "murk", "gloom",
    "fracture", "fractured", "fractures", "fracturing",
    "shatter", "shattered", "shatters",
    "break", "broken", "breaking",
    "dissolve", "dissolved", "dissolving", "dissolution",
    "disintegrate", "disintegrating",
    "crumble", "crumbling", "decay", "decaying",
    "erode", "eroding", "erosion",
    "collapse", "collapsed", "collapsing",
    "fray", "fraying", "frayed",
    "wither", "withered", "fade", "fading", "faded",
    "bleed", "bleeding", "bleeds", "blood", "wound", "wounded",
    "scar", "scarred",
    "lost", "loss", "vanish", "vanished", "vanishing",
    "gone", "disappear", "disappeared", "missing", "absent",
    "cage", "caged", "trap", "trapped", "prison", "imprisoned",
    "isolation", "isolated", "alone", "solitude",
    "death", "dead", "die", "dying", "end", "ending",
    "perish", "doom", "doomed", "grave",
    "chaos", "chaotic", "twisted", "distorted", "warped",
    "ghost", "ghosts", "ghostly", "specter", "spectral",
    "phantom", "haunted", "haunting",
    "silence", "silent", "still", "stillness", "mute", "muted",
    "hush", "hushed", "quiet",
    "drift", "drifting", "drifted", "wander", "wandering", "aimless",
    "edge", "edges", "brink", "precipice", "threshold",
    "whisper", "whispers", "whispering", "murmur",
    NULL
};

/* ── Hash Set (for cluster membership) ───────────────────────────── */

typedef struct {
    char words[HASH_SIZE][MAX_WORD_LEN];
    int  used[HASH_SIZE];
    int  count;
} HashSet;

static unsigned hash_str(const char *s) {
    unsigned h = 5381;
    while (*s)
        h = ((h << 5) + h) ^ (unsigned char)*s++;
    return h;
}

static void hs_init(HashSet *hs) {
    memset(hs->used, 0, sizeof(hs->used));
    hs->count = 0;
}

static int hs_insert(HashSet *hs, const char *word) {
    unsigned idx = hash_str(word) & (HASH_SIZE - 1);
    for (int i = 0; i < HASH_SIZE; i++) {
        unsigned pos = (idx + i) & (HASH_SIZE - 1);
        if (!hs->used[pos]) {
            strncpy(hs->words[pos], word, MAX_WORD_LEN - 1);
            hs->words[pos][MAX_WORD_LEN - 1] = '\0';
            hs->used[pos] = 1;
            hs->count++;
            return 1;
        }
        if (strcmp(hs->words[pos], word) == 0)
            return 0;
    }
    return 0;
}

static int hs_contains(const HashSet *hs, const char *word) {
    unsigned idx = hash_str(word) & (HASH_SIZE - 1);
    for (int i = 0; i < HASH_SIZE; i++) {
        unsigned pos = (idx + i) & (HASH_SIZE - 1);
        if (!hs->used[pos])
            return 0;
        if (strcmp(hs->words[pos], word) == 0)
            return 1;
    }
    return 0;
}

/* ── Vocabulary (word → id mapping + frequency) ──────────────────── */

typedef struct {
    char   word[MAX_WORD_LEN];
    int    count;
    int    is_cluster;     /* 1 if in semantic cluster */
    int    positions[256]; /* up to 256 positions where this word appears */
    int    n_positions;
} VocabEntry;

static VocabEntry vocab[MAX_TYPES];
static int        vocab_size = 0;

/* Hash map for word → vocab index */
static int vocab_map[HASH_SIZE];
static int vocab_map_used[HASH_SIZE];

static void vocab_init(void) {
    vocab_size = 0;
    memset(vocab_map_used, 0, sizeof(vocab_map_used));
}

static int vocab_get_or_create(const char *word, int is_cluster) {
    unsigned idx = hash_str(word) & (HASH_SIZE - 1);
    for (int i = 0; i < HASH_SIZE; i++) {
        unsigned pos = (idx + i) & (HASH_SIZE - 1);
        if (!vocab_map_used[pos]) {
            /* Create new entry */
            if (vocab_size >= MAX_TYPES) return -1;
            vocab_map_used[pos] = 1;
            vocab_map[pos] = vocab_size;
            strncpy(vocab[vocab_size].word, word, MAX_WORD_LEN - 1);
            vocab[vocab_size].word[MAX_WORD_LEN - 1] = '\0';
            vocab[vocab_size].count = 0;
            vocab[vocab_size].is_cluster = is_cluster;
            vocab[vocab_size].n_positions = 0;
            return vocab_size++;
        }
        if (strcmp(vocab[vocab_map[pos]].word, word) == 0) {
            return vocab_map[pos];
        }
    }
    return -1;
}

/* ── Token sequence ──────────────────────────────────────────────── */

static int token_ids[MAX_TOKENS];      /* sequence of vocab indices */
static int token_cluster[MAX_TOKENS];  /* 1 if token is in cluster */
static int n_tokens = 0;

/* ── Bigram hash table ───────────────────────────────────────────── */

typedef struct {
    int w1, w2;   /* vocab indices */
    int count;
    int used;
} BigramEntry;

static BigramEntry bigrams[BIGRAM_HASH];

static void bigram_init(void) {
    memset(bigrams, 0, sizeof(bigrams));
}

static unsigned bigram_hash(int w1, int w2) {
    return (unsigned)(w1 * 31337 + w2 * 7919) & (BIGRAM_HASH - 1);
}

static void bigram_add(int w1, int w2) {
    unsigned idx = bigram_hash(w1, w2);
    for (int i = 0; i < BIGRAM_HASH; i++) {
        unsigned pos = (idx + i) & (BIGRAM_HASH - 1);
        if (!bigrams[pos].used) {
            bigrams[pos].w1 = w1;
            bigrams[pos].w2 = w2;
            bigrams[pos].count = 1;
            bigrams[pos].used = 1;
            return;
        }
        if (bigrams[pos].w1 == w1 && bigrams[pos].w2 == w2) {
            bigrams[pos].count++;
            return;
        }
    }
}

static int bigram_get(int w1, int w2) {
    unsigned idx = bigram_hash(w1, w2);
    for (int i = 0; i < BIGRAM_HASH; i++) {
        unsigned pos = (idx + i) & (BIGRAM_HASH - 1);
        if (!bigrams[pos].used) return 0;
        if (bigrams[pos].w1 == w1 && bigrams[pos].w2 == w2)
            return bigrams[pos].count;
    }
    return 0;
}

/* ── Tokenizer ───────────────────────────────────────────────────── */

static void lowercase(char *s) {
    for (; *s; s++)
        *s = tolower((unsigned char)*s);
}

static int is_word_char(int c) {
    return isalpha(c) || c == '\'' || c == '-';
}

/* ── Entropy computations ────────────────────────────────────────── */

static double log2_safe(double x) {
    if (x <= 0.0) return 0.0;
    return log(x) / log(2.0);
}

/* Unigram entropy H(W) */
static double compute_unigram_entropy(void) {
    double H = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        double p = (double)vocab[i].count / n_tokens;
        if (p > 0.0)
            H -= p * log2_safe(p);
    }
    return H;
}

/* Bigram entropy H(W_t, W_{t+1}) */
static double compute_bigram_entropy(void) {
    double H = 0.0;
    int total_bigrams = n_tokens - 1;
    if (total_bigrams <= 0) return 0.0;

    for (int i = 0; i < BIGRAM_HASH; i++) {
        if (bigrams[i].used) {
            double p = (double)bigrams[i].count / total_bigrams;
            if (p > 0.0)
                H -= p * log2_safe(p);
        }
    }
    return H;
}

/* Conditional entropy H(W_{t+1} | W_t) = H(W_t, W_{t+1}) - H(W_t) */
static double compute_conditional_entropy(void) {
    return compute_bigram_entropy() - compute_unigram_entropy();
}

/* ── Burstiness analysis ─────────────────────────────────────────── */

typedef struct {
    char   word[MAX_WORD_LEN];
    double beta;          /* burstiness parameter */
    double mean_iat;      /* mean inter-arrival time */
    double std_iat;       /* std dev of inter-arrival time */
    int    occurrences;
} BurstResult;

static BurstResult burst_results[MAX_TYPES];
static int         n_burst_results = 0;

static void compute_burstiness(void) {
    n_burst_results = 0;

    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].n_positions < 3) continue; /* need at least 3 for meaningful stats */

        /* Compute inter-arrival times */
        int n_iat = vocab[i].n_positions - 1;
        double *iats = (double *)malloc(n_iat * sizeof(double));
        if (!iats) continue;

        for (int j = 0; j < n_iat; j++) {
            iats[j] = (double)(vocab[i].positions[j + 1] - vocab[i].positions[j]);
        }

        /* Mean */
        double sum = 0.0;
        for (int j = 0; j < n_iat; j++) sum += iats[j];
        double mean = sum / n_iat;

        /* Std dev */
        double sq_sum = 0.0;
        for (int j = 0; j < n_iat; j++)
            sq_sum += (iats[j] - mean) * (iats[j] - mean);
        double std = sqrt(sq_sum / n_iat);

        /* Burstiness beta = (sigma/mu - 1) / (sigma/mu + 1) */
        double cv = (mean > 0.0) ? std / mean : 0.0;
        double beta = (cv - 1.0) / (cv + 1.0);

        if (n_burst_results < MAX_TYPES) {
            strncpy(burst_results[n_burst_results].word, vocab[i].word, MAX_WORD_LEN - 1);
            burst_results[n_burst_results].beta = beta;
            burst_results[n_burst_results].mean_iat = mean;
            burst_results[n_burst_results].std_iat = std;
            burst_results[n_burst_results].occurrences = vocab[i].n_positions;
            n_burst_results++;
        }

        free(iats);
    }
}

/* ── Cluster self-reinforcement ──────────────────────────────────── */

typedef struct {
    int cluster_after_cluster;    /* # times cluster token follows cluster token */
    int noncluster_after_cluster; /* # times non-cluster follows cluster */
    int cluster_after_noncluster; /* # times cluster follows non-cluster */
    int noncluster_after_noncluster;
    double p_cc;  /* P(cluster | prev=cluster) */
    double p_cn;  /* P(cluster | prev=non-cluster) */
    double basin_strength;  /* p_cc / p_cn */
} SelfReinforcement;

static SelfReinforcement compute_self_reinforcement(void) {
    SelfReinforcement sr = {0, 0, 0, 0, 0.0, 0.0, 0.0};

    for (int t = 1; t < n_tokens; t++) {
        int prev_c = token_cluster[t - 1];
        int curr_c = token_cluster[t];

        if (prev_c && curr_c) sr.cluster_after_cluster++;
        else if (prev_c && !curr_c) sr.noncluster_after_cluster++;
        else if (!prev_c && curr_c) sr.cluster_after_noncluster++;
        else sr.noncluster_after_noncluster++;
    }

    int after_cluster = sr.cluster_after_cluster + sr.noncluster_after_cluster;
    int after_noncluster = sr.cluster_after_noncluster + sr.noncluster_after_noncluster;

    sr.p_cc = (after_cluster > 0) ? (double)sr.cluster_after_cluster / after_cluster : 0.0;
    sr.p_cn = (after_noncluster > 0) ? (double)sr.cluster_after_noncluster / after_noncluster : 0.0;
    sr.basin_strength = (sr.p_cn > 0.0) ? sr.p_cc / sr.p_cn : 0.0;

    return sr;
}

/* ── Exponential semantic state tracker ──────────────────────────── */

typedef struct {
    double *state;     /* semantic state at each position */
    double  mean;
    double  max;
    double  min;
    double  final;
    int     n;
} SemanticTrace;

static SemanticTrace compute_semantic_trace(double alpha) {
    SemanticTrace tr;
    tr.n = n_tokens;
    tr.state = (double *)malloc(n_tokens * sizeof(double));
    if (!tr.state) {
        tr.n = 0;
        tr.mean = tr.max = tr.min = tr.final = 0.0;
        return tr;
    }

    tr.state[0] = token_cluster[0] ? 1.0 : 0.0;
    tr.max = tr.state[0];
    tr.min = tr.state[0];
    double sum = tr.state[0];

    for (int t = 1; t < n_tokens; t++) {
        tr.state[t] = alpha * (token_cluster[t] ? 1.0 : 0.0) +
                       (1.0 - alpha) * tr.state[t - 1];
        if (tr.state[t] > tr.max) tr.max = tr.state[t];
        if (tr.state[t] < tr.min) tr.min = tr.state[t];
        sum += tr.state[t];
    }

    tr.mean = sum / n_tokens;
    tr.final = tr.state[n_tokens - 1];
    return tr;
}

/* ── Compression ratio estimation ────────────────────────────────── */

/* Simple LZ77-style compression ratio estimate using unique bigram ratio */
static double estimate_compression_ratio(void) {
    int total_bigrams = n_tokens - 1;
    if (total_bigrams <= 0) return 1.0;

    int unique_bigrams = 0;
    for (int i = 0; i < BIGRAM_HASH; i++) {
        if (bigrams[i].used)
            unique_bigrams++;
    }

    /* Ratio of unique bigrams to total bigrams */
    /* Higher = less compressible = more random/diverse */
    return (double)unique_bigrams / total_bigrams;
}

/* ── Hapax and frequency spectrum ────────────────────────────────── */

typedef struct {
    int hapax;          /* words appearing exactly 1 time */
    int dis_legomena;   /* words appearing exactly 2 times */
    double ttr;         /* type-token ratio */
    double yules_k;     /* Yule's K measure */
} LexicalDiversity;

static LexicalDiversity compute_lexical_diversity(void) {
    LexicalDiversity ld = {0, 0, 0.0, 0.0};
    int freq_of_freq[256] = {0}; /* frequency of frequency classes */

    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].count == 1) ld.hapax++;
        if (vocab[i].count == 2) ld.dis_legomena++;
        if (vocab[i].count < 256)
            freq_of_freq[vocab[i].count]++;
    }

    ld.ttr = (n_tokens > 0) ? (double)vocab_size / n_tokens : 0.0;

    /* Yule's K = 10^4 * (M2 - N) / N^2 where M2 = sum(i^2 * V_i) */
    double M2 = 0.0;
    for (int i = 1; i < 256; i++) {
        M2 += (double)i * (double)i * freq_of_freq[i];
    }
    double N = (double)n_tokens;
    ld.yules_k = (N > 0) ? 10000.0 * (M2 - N) / (N * N) : 0.0;

    return ld;
}

/* ── Line length analysis ────────────────────────────────────────── */

typedef struct {
    int    n_lines;
    double mean;
    double stddev;
    double cv;       /* coefficient of variation */
    int    min_len;
    int    max_len;
} LineLengthStats;

static LineLengthStats line_stats;
static int line_lengths[4096];
static int n_lines_found = 0;

static void compute_line_lengths(const char *text) {
    n_lines_found = 0;
    int words_in_line = 0;
    int in_word = 0;

    for (const char *p = text; ; p++) {
        if (*p == '\0' || *p == '\n' || *p == '\r') {
            if (words_in_line > 0 && n_lines_found < 4096) {
                line_lengths[n_lines_found++] = words_in_line;
            }
            words_in_line = 0;
            in_word = 0;
            if (*p == '\0') break;
            /* Skip \r\n */
            if (*p == '\r' && *(p + 1) == '\n') p++;
        } else if (is_word_char(*p)) {
            if (!in_word) {
                words_in_line++;
                in_word = 1;
            }
        } else {
            in_word = 0;
        }
    }

    line_stats.n_lines = n_lines_found;
    if (n_lines_found == 0) {
        line_stats.mean = line_stats.stddev = line_stats.cv = 0.0;
        line_stats.min_len = line_stats.max_len = 0;
        return;
    }

    double sum = 0.0;
    line_stats.min_len = line_lengths[0];
    line_stats.max_len = line_lengths[0];
    for (int i = 0; i < n_lines_found; i++) {
        sum += line_lengths[i];
        if (line_lengths[i] < line_stats.min_len) line_stats.min_len = line_lengths[i];
        if (line_lengths[i] > line_stats.max_len) line_stats.max_len = line_lengths[i];
    }
    line_stats.mean = sum / n_lines_found;

    double sq_sum = 0.0;
    for (int i = 0; i < n_lines_found; i++) {
        double d = line_lengths[i] - line_stats.mean;
        sq_sum += d * d;
    }
    line_stats.stddev = sqrt(sq_sum / n_lines_found);
    line_stats.cv = (line_stats.mean > 0.0) ? line_stats.stddev / line_stats.mean : 0.0;
}

/* ── Cluster transition matrix ───────────────────────────────────── */

/* Track transitions between cluster/non-cluster with n-gram context */
typedef struct {
    /* Runs: consecutive cluster or non-cluster tokens */
    int    *run_lengths;
    int    *run_types;   /* 1 = cluster run, 0 = non-cluster run */
    int     n_runs;
    double  mean_cluster_run;
    double  mean_noncluster_run;
} RunAnalysis;

static RunAnalysis compute_runs(void) {
    RunAnalysis ra;
    ra.run_lengths = (int *)malloc(n_tokens * sizeof(int));
    ra.run_types   = (int *)malloc(n_tokens * sizeof(int));
    ra.n_runs = 0;
    ra.mean_cluster_run = 0.0;
    ra.mean_noncluster_run = 0.0;

    if (!ra.run_lengths || !ra.run_types || n_tokens == 0) return ra;

    int current_type = token_cluster[0];
    int current_len = 1;

    for (int t = 1; t < n_tokens; t++) {
        if (token_cluster[t] == current_type) {
            current_len++;
        } else {
            ra.run_types[ra.n_runs] = current_type;
            ra.run_lengths[ra.n_runs] = current_len;
            ra.n_runs++;
            current_type = token_cluster[t];
            current_len = 1;
        }
    }
    /* Final run */
    ra.run_types[ra.n_runs] = current_type;
    ra.run_lengths[ra.n_runs] = current_len;
    ra.n_runs++;

    /* Compute means */
    int sum_c = 0, count_c = 0;
    int sum_n = 0, count_n = 0;
    for (int i = 0; i < ra.n_runs; i++) {
        if (ra.run_types[i]) {
            sum_c += ra.run_lengths[i];
            count_c++;
        } else {
            sum_n += ra.run_lengths[i];
            count_n++;
        }
    }
    ra.mean_cluster_run = (count_c > 0) ? (double)sum_c / count_c : 0.0;
    ra.mean_noncluster_run = (count_n > 0) ? (double)sum_n / count_n : 0.0;

    return ra;
}

/* ── Sort helper for burst results ───────────────────────────────── */

static int burst_cmp(const void *a, const void *b) {
    double da = ((const BurstResult *)a)->beta;
    double db = ((const BurstResult *)b)->beta;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ── Main ────────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options] [file ...]\n"
        "\n"
        "Transformer generation signature analyzer.\n"
        "Reads from stdin if no files given.\n"
        "\n"
        "Analyzes text for computational signatures of autoregressive\n"
        "transformer generation: entropy, burstiness, self-reinforcement,\n"
        "compression ratio, and repetition penalty artifacts.\n"
        "\n"
        "Options:\n"
        "  -w FILE   Load cluster wordlist from FILE (one word per line)\n"
        "  -a FLOAT  Semantic trace decay rate alpha (default: 0.15)\n"
        "  -q        Quiet mode (machine-readable output)\n"
        "  -h        Show this help\n"
        "\n"
        "Default cluster: void/dissolution/darkness (~110 terms)\n"
        "\n"
        "Output includes:\n"
        "  - Unigram entropy H(W)\n"
        "  - Conditional entropy H(W_{t+1} | W_t)\n"
        "  - Lexical diversity (TTR, Yule's K, hapax rate)\n"
        "  - Burstiness parameter beta per word\n"
        "  - Cluster self-reinforcement (basin strength)\n"
        "  - Exponential semantic trace\n"
        "  - Compression ratio estimate\n"
        "  - Run-length analysis\n"
        "  - Line-length regularity (CV)\n",
        prog);
}

int main(int argc, char *argv[]) {
    HashSet cluster;
    double  alpha = 0.15;
    int     quiet = 0;
    const char *wordlist_file = NULL;

    /* Parse args */
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "-w") == 0 && argi + 1 < argc) {
            wordlist_file = argv[++argi];
        } else if (strcmp(argv[argi], "-a") == 0 && argi + 1 < argc) {
            alpha = atof(argv[++argi]);
        } else if (strcmp(argv[argi], "-q") == 0) {
            quiet = 1;
        } else if (strcmp(argv[argi], "-h") == 0 ||
                   strcmp(argv[argi], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[argi], "--") == 0) {
            argi++;
            break;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[argi]);
            return 1;
        }
        argi++;
    }

    /* Build cluster hash set */
    hs_init(&cluster);
    if (wordlist_file) {
        FILE *wf = fopen(wordlist_file, "r");
        if (!wf) { perror(wordlist_file); return 1; }
        char line[MAX_WORD_LEN];
        while (fgets(line, sizeof(line), wf)) {
            line[strcspn(line, "\r\n")] = '\0';
            lowercase(line);
            if (line[0] && line[0] != '#')
                hs_insert(&cluster, line);
        }
        fclose(wf);
    } else {
        for (int i = 0; DEFAULT_CLUSTER[i]; i++) {
            char buf[MAX_WORD_LEN];
            strncpy(buf, DEFAULT_CLUSTER[i], MAX_WORD_LEN - 1);
            buf[MAX_WORD_LEN - 1] = '\0';
            lowercase(buf);
            hs_insert(&cluster, buf);
        }
    }

    /* Read all input into a buffer (needed for line analysis) */
    char *text_buf = NULL;
    size_t text_len = 0;
    size_t text_cap = 0;

    FILE *inputs[256];
    int   n_inputs = 0;

    if (argi >= argc) {
        inputs[n_inputs++] = stdin;
    } else {
        for (; argi < argc && n_inputs < 256; argi++) {
            FILE *f = fopen(argv[argi], "r");
            if (!f) { perror(argv[argi]); continue; }
            inputs[n_inputs++] = f;
        }
    }

    /* Accumulate all text */
    text_cap = 65536;
    text_buf = (char *)malloc(text_cap);
    if (!text_buf) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }
    text_len = 0;

    for (int fi = 0; fi < n_inputs; fi++) {
        int c;
        while ((c = fgetc(inputs[fi])) != EOF) {
            if (text_len + 1 >= text_cap) {
                text_cap *= 2;
                text_buf = (char *)realloc(text_buf, text_cap);
                if (!text_buf) {
                    fprintf(stderr, "Out of memory\n");
                    return 1;
                }
            }
            text_buf[text_len++] = (char)c;
        }
        if (inputs[fi] != stdin)
            fclose(inputs[fi]);
    }
    text_buf[text_len] = '\0';

    /* Tokenize and build data structures */
    vocab_init();
    bigram_init();
    n_tokens = 0;

    char word[MAX_WORD_LEN];
    int wpos = 0;
    int prev_id = -1;

    for (size_t i = 0; i <= text_len; i++) {
        char c = (i < text_len) ? text_buf[i] : '\0';
        if (is_word_char(c)) {
            if (wpos < MAX_WORD_LEN - 1)
                word[wpos++] = c;
        } else {
            if (wpos > 0) {
                word[wpos] = '\0';
                lowercase(word);
                if (wpos >= 1 && isalpha((unsigned char)word[0]) && n_tokens < MAX_TOKENS) {
                    int is_cl = hs_contains(&cluster, word);
                    int id = vocab_get_or_create(word, is_cl);
                    if (id >= 0) {
                        vocab[id].count++;
                        if (vocab[id].n_positions < 256)
                            vocab[id].positions[vocab[id].n_positions++] = n_tokens;

                        token_ids[n_tokens] = id;
                        token_cluster[n_tokens] = is_cl;

                        if (prev_id >= 0)
                            bigram_add(prev_id, id);

                        prev_id = id;
                        n_tokens++;
                    }
                }
                wpos = 0;
            }
        }
    }

    if (n_tokens == 0) {
        fprintf(stderr, "No tokens found.\n");
        free(text_buf);
        return 1;
    }

    /* ── Compute all analyses ────────────────────────────────────── */

    double H_unigram = compute_unigram_entropy();
    double H_bigram  = compute_bigram_entropy();
    double H_cond    = compute_conditional_entropy();
    double max_entropy = log2_safe((double)vocab_size);

    LexicalDiversity ld = compute_lexical_diversity();

    compute_burstiness();

    SelfReinforcement sr = compute_self_reinforcement();

    SemanticTrace tr = compute_semantic_trace(alpha);

    double comp_ratio = estimate_compression_ratio();

    RunAnalysis ra = compute_runs();

    compute_line_lengths(text_buf);

    /* Count cluster stats */
    int cluster_tokens = 0;
    int cluster_types = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].is_cluster) {
            cluster_tokens += vocab[i].count;
            cluster_types++;
        }
    }
    double cluster_density = (double)cluster_tokens / n_tokens;

    /* ── Quiet mode ──────────────────────────────────────────────── */

    if (quiet) {
        printf("tokens\t%d\n", n_tokens);
        printf("types\t%d\n", vocab_size);
        printf("H_unigram\t%.4f\n", H_unigram);
        printf("H_cond\t%.4f\n", H_cond);
        printf("TTR\t%.4f\n", ld.ttr);
        printf("hapax_rate\t%.4f\n", (double)ld.hapax / vocab_size);
        printf("yules_k\t%.1f\n", ld.yules_k);
        printf("cluster_density\t%.4f\n", cluster_density);
        printf("basin_strength\t%.3f\n", sr.basin_strength);
        printf("comp_ratio\t%.4f\n", comp_ratio);
        printf("line_cv\t%.4f\n", line_stats.cv);
        printf("sem_trace_mean\t%.4f\n", tr.mean);
        free(text_buf);
        if (tr.state) free(tr.state);
        if (ra.run_lengths) free(ra.run_lengths);
        if (ra.run_types) free(ra.run_types);
        return 0;
    }

    /* ── Full report ─────────────────────────────────────────────── */

    printf("==================================================================\n");
    printf("  TRANSFORMER GENERATION SIGNATURE ANALYSIS\n");
    printf("==================================================================\n\n");

    /* 1. Basic stats */
    printf("  1. CORPUS STATISTICS\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Total tokens:              %d\n", n_tokens);
    printf("  Unique types:              %d\n", vocab_size);
    printf("  Cluster tokens:            %d (%.1f%%)\n",
           cluster_tokens, cluster_density * 100.0);
    printf("  Cluster types:             %d\n\n", cluster_types);

    /* 2. Entropy analysis */
    printf("  2. ENTROPY ANALYSIS\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Unigram entropy H(W):      %.3f bits\n", H_unigram);
    printf("  Max entropy (uniform):     %.3f bits\n", max_entropy);
    printf("  Entropy ratio H/Hmax:      %.3f\n", H_unigram / max_entropy);
    printf("  Bigram entropy H(W,W'):    %.3f bits\n", H_bigram);
    printf("  Conditional H(W'|W):       %.3f bits\n", H_cond);
    printf("  Predictability (1-H/Hmax): %.3f\n\n", 1.0 - H_unigram / max_entropy);

    printf("  Interpretation:\n");
    double h_ratio = H_unigram / max_entropy;
    if (h_ratio > 0.90)
        printf("    -> NEAR-MAXIMUM entropy: strong repetition penalty signature\n");
    else if (h_ratio > 0.80)
        printf("    -> ELEVATED entropy: moderate repetition penalty signature\n");
    else if (h_ratio > 0.65)
        printf("    -> NORMAL entropy: typical natural language\n");
    else
        printf("    -> LOW entropy: highly repetitive or formulaic\n");

    printf("    -> English prose: ~0.65-0.75 | Lyrics: ~0.70-0.85 | Penalized LLM: ~0.85-0.95\n\n");

    /* 3. Lexical diversity */
    printf("  3. LEXICAL DIVERSITY\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Type-Token Ratio:          %.3f\n", ld.ttr);
    printf("  Hapax legomena:            %d (%.1f%% of types)\n",
           ld.hapax, 100.0 * ld.hapax / (vocab_size > 0 ? vocab_size : 1));
    printf("  Hapax %% of tokens:         %.1f%%\n",
           100.0 * ld.hapax / (n_tokens > 0 ? n_tokens : 1));
    printf("  Dis legomena:              %d\n", ld.dis_legomena);
    printf("  Yule's K:                  %.1f\n\n", ld.yules_k);

    printf("  Interpretation:\n");
    double hapax_rate = (double)ld.hapax / vocab_size;
    if (hapax_rate > 0.62)
        printf("    -> HIGH hapax rate: consistent with repetition penalty\n");
    else if (hapax_rate > 0.50)
        printf("    -> NORMAL hapax rate: typical of natural text\n");
    else
        printf("    -> LOW hapax rate: repetitive text\n");
    printf("    -> Natural text: 50-60%% | Penalized LLM: 60-70%% | Random: ~63%%\n\n");

    /* 4. Burstiness */
    printf("  4. BURSTINESS ANALYSIS (words with 3+ occurrences)\n");
    printf("  -----------------------------------------------------------\n");

    qsort(burst_results, n_burst_results, sizeof(BurstResult), burst_cmp);

    printf("  +-------------------------+------+---------+--------+--------+\n");
    printf("  | Word                    | Freq | Mean IAT| Std IAT|  Beta  |\n");
    printf("  +-------------------------+------+---------+--------+--------+\n");

    int show_burst = n_burst_results < 25 ? n_burst_results : 25;
    double beta_sum = 0.0;
    for (int i = 0; i < show_burst; i++) {
        printf("  | %-23s | %4d | %7.1f | %6.1f | %+5.2f  |\n",
               burst_results[i].word,
               burst_results[i].occurrences,
               burst_results[i].mean_iat,
               burst_results[i].std_iat,
               burst_results[i].beta);
        beta_sum += burst_results[i].beta;
    }
    printf("  +-------------------------+------+---------+--------+--------+\n");

    double mean_beta = (n_burst_results > 0) ? beta_sum / n_burst_results : 0.0;
    printf("  Mean beta across all qualifying words: %+.3f\n\n", mean_beta);

    printf("  Interpretation:\n");
    printf("    -> beta > 0: bursty (natural language pattern)\n");
    printf("    -> beta ~ 0: Poisson (random)\n");
    printf("    -> beta < 0: anti-bursty (repetition penalty signature)\n");
    if (mean_beta < -0.1)
        printf("    -> DETECTED: Anti-bursty distribution (repetition penalty active)\n");
    else if (mean_beta < 0.1)
        printf("    -> DETECTED: Near-Poisson distribution (weak/no burstiness)\n");
    else
        printf("    -> DETECTED: Bursty distribution (natural language pattern)\n");
    printf("\n");

    /* 5. Self-reinforcement */
    printf("  5. CLUSTER SELF-REINFORCEMENT\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Transitions:\n");
    printf("    cluster -> cluster:      %d\n", sr.cluster_after_cluster);
    printf("    cluster -> non-cluster:  %d\n", sr.noncluster_after_cluster);
    printf("    non-cluster -> cluster:  %d\n", sr.cluster_after_noncluster);
    printf("    non-cluster -> non-clust:%d\n", sr.noncluster_after_noncluster);
    printf("  P(cluster | prev=cluster): %.3f\n", sr.p_cc);
    printf("  P(cluster | prev=non-cl):  %.3f\n", sr.p_cn);
    printf("  Basin strength (ratio):    %.2f\n\n", sr.basin_strength);

    printf("  Interpretation:\n");
    if (sr.basin_strength > 1.5)
        printf("    -> STRONG self-reinforcement: attractor basin detected\n");
    else if (sr.basin_strength > 1.1)
        printf("    -> WEAK self-reinforcement: mild attractor basin\n");
    else if (sr.basin_strength > 0.9)
        printf("    -> NO self-reinforcement: cluster density is conditioning-driven\n");
    else
        printf("    -> ANTI-reinforcement: cluster tokens are dispersed\n");
    printf("\n");

    /* 6. Semantic trace */
    printf("  6. SEMANTIC STATE TRACE (alpha=%.2f)\n", alpha);
    printf("  -----------------------------------------------------------\n");
    printf("  Mean semantic state:       %.3f\n", tr.mean);
    printf("  Max semantic state:        %.3f\n", tr.max);
    printf("  Min semantic state:        %.3f\n", tr.min);
    printf("  Final semantic state:      %.3f\n", tr.final);
    printf("  Baseline (cluster density):%.3f\n\n", cluster_density);

    printf("  Interpretation:\n");
    if (fabs(tr.mean - cluster_density) < 0.02)
        printf("    -> Semantic state MATCHES cluster density: uniform distribution\n");
    else if (tr.mean > cluster_density + 0.02)
        printf("    -> Semantic state ABOVE density: cluster terms clump together\n");
    else
        printf("    -> Semantic state BELOW density: cluster terms are dispersed\n");

    /* Mini sparkline of semantic trace (10 bins) */
    if (tr.n > 0 && tr.state) {
        printf("\n  Trace (10 bins, 0=none, #=cluster):\n    ");
        int bin_size = tr.n / 10;
        for (int b = 0; b < 10; b++) {
            double bsum = 0.0;
            int start = b * bin_size;
            int end = (b == 9) ? tr.n : (b + 1) * bin_size;
            for (int t = start; t < end; t++)
                bsum += tr.state[t];
            double bavg = bsum / (end - start);
            /* Map to 0-8 scale */
            int level = (int)(bavg * 8.0 + 0.5);
            if (level > 8) level = 8;
            char bar[] = "........";
            for (int j = 0; j < level; j++) bar[j] = '#';
            printf("[%s] ", bar);
        }
        printf("\n");
    }
    printf("\n");

    /* 7. Compression ratio */
    printf("  7. COMPRESSION RATIO ESTIMATE\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Unique bigrams:            %d (of %d total)\n",
           (int)(comp_ratio * (n_tokens - 1)), n_tokens - 1);
    printf("  Bigram uniqueness ratio:   %.3f\n\n", comp_ratio);

    printf("  Interpretation:\n");
    if (comp_ratio > 0.90)
        printf("    -> VERY HIGH uniqueness: near-zero repetition (strong penalty)\n");
    else if (comp_ratio > 0.75)
        printf("    -> HIGH uniqueness: low repetition (moderate penalty)\n");
    else if (comp_ratio > 0.50)
        printf("    -> MODERATE uniqueness: some repetition (natural or light penalty)\n");
    else
        printf("    -> LOW uniqueness: highly repetitive (chorus-heavy or formulaic)\n");
    printf("\n");

    /* 8. Run-length analysis */
    printf("  8. RUN-LENGTH ANALYSIS\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Total runs:                %d\n", ra.n_runs);
    printf("  Mean cluster run length:   %.2f\n", ra.mean_cluster_run);
    printf("  Mean non-cluster run:      %.2f\n\n", ra.mean_noncluster_run);

    printf("  Interpretation:\n");
    if (ra.mean_cluster_run < 1.5)
        printf("    -> Short cluster runs: cluster tokens are isolated/dispersed\n");
    else if (ra.mean_cluster_run < 3.0)
        printf("    -> Medium cluster runs: some local clustering\n");
    else
        printf("    -> Long cluster runs: strong local clustering\n");
    printf("\n");

    /* 9. Line-length regularity */
    printf("  9. LINE-LENGTH REGULARITY\n");
    printf("  -----------------------------------------------------------\n");
    printf("  Lines analyzed:            %d\n", line_stats.n_lines);
    printf("  Mean words/line:           %.1f\n", line_stats.mean);
    printf("  Std dev:                   %.1f\n", line_stats.stddev);
    printf("  CV (coeff. of variation):  %.3f\n", line_stats.cv);
    printf("  Min words/line:            %d\n", line_stats.min_len);
    printf("  Max words/line:            %d\n\n", line_stats.max_len);

    printf("  Interpretation:\n");
    if (line_stats.cv < 0.15)
        printf("    -> EXTREME regularity: template-driven generation\n");
    else if (line_stats.cv < 0.25)
        printf("    -> HIGH regularity: structured generation with some variance\n");
    else if (line_stats.cv < 0.40)
        printf("    -> MODERATE regularity: typical structured lyrics\n");
    else
        printf("    -> LOW regularity: free-form or prose-like\n");

    printf("    -> Formal meter: 0.05-0.10 | Structured lyrics: 0.15-0.30 | Free verse: 0.30-0.50\n\n");

    /* 10. Summary verdict */
    printf("  ===========================================================\n");
    printf("  GENERATION SIGNATURE SUMMARY\n");
    printf("  ===========================================================\n\n");

    int sig_count = 0;
    printf("  Transformer generation indicators:\n");

    if (h_ratio > 0.85) {
        printf("    [X] High entropy ratio (%.3f) -> repetition penalty\n", h_ratio);
        sig_count++;
    } else {
        printf("    [ ] Entropy ratio (%.3f) within natural range\n", h_ratio);
    }

    if (hapax_rate > 0.62) {
        printf("    [X] Elevated hapax rate (%.1f%%) -> repetition penalty\n",
               hapax_rate * 100.0);
        sig_count++;
    } else {
        printf("    [ ] Hapax rate (%.1f%%) within natural range\n",
               hapax_rate * 100.0);
    }

    if (mean_beta < -0.05) {
        printf("    [X] Anti-bursty distribution (beta=%.3f) -> repetition penalty\n",
               mean_beta);
        sig_count++;
    } else {
        printf("    [ ] Burstiness (beta=%.3f) within natural range\n", mean_beta);
    }

    if (line_stats.cv < 0.20 && line_stats.n_lines > 5) {
        printf("    [X] Extreme line regularity (CV=%.3f) -> template generation\n",
               line_stats.cv);
        sig_count++;
    } else if (line_stats.n_lines > 5) {
        printf("    [ ] Line regularity (CV=%.3f) within normal range\n", line_stats.cv);
    }

    if (comp_ratio > 0.85) {
        printf("    [X] High bigram uniqueness (%.3f) -> repetition penalty\n",
               comp_ratio);
        sig_count++;
    } else {
        printf("    [ ] Bigram uniqueness (%.3f) within normal range\n", comp_ratio);
    }

    if (sr.basin_strength > 0.9 && sr.basin_strength < 1.5) {
        printf("    [X] Weak self-reinforcement (%.2f) -> conditioning-driven\n",
               sr.basin_strength);
        sig_count++;
    } else if (sr.basin_strength >= 1.5) {
        printf("    [!] Strong self-reinforcement (%.2f) -> attractor basin\n",
               sr.basin_strength);
    }

    printf("\n  Signatures detected: %d/6\n", sig_count);
    if (sig_count >= 4)
        printf("  Verdict: STRONG transformer generation signature\n");
    else if (sig_count >= 2)
        printf("  Verdict: MODERATE transformer generation signature\n");
    else
        printf("  Verdict: WEAK/NO transformer generation signature\n");

    printf("\n");

    /* Cleanup */
    free(text_buf);
    if (tr.state) free(tr.state);
    if (ra.run_lengths) free(ra.run_lengths);
    if (ra.run_types) free(ra.run_types);

    return 0;
}
