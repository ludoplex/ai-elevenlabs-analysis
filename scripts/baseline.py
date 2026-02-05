#!/usr/bin/env python3
"""
Baseline & Comparison Framework

Loads analysis JSON outputs from collect.py, computes corpus-level baselines,
performs cross-song and cross-platform comparisons, and detects statistical outliers.

Usage:
    # Compute baselines from all collected JSON files
    python baseline.py output/ --report

    # Compare a single song against the corpus
    python baseline.py output/ --compare output/shadows_of_geometry_abc123.json

    # Cross-platform comparison
    python baseline.py output/ --cross-platform

    # Outlier detection
    python baseline.py output/ --outliers

    # Full report (all analyses)
    python baseline.py output/ --full --format markdown > report.md

    # Export baseline data for other repos
    python baseline.py output/ --export-baselines baselines.json
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

__version__ = "0.2.0"


# ─── Statistical Utilities ───────────────────────────────────────

def mean(values: list[float]) -> float:
    """Arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def std(values: list[float], population: bool = False) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    ddof = 0 if population else 1
    variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def median(values: list[float]) -> float:
    """Median."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def iqr(values: list[float]) -> tuple[float, float, float]:
    """Q1, median, Q3."""
    s = sorted(values)
    n = len(s)
    q1 = median(s[: n // 2])
    q3 = median(s[(n + 1) // 2 :])
    return q1, median(s), q3


def z_score(value: float, mean_val: float, std_val: float) -> float:
    """Z-score of a single value against a distribution."""
    if std_val == 0:
        return 0.0
    return (value - mean_val) / std_val


def z_test_proportion(p_obs: float, p_exp: float, n: int) -> dict:
    """One-tailed z-test for proportions."""
    se = math.sqrt(p_exp * (1 - p_exp) / n) if p_exp > 0 and p_exp < 1 and n > 0 else 0.001
    z = (p_obs - p_exp) / se
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return {"z": round(z, 3), "p": p_value}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return round(abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))), 4)


def confidence_interval(m: float, s: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Approximate confidence interval (normal assumption)."""
    # z* for common confidence levels
    z_stars = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z_star = z_stars.get(confidence, 1.960)
    margin = z_star * s / math.sqrt(n) if n > 0 else 0
    return (round(m - margin, 6), round(m + margin, 6))


# ─── Corpus Loader ───────────────────────────────────────────────

class CorpusLoader:
    """Load and organize analysis JSON files from collect.py output."""

    def __init__(self, directory: str | Path, pattern: str = "*.json"):
        self.directory = Path(directory)
        self.pattern = pattern
        self.entries = []
        self._load()

    def _load(self):
        """Load all JSON files matching pattern."""
        for filepath in sorted(self.directory.glob(self.pattern)):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                # Only include valid analysis results (not catalogs, baselines, etc.)
                if "tokens" in data and "semantic_clusters" in data:
                    data["_source_json"] = str(filepath)
                    self.entries.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: skipping {filepath}: {e}", file=sys.stderr)

    @property
    def count(self) -> int:
        return len(self.entries)

    def by_platform(self) -> dict[str, list[dict]]:
        """Group entries by platform."""
        groups = defaultdict(list)
        for entry in self.entries:
            platform = entry.get("platform", "unknown")
            groups[platform].append(entry)
        return dict(groups)

    def titles(self) -> list[str]:
        """List all song titles."""
        return [
            e.get("metadata", {}).get("title", e.get("metadata", {}).get("source_file", "untitled"))
            for e in self.entries
        ]

    def extract_metric(self, path: str) -> list[tuple[str, float]]:
        """Extract a nested metric from all entries.

        path is dot-separated, e.g. 'semantic_clusters.void_dissolution.percent'
        Returns list of (title, value) tuples.
        """
        results = []
        for entry in self.entries:
            title = entry.get("metadata", {}).get("title", "untitled")
            val = entry
            try:
                for key in path.split("."):
                    val = val[key]
                results.append((title, float(val)))
            except (KeyError, TypeError, ValueError):
                continue
        return results


# ─── Baseline Computer ───────────────────────────────────────────

class BaselineComputer:
    """Compute corpus-level baselines from collected analysis data."""

    def __init__(self, corpus: CorpusLoader):
        self.corpus = corpus

    def compute_all(self) -> dict:
        """Compute comprehensive baselines from the entire corpus."""
        if self.corpus.count == 0:
            return {"error": "No data in corpus"}

        baselines = {
            "corpus_size": self.corpus.count,
            "token_statistics": self._token_stats(),
            "lexical_baselines": self._lexical_baselines(),
            "cluster_baselines": self._cluster_baselines(),
            "structural_baselines": self._structural_baselines(),
        }

        return baselines

    def _token_stats(self) -> dict:
        """Token count statistics across corpus."""
        counts = [e["tokens"]["total"] for e in self.corpus.entries]
        unique_counts = [e["tokens"]["unique"] for e in self.corpus.entries]
        return {
            "total_tokens_corpus": sum(counts),
            "mean_tokens_per_song": round(mean(counts), 1),
            "std_tokens_per_song": round(std(counts), 1),
            "median_tokens_per_song": round(median(counts), 1),
            "mean_unique_per_song": round(mean(unique_counts), 1),
        }

    def _lexical_baselines(self) -> dict:
        """Lexical diversity metric baselines."""
        metrics = {}
        for metric_name in ["type_token_ratio", "yules_k", "hapax_ratio", "simpsons_diversity"]:
            values = []
            for entry in self.corpus.entries:
                v = entry.get("lexical_metrics", {}).get(metric_name)
                if v is not None:
                    values.append(float(v))
            if values:
                m = mean(values)
                s = std(values)
                metrics[metric_name] = {
                    "mean": round(m, 4),
                    "std": round(s, 4),
                    "median": round(median(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "ci_95": confidence_interval(m, s, len(values)),
                    "n": len(values),
                }
        return metrics

    def _cluster_baselines(self) -> dict:
        """Semantic cluster density baselines."""
        cluster_data = defaultdict(list)

        for entry in self.corpus.entries:
            clusters = entry.get("semantic_clusters", {})
            for cname, cdata in clusters.items():
                if cname.startswith("_"):
                    continue
                proportion = cdata.get("proportion", 0)
                cluster_data[cname].append(proportion)

        baselines = {}
        for cname, proportions in cluster_data.items():
            if proportions:
                m = mean(proportions)
                s = std(proportions)
                baselines[cname] = {
                    "mean_proportion": round(m, 6),
                    "std_proportion": round(s, 6),
                    "mean_percent": round(m * 100, 2),
                    "std_percent": round(s * 100, 2),
                    "median_proportion": round(median(proportions), 6),
                    "min_proportion": round(min(proportions), 6),
                    "max_proportion": round(max(proportions), 6),
                    "ci_95": confidence_interval(m, s, len(proportions)),
                    "n": len(proportions),
                }

        return baselines

    def _structural_baselines(self) -> dict:
        """Structural metric baselines."""
        avg_wpls = []
        cv_wpls = []
        line_counts = []

        for entry in self.corpus.entries:
            sm = entry.get("structural_metrics", {})
            if sm.get("avg_words_per_line"):
                avg_wpls.append(float(sm["avg_words_per_line"]))
            if sm.get("cv_words_per_line"):
                cv_wpls.append(float(sm["cv_words_per_line"]))
            if sm.get("total_lines"):
                line_counts.append(int(sm["total_lines"]))

        result = {}
        for name, values in [("avg_words_per_line", avg_wpls),
                              ("cv_words_per_line", cv_wpls),
                              ("total_lines", line_counts)]:
            if values:
                m = mean(values)
                s = std(values)
                result[name] = {
                    "mean": round(m, 3),
                    "std": round(s, 3),
                    "median": round(median(values), 3),
                    "n": len(values),
                }

        return result


# ─── Comparison Engine ───────────────────────────────────────────

class ComparisonEngine:
    """Compare individual songs against corpus baselines."""

    def __init__(self, baselines: dict, corpus: CorpusLoader):
        self.baselines = baselines
        self.corpus = corpus

    def compare_song(self, song: dict) -> dict:
        """Compare a single song against corpus baselines."""
        results = {
            "title": song.get("metadata", {}).get("title", "untitled"),
            "platform": song.get("platform", "unknown"),
            "comparisons": {},
        }

        # Lexical comparisons
        lexical_baselines = self.baselines.get("lexical_baselines", {})
        song_lexical = song.get("lexical_metrics", {})
        for metric, baseline in lexical_baselines.items():
            song_val = song_lexical.get(metric)
            if song_val is not None:
                z = z_score(float(song_val), baseline["mean"], baseline["std"])
                results["comparisons"][f"lexical.{metric}"] = {
                    "value": round(float(song_val), 4),
                    "baseline_mean": baseline["mean"],
                    "baseline_std": baseline["std"],
                    "z_score": round(z, 3),
                    "is_outlier": abs(z) > 2.0,
                    "direction": "above" if z > 0 else "below",
                }

        # Cluster density comparisons
        cluster_baselines = self.baselines.get("cluster_baselines", {})
        song_clusters = song.get("semantic_clusters", {})
        for cname, baseline in cluster_baselines.items():
            song_cluster = song_clusters.get(cname, {})
            song_prop = song_cluster.get("proportion", 0)
            n = song.get("tokens", {}).get("total", 0)

            z = z_score(song_prop, baseline["mean_proportion"], baseline["std_proportion"])

            # Also do a proper z-test against baseline proportion
            ztest = z_test_proportion(song_prop, baseline["mean_proportion"], n) if n > 0 else {"z": 0, "p": 1}
            h = cohens_h(song_prop, baseline["mean_proportion"])

            results["comparisons"][f"cluster.{cname}"] = {
                "value": round(song_prop, 6),
                "percent": round(song_prop * 100, 2),
                "baseline_mean": baseline["mean_proportion"],
                "baseline_std": baseline["std_proportion"],
                "z_score_within_corpus": round(z, 3),
                "z_test_vs_baseline": ztest,
                "cohens_h": h,
                "is_outlier": abs(z) > 2.0,
                "ratio": round(song_prop / baseline["mean_proportion"], 2) if baseline["mean_proportion"] > 0 else 0,
            }

        return results

    def compare_all(self) -> list[dict]:
        """Compare every song in corpus against corpus baselines."""
        return [self.compare_song(entry) for entry in self.corpus.entries]


# ─── Cross-Platform Comparison ───────────────────────────────────

class CrossPlatformAnalyzer:
    """Compare metrics across different platforms/sources."""

    def __init__(self, corpus: CorpusLoader):
        self.corpus = corpus

    def analyze(self) -> dict:
        """Compute per-platform statistics and compare."""
        by_platform = self.corpus.by_platform()

        if len(by_platform) < 2:
            return {
                "note": f"Only {len(by_platform)} platform(s) in corpus. Need ≥2 for cross-platform comparison.",
                "platforms": list(by_platform.keys()),
            }

        platform_stats = {}
        for platform, entries in by_platform.items():
            # Void dissolution density
            void_props = []
            ttr_vals = []
            yules_vals = []
            for e in entries:
                vd = e.get("semantic_clusters", {}).get("void_dissolution", {})
                if vd:
                    void_props.append(vd.get("proportion", 0))
                lm = e.get("lexical_metrics", {})
                if lm.get("type_token_ratio"):
                    ttr_vals.append(lm["type_token_ratio"])
                if lm.get("yules_k"):
                    yules_vals.append(lm["yules_k"])

            platform_stats[platform] = {
                "n": len(entries),
                "void_dissolution": {
                    "mean": round(mean(void_props) * 100, 2) if void_props else None,
                    "std": round(std(void_props) * 100, 2) if void_props else None,
                    "values": [round(v * 100, 2) for v in void_props],
                },
                "type_token_ratio": {
                    "mean": round(mean(ttr_vals), 4) if ttr_vals else None,
                    "std": round(std(ttr_vals), 4) if ttr_vals else None,
                },
                "yules_k": {
                    "mean": round(mean(yules_vals), 1) if yules_vals else None,
                    "std": round(std(yules_vals), 1) if yules_vals else None,
                },
            }

        # Pairwise comparisons for void dissolution
        comparisons = []
        platforms = list(platform_stats.keys())
        for i in range(len(platforms)):
            for j in range(i + 1, len(platforms)):
                p1, p2 = platforms[i], platforms[j]
                s1 = platform_stats[p1]["void_dissolution"]
                s2 = platform_stats[p2]["void_dissolution"]
                if s1["mean"] is not None and s2["mean"] is not None:
                    diff = s1["mean"] - s2["mean"]
                    comparisons.append({
                        "platform_a": p1,
                        "platform_b": p2,
                        "void_density_diff": round(diff, 2),
                        "direction": f"{p1} higher" if diff > 0 else f"{p2} higher",
                    })

        return {
            "platform_stats": platform_stats,
            "comparisons": comparisons,
        }


# ─── Outlier Detection ───────────────────────────────────────────

class OutlierDetector:
    """Detect statistical outliers within the corpus."""

    def __init__(self, corpus: CorpusLoader, baselines: dict):
        self.corpus = corpus
        self.baselines = baselines

    def detect(self, z_threshold: float = 2.0) -> dict:
        """Find outlier songs across all metrics."""
        outliers = []

        engine = ComparisonEngine(self.baselines, self.corpus)

        for entry in self.corpus.entries:
            comparison = engine.compare_song(entry)
            song_outliers = []

            for metric_key, comp in comparison["comparisons"].items():
                if comp.get("is_outlier", False):
                    song_outliers.append({
                        "metric": metric_key,
                        "value": comp.get("value") or comp.get("percent"),
                        "z_score": comp.get("z_score_within_corpus") or comp.get("z_score"),
                        "direction": comp.get("direction", ""),
                    })

            if song_outliers:
                outliers.append({
                    "title": comparison["title"],
                    "platform": comparison["platform"],
                    "source": entry.get("metadata", {}).get("source_file", ""),
                    "outlier_metrics": song_outliers,
                    "outlier_count": len(song_outliers),
                })

        # Sort by number of outlier metrics (most unusual first)
        outliers.sort(key=lambda x: x["outlier_count"], reverse=True)

        return {
            "z_threshold": z_threshold,
            "total_songs": self.corpus.count,
            "songs_with_outliers": len(outliers),
            "outliers": outliers,
        }


# ─── Report Formatters ───────────────────────────────────────────

def format_report_markdown(baselines: dict, comparisons: list[dict] | None = None,
                           cross_platform: dict | None = None,
                           outliers: dict | None = None) -> str:
    """Format a full markdown report."""
    lines = []
    lines.append("# Corpus Baseline & Comparison Report\n")
    lines.append(f"**Corpus size:** {baselines.get('corpus_size', 0)} songs\n")

    # Token stats
    ts = baselines.get("token_statistics", {})
    if ts:
        lines.append("## Token Statistics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|------:|")
        lines.append(f"| Total tokens (corpus) | {ts.get('total_tokens_corpus', 0):,} |")
        lines.append(f"| Mean tokens/song | {ts.get('mean_tokens_per_song', 0):.1f} |")
        lines.append(f"| Std tokens/song | {ts.get('std_tokens_per_song', 0):.1f} |")
        lines.append(f"| Mean unique/song | {ts.get('mean_unique_per_song', 0):.1f} |")
        lines.append("")

    # Lexical baselines
    lb = baselines.get("lexical_baselines", {})
    if lb:
        lines.append("## Lexical Diversity Baselines\n")
        lines.append("| Metric | Mean | Std | Median | 95% CI |")
        lines.append("|--------|------|-----|--------|--------|")
        for metric, data in lb.items():
            ci = data.get("ci_95", (0, 0))
            lines.append(
                f"| {metric} | {data['mean']:.4f} | {data['std']:.4f} | "
                f"{data['median']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
            )
        lines.append("")

    # Cluster baselines
    cb = baselines.get("cluster_baselines", {})
    if cb:
        lines.append("## Semantic Cluster Density Baselines\n")
        lines.append("| Cluster | Mean % | Std % | Median % | Range | 95% CI |")
        lines.append("|---------|--------|-------|----------|-------|--------|")
        for cname, data in cb.items():
            ci = data.get("ci_95", (0, 0))
            lines.append(
                f"| {cname} | {data['mean_percent']:.2f} | {data['std_percent']:.2f} | "
                f"{data['median_proportion'] * 100:.2f} | "
                f"{data['min_proportion'] * 100:.2f}-{data['max_proportion'] * 100:.2f} | "
                f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}] |"
            )
        lines.append("")

    # Per-song comparisons
    if comparisons:
        lines.append("## Per-Song Comparisons\n")
        for comp in comparisons:
            title = comp.get("title", "untitled")
            platform = comp.get("platform", "unknown")
            lines.append(f"### {title} ({platform})\n")
            lines.append("| Metric | Value | Baseline | Z-score | Outlier? |")
            lines.append("|--------|-------|----------|---------|----------|")
            for metric_key, data in comp.get("comparisons", {}).items():
                val = data.get("percent", data.get("value", ""))
                base = data.get("baseline_mean", "")
                if isinstance(base, float) and base < 1:
                    base = f"{base * 100:.2f}%"
                z = data.get("z_score_within_corpus", data.get("z_score", ""))
                outlier = "⚠️ YES" if data.get("is_outlier") else "no"
                lines.append(f"| {metric_key} | {val} | {base} | {z} | {outlier} |")
            lines.append("")

    # Cross-platform
    if cross_platform and cross_platform.get("platform_stats"):
        lines.append("## Cross-Platform Comparison\n")
        ps = cross_platform["platform_stats"]
        lines.append("| Platform | N | Void % (mean) | Void % (std) | TTR (mean) | Yule's K |")
        lines.append("|----------|---|---------------|--------------|------------|----------|")
        for platform, data in ps.items():
            vd = data["void_dissolution"]
            ttr = data["type_token_ratio"]
            yk = data["yules_k"]
            lines.append(
                f"| {platform} | {data['n']} | "
                f"{vd['mean'] or 'N/A'} | {vd['std'] or 'N/A'} | "
                f"{ttr['mean'] or 'N/A'} | {yk['mean'] or 'N/A'} |"
            )
        lines.append("")

    # Outliers
    if outliers and outliers.get("outliers"):
        lines.append("## Outlier Songs\n")
        lines.append(f"Z-score threshold: ±{outliers['z_threshold']}\n")
        lines.append(f"Songs with outliers: {outliers['songs_with_outliers']}/{outliers['total_songs']}\n")
        for o in outliers["outliers"]:
            lines.append(f"### {o['title']} ({o['platform']})")
            lines.append(f"Outlier metrics: {o['outlier_count']}\n")
            for m in o["outlier_metrics"]:
                lines.append(f"- **{m['metric']}**: {m['value']} (z={m['z_score']}, {m['direction']})")
            lines.append("")

    return "\n".join(lines)


def format_report_json(baselines: dict, comparisons: list[dict] | None = None,
                       cross_platform: dict | None = None,
                       outliers: dict | None = None) -> str:
    """Format full report as JSON."""
    report = {
        "baselines": baselines,
    }
    if comparisons:
        report["comparisons"] = comparisons
    if cross_platform:
        report["cross_platform"] = cross_platform
    if outliers:
        report["outliers"] = outliers
    return json.dumps(report, indent=2, ensure_ascii=False)


# ─── External Baseline Import ────────────────────────────────────

# Literature-based genre baselines (from methodology.md)
GENRE_BASELINES = {
    "general_rock": {
        "void_dissolution_proportion": 0.02,
        "source": "Cross-genre lyric corpora (Fell 2014, Nichols et al. 2009)",
    },
    "general_prog": {
        "void_dissolution_proportion": 0.03,
        "source": "Prog-specific subcorpus (Yes, Genesis, Rush, Dream Theater)",
    },
    "dark_prog": {
        "void_dissolution_proportion": 0.05,
        "source": "Tool, Porcupine Tree, dark-era Pink Floyd (generous ceiling)",
    },
    "metal": {
        "void_dissolution_proportion": 0.06,
        "source": "Broad metal lyrics corpora",
    },
    "doom_metal": {
        "void_dissolution_proportion": 0.08,
        "source": "Funeral doom, sludge (My Dying Bride, Shape of Despair)",
    },
    "dark_ambient": {
        "void_dissolution_proportion": 0.10,
        "source": "Lustmord, Atrium Carceri, Cryo Chamber catalog",
    },
}


def compare_vs_genre_baselines(song: dict) -> dict:
    """Compare a single song's void density against genre baselines."""
    vd = song.get("semantic_clusters", {}).get("void_dissolution", {})
    prop = vd.get("proportion", 0)
    n = song.get("tokens", {}).get("total", 0)

    results = {}
    for genre, baseline in GENRE_BASELINES.items():
        bp = baseline["void_dissolution_proportion"]
        ztest = z_test_proportion(prop, bp, n) if n > 0 else {"z": 0, "p": 1}
        h = cohens_h(prop, bp)
        results[genre] = {
            "baseline": bp,
            "observed": round(prop, 4),
            "ratio": round(prop / bp, 2) if bp > 0 else 0,
            "z": ztest["z"],
            "p": ztest["p"],
            "cohens_h": h,
            "significant": ztest["p"] < 0.05 and ztest["z"] > 0,
            "source": baseline["source"],
        }

    return results


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline & Comparison Framework for AI Lyrics Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("corpus_dir", help="Directory containing analysis JSON files from collect.py")
    parser.add_argument("--report", action="store_true", help="Generate baseline report")
    parser.add_argument("--compare", help="Compare a specific JSON file against corpus")
    parser.add_argument("--cross-platform", action="store_true", help="Cross-platform comparison")
    parser.add_argument("--outliers", action="store_true", help="Detect outlier songs")
    parser.add_argument("--genre-baselines", help="Compare a JSON file vs genre baselines")
    parser.add_argument("--full", action="store_true", help="Full report (all analyses)")
    parser.add_argument("--export-baselines", help="Export computed baselines to JSON file")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--z-threshold", type=float, default=2.0, help="Z-score threshold for outliers")

    args = parser.parse_args()

    # Load corpus
    corpus = CorpusLoader(args.corpus_dir)
    if corpus.count == 0:
        print(f"Error: No valid analysis JSON files found in {args.corpus_dir}", file=sys.stderr)
        print("Run collect.py first to generate analysis data.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {corpus.count} songs from {args.corpus_dir}", file=sys.stderr)

    # Compute baselines
    computer = BaselineComputer(corpus)
    baselines = computer.compute_all()

    # Export baselines
    if args.export_baselines:
        export_path = Path(args.export_baselines)
        export_path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
        print(f"Exported baselines to {export_path}", file=sys.stderr)

    # Compare specific song
    comparisons = None
    if args.compare or args.full:
        engine = ComparisonEngine(baselines, corpus)
        if args.compare:
            song_data = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            comparisons = [engine.compare_song(song_data)]
        elif args.full:
            comparisons = engine.compare_all()

    # Cross-platform
    cross_platform = None
    if args.cross_platform or args.full:
        cpa = CrossPlatformAnalyzer(corpus)
        cross_platform = cpa.analyze()

    # Outliers
    outliers = None
    if args.outliers or args.full:
        detector = OutlierDetector(corpus, baselines)
        outliers = detector.detect(z_threshold=args.z_threshold)

    # Genre baselines for specific file
    if args.genre_baselines:
        song_data = json.loads(Path(args.genre_baselines).read_text(encoding="utf-8"))
        genre_results = compare_vs_genre_baselines(song_data)
        print(json.dumps(genre_results, indent=2))
        return

    # Report
    if args.report or args.full or args.compare or args.cross_platform or args.outliers:
        if args.format == "markdown":
            print(format_report_markdown(baselines, comparisons, cross_platform, outliers))
        else:
            print(format_report_json(baselines, comparisons, cross_platform, outliers))
    else:
        # Default: just print baselines
        print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
