"""
train_model.py — Parse OpenAI agent response JSONs, extract features using
hard-coded question labels, and train per-platform Ridge regression models
for the MCP Cost Estimator.

Usage:
    python train_model.py

Output:
    backend/models/{platform}_input_tokens_model.joblib
    backend/models/{platform}_output_tokens_model.joblib
    charts/01_token_distributions.png
    charts/02_platform_comparison.png
    charts/03_token_components.png
    charts/04_categorical_distributions.png
    charts/05_correlation_heatmap.png
    charts/06_top_predictors_scatter.png
    charts/07_complexity_breakdown.png

Features (pre-execution, available at inference time — must match inference.py):
    question_length  — character count of the user question
    question_word_count — word count of the user question
    domain_id        — 0=banking, 1=supply_chain, 2=healthcare, 3=general
    category_enc     — 0=Direct, 1=Generic
    complexity_enc   — 0=Easy, 1=Medium, 2=Hard
    result_size_enc  — 0=small, 1=medium, 2=large
    intent           — categorical (OHE): lookup, aggregate, list, comparison, trend, anomaly_detection
    answer_type      — categorical (OHE): single_number, list, chart, table

Model: Ridge regression on log1p(token_count), back-transformed with expm1 at inference.
"""

import json
import logging
import math
import os
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "backend" / "models"
CHARTS_DIR = ROOT / "charts"

# ---------------------------------------------------------------------------
# Pricing constants (for cost annotation in EDA)
# ---------------------------------------------------------------------------

INPUT_PRICE  = 2.50  / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000

# ---------------------------------------------------------------------------
# Feature schema (must match inference.py)
# ---------------------------------------------------------------------------

NUMERIC_FEATS   = [
    "question_length", "question_word_count", "domain_id",
    "category_enc", "complexity_enc", "result_size_enc",
]
CATEGORIC_FEATS = ["intent", "answer_type"]
FEATURE_NAMES   = NUMERIC_FEATS + CATEGORIC_FEATS

DOMAIN_MAP      = {"banking": 0, "supply_chain": 1, "healthcare": 2, "general": 3}
CATEGORY_ENC    = {"Direct": 0, "Generic": 1}
COMPLEXITY_ENC  = {"Easy": 0, "Medium": 1, "Hard": 2}
RESULT_SIZE_ENC = {"small": 0, "medium": 1, "large": 2}

# ---------------------------------------------------------------------------
# Hard-coded question labels per dataset (intent, result_size, answer_type,
# complexity, category) — derived from the MCP Cost Estimation notebook.
# ---------------------------------------------------------------------------

UNITUS_LABELS: dict[int, dict] = {
    1:  {"intent": "comparison",        "result_size": "small",  "answer_type": "chart",         "complexity": "Medium", "category": "Direct"},
    2:  {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    3:  {"intent": "list",              "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    4:  {"intent": "trend",             "result_size": "small",  "answer_type": "chart",         "complexity": "Medium", "category": "Direct"},
    5:  {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    6:  {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    7:  {"intent": "aggregate",         "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    8:  {"intent": "anomaly_detection", "result_size": "medium", "answer_type": "list",          "complexity": "Hard",   "category": "Generic"},
    9:  {"intent": "aggregate",         "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    10: {"intent": "trend",             "result_size": "medium", "answer_type": "chart",         "complexity": "Medium", "category": "Direct"},
    11: {"intent": "lookup",            "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    12: {"intent": "lookup",            "result_size": "medium", "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    13: {"intent": "lookup",            "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    14: {"intent": "aggregate",         "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    15: {"intent": "trend",             "result_size": "small",  "answer_type": "chart",         "complexity": "Medium", "category": "Direct"},
    16: {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    17: {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    18: {"intent": "lookup",            "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    19: {"intent": "aggregate",         "result_size": "small",  "answer_type": "single_number", "complexity": "Medium", "category": "Direct"},
    20: {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    21: {"intent": "aggregate",         "result_size": "small",  "answer_type": "single_number", "complexity": "Medium", "category": "Generic"},
    22: {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    23: {"intent": "aggregate",         "result_size": "small",  "answer_type": "single_number", "complexity": "Medium", "category": "Direct"},
    24: {"intent": "list",              "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    25: {"intent": "lookup",            "result_size": "medium", "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    26: {"intent": "comparison",        "result_size": "small",  "answer_type": "single_number", "complexity": "Medium", "category": "Generic"},
    27: {"intent": "lookup",            "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
}

UMCU_LABELS: dict[int, dict] = {
    1:  {"intent": "aggregate",  "result_size": "medium", "answer_type": "table",         "complexity": "Hard",   "category": "Generic"},
    2:  {"intent": "aggregate",  "result_size": "large",  "answer_type": "table",         "complexity": "Hard",   "category": "Generic"},
    3:  {"intent": "list",       "result_size": "medium", "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    4:  {"intent": "list",       "result_size": "medium", "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    5:  {"intent": "aggregate",  "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    6:  {"intent": "list",       "result_size": "medium", "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    7:  {"intent": "lookup",     "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Generic"},
    8:  {"intent": "aggregate",  "result_size": "medium", "answer_type": "table",         "complexity": "Hard",   "category": "Direct"},
    9:  {"intent": "aggregate",  "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    10: {"intent": "comparison", "result_size": "small",  "answer_type": "table",         "complexity": "Medium", "category": "Direct"},
    11: {"intent": "aggregate",  "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
}

TPCH_LABELS: dict[int, dict] = {
    1:  {"intent": "trend",      "result_size": "small",  "answer_type": "chart",         "complexity": "Medium", "category": "Direct"},
    2:  {"intent": "lookup",     "result_size": "small",  "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    3:  {"intent": "comparison", "result_size": "medium", "answer_type": "table",         "complexity": "Medium", "category": "Direct"},
    4:  {"intent": "lookup",     "result_size": "small",  "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    5:  {"intent": "aggregate",  "result_size": "medium", "answer_type": "table",         "complexity": "Medium", "category": "Direct"},
    6:  {"intent": "lookup",     "result_size": "small",  "answer_type": "list",          "complexity": "Easy",   "category": "Direct"},
    7:  {"intent": "aggregate",  "result_size": "small",  "answer_type": "single_number", "complexity": "Hard",   "category": "Generic"},
    8:  {"intent": "comparison", "result_size": "medium", "answer_type": "table",         "complexity": "Hard",   "category": "Direct"},
    9:  {"intent": "aggregate",  "result_size": "medium", "answer_type": "table",         "complexity": "Hard",   "category": "Generic"},
    10: {"intent": "lookup",     "result_size": "small",  "answer_type": "single_number", "complexity": "Easy",   "category": "Direct"},
    11: {"intent": "lookup",     "result_size": "small",  "answer_type": "list",          "complexity": "Medium", "category": "Direct"},
    12: {"intent": "comparison", "result_size": "medium", "answer_type": "table",         "complexity": "Hard",   "category": "Generic"},
}

DATASET_LABELS: dict[str, dict[int, dict]] = {
    "unitus": UNITUS_LABELS,
    "umcu":   UMCU_LABELS,
    "tpch":   TPCH_LABELS,
}

# ---------------------------------------------------------------------------
# (server, dataset) → [(data_subdir, dataset_name), ...]
# One model per combo; file: {server}_{dataset}_{target}_model.joblib
# ---------------------------------------------------------------------------

COMBO_DATA: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("sql_server", "unitus"): [("sqlserver/unitus",        "unitus")],
    ("supabase",   "unitus"): [("supabase/unitus",         "unitus")],
    ("tursio",     "unitus"): [("tursio/unitus-sqlserver", "unitus"),
                               ("tursio/unitus-supabase",  "unitus")],
    ("snowflake",  "umcu"):   [("snowflake/umcu",          "umcu")],
    ("supabase",   "umcu"):   [("supabase/umcu",           "umcu")],
    ("tursio",     "umcu"):   [("tursio/umcu-snowflake",   "umcu"),
                               ("tursio/umcu-supabase",    "umcu")],
    ("motherduck", "tpch"):   [("motherduck/tpch",         "tpch")],
    ("supabase",   "tpch"):   [("supabase/tpch",           "tpch")],
    ("tursio",     "tpch"):   [("tursio/tpch-snowflake",   "tpch"),
                               ("tursio/tpch-supabase",    "tpch")],
}

# Dataset → available servers (mirrors COMBO_DATA; shared with inference.py / frontend)
DATASET_SERVERS: dict[str, list[str]] = {
    "unitus": ["sql_server", "tursio", "supabase"],
    "umcu":   ["snowflake",  "tursio", "supabase"],
    "tpch":   ["motherduck", "tursio", "supabase"],
}

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

PLATFORM_COLORS = {
    "snowflake":  "#29B5E8",
    "sql_server": "#E74C3C",
    "tursio":     "#9B59B6",
    "motherduck": "#F5B942",
    "supabase":   "#3ECF8E",
}
PLATFORM_LABELS = {
    "snowflake":  "Snowflake",
    "sql_server": "SQL Server",
    "tursio":     "Tursio",
    "motherduck": "MotherDuck",
    "supabase":   "Supabase",
}
TOKEN_COLORS = {
    "input_tokens":     "#3498DB",
    "output_tokens":    "#E67E22",
    "reasoning_tokens": "#2ECC71",
    "cached_tokens":    "#95A5A6",
}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})


def _combo_color(key: str) -> str:
    """Return the platform color for a combo key like 'sql_server+unitus'."""
    return PLATFORM_COLORS.get(key.split("+")[0], "#888")


def _combo_label(key: str) -> str:
    """Return a human-readable label for a combo key like 'sql_server+unitus'."""
    server, dataset = key.split("+", 1)
    return f"{PLATFORM_LABELS.get(server, server)} / {dataset.upper()}"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def extract_usage(data: dict) -> dict:
    usage = data.get("usage", {})
    details_in  = usage.get("input_tokens_details")  or {}
    details_out = usage.get("output_tokens_details") or {}
    return {
        "input_tokens":     usage.get("input_tokens",  0) or 0,
        "output_tokens":    usage.get("output_tokens", 0) or 0,
        "total_tokens":     usage.get("total_tokens",  0) or 0,
        "cached_tokens":    details_in.get("cached_tokens",       0) or 0,
        "reasoning_tokens": details_out.get("reasoning_tokens",   0) or 0,
    }


def extract_response_features(data: dict) -> dict:
    """Extract behavioural features from the agent output (used for EDA)."""
    output = data.get("output", [])
    num_mcp_calls      = 0
    num_reasoning      = 0
    num_errors         = 0
    total_sql_length   = 0
    total_result_length = 0

    for item in output:
        t = item.get("type", "")
        if t == "mcp_call":
            num_mcp_calls += 1
            if item.get("error"):
                num_errors += 1
            args_raw = item.get("arguments", "")
            out_raw  = str(item.get("output", ""))
            name     = item.get("name", "")
            if name in ("banking_sql_execution", "read_data"):
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    sql  = args.get("sql") or args.get("query", "")
                    total_sql_length += len(sql)
                except Exception:
                    pass
            total_result_length += len(out_raw)
        elif t == "reasoning":
            num_reasoning += 1

    return {
        "num_mcp_calls":       num_mcp_calls,
        "num_reasoning_steps": num_reasoning,
        "num_errors":          num_errors,
        "total_sql_length":    total_sql_length,
        "total_result_length": total_result_length,
    }


# ---------------------------------------------------------------------------
# Question text extraction and domain classification
# ---------------------------------------------------------------------------

def extract_question_text(data: dict) -> str | None:
    """
    Try three strategies to recover the user question from a response JSON:
    1. Top-level Question_Text / question_text field (sql_server files)
    2. tursio_search / search call arguments (tursio files)
    3. First reasoning summary text (snowflake/motherduck files)
    """
    q = data.get("Question_Text") or data.get("question_text")
    if q:
        return str(q).strip()

    for item in data.get("output", []):
        if item.get("type") == "mcp_call" and item.get("name") in ("tursio_search", "search"):
            try:
                args = item.get("arguments", "{}")
                args = json.loads(args) if isinstance(args, str) else args
                q = args.get("query", "")
                if q:
                    return str(q).strip()
            except Exception:
                pass

    for item in data.get("output", []):
        if item.get("type") == "reasoning":
            for s in item.get("summary", []):
                text = s.get("text", "").strip()
                if text and len(text) > 30:
                    return text[:150].strip()

    return None


def classify_domain(text: str) -> str:
    q = text.lower()
    if any(w in q for w in [
        "money", "revenue", "sales", "bank", "finance", "payment",
        "loan", "account", "card", "deposit", "dormant", "member",
        "balance", "transaction", "login", "credit",
    ]):
        return "banking"
    if any(w in q for w in [
        "supply", "inventory", "order", "shipment", "logistics",
        "warehouse", "supplier", "customer", "part", "nation",
        "region", "lineitem",
    ]):
        return "supply_chain"
    if any(w in q for w in [
        "patient", "health", "doctor", "clinical", "medical",
        "treatment", "diagnosis",
    ]):
        return "healthcare"
    return "general"


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_combo_data(server: str, dataset: str) -> list[dict]:
    """
    Load all JSON files for a (server, dataset) combo, merge with hard-coded
    question labels, collapse multiple runs to median token counts.
    """
    subdir_list = COMBO_DATA.get((server, dataset), [])
    if not subdir_list:
        logger.warning("No data paths configured for (%s, %s)", server, dataset)
        return []

    # Collect raw rows keyed by (dataset, query_id) → list of run results
    raw: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for subdir, dataset_name in subdir_list:
        base = DATA_DIR / subdir
        if not base.exists():
            logger.warning("Directory not found: %s", base)
            continue

        labels = DATASET_LABELS.get(dataset_name, {})

        for json_path in sorted(base.rglob("*.json")):
            stem = json_path.stem          # e.g. "q1"
            if not stem.startswith("q"):
                continue
            try:
                query_id = int(stem[1:])
            except ValueError:
                continue

            label = labels.get(query_id)
            if label is None:
                logger.debug("No label for %s query_id=%d — skipping", dataset_name, query_id)
                continue

            data = load_json(json_path)
            if data is None:
                continue
            if data.get("status") != "completed":
                continue

            usage = extract_usage(data)
            if usage["input_tokens"] == 0:
                continue

            resp_feats    = extract_response_features(data)
            question_text = extract_question_text(data) or ""

            raw[(dataset_name, query_id)].append({
                "usage":         usage,
                "resp_feats":    resp_feats,
                "label":         label,
                "question_text": question_text,
            })

    if not raw:
        logger.info("No records loaded for platform '%s'", platform)
        return []

    # Collapse multiple runs → median token counts
    records = []
    for (dataset_name, query_id), runs in sorted(raw.items()):
        label = runs[0]["label"]

        # Use the longest available question text across runs as the best proxy
        question_text = max(
            (r["question_text"] for r in runs), key=len, default=""
        )
        domain = classify_domain(question_text) if question_text else "general"

        def med(key: str) -> float:
            return float(np.median([r["usage"][key] for r in runs]))

        def med_resp(key: str) -> float:
            return float(np.median([r["resp_feats"][key] for r in runs]))

        in_tok  = med("input_tokens")
        out_tok = med("output_tokens")

        record = {
            "platform":   server,
            "dataset":    dataset,
            "query_id":   query_id,
            # Question-text features
            "question_length":     len(question_text),
            "question_word_count": len(question_text.split()),
            "domain":              domain,
            "domain_id":           DOMAIN_MAP.get(domain, 3),
            # Label-derived raw strings (for EDA)
            "category":    label["category"],
            "complexity":  label["complexity"],
            "result_size": label["result_size"],
            "intent":      label["intent"],
            "answer_type": label["answer_type"],
            # Encoded numerics
            "category_enc":    CATEGORY_ENC[label["category"]],
            "complexity_enc":  COMPLEXITY_ENC[label["complexity"]],
            "result_size_enc": RESULT_SIZE_ENC[label["result_size"]],
            # Token targets
            "input_tokens":     in_tok,
            "output_tokens":    out_tok,
            "total_tokens":     med("total_tokens"),
            "cached_tokens":    med("cached_tokens"),
            "reasoning_tokens": med("reasoning_tokens"),
            # Cost annotation
            "cost_usd": in_tok * INPUT_PRICE + out_tok * OUTPUT_PRICE,
            # Behavioural features (EDA only)
            "num_mcp_calls":       med_resp("num_mcp_calls"),
            "num_reasoning_steps": med_resp("num_reasoning_steps"),
            "num_errors":          med_resp("num_errors"),
            "total_sql_length":    med_resp("total_sql_length"),
            "total_result_length": med_resp("total_result_length"),
        }
        records.append(record)

    logger.info("Loaded %d records for (%s, %s)", len(records), server, dataset)
    return records


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def _stats(values: list) -> str:
    if not values:
        return "N/A"
    mn, mx  = min(values), max(values)
    mean    = statistics.mean(values)
    med     = statistics.median(values)
    std     = statistics.stdev(values) if len(values) > 1 else 0
    return f"mean={mean:.0f}  median={med:.0f}  std={std:.0f}  min={mn:.0f}  max={mx:.0f}"


def _correlation(x: list, y: list) -> float:
    if len(x) < 2:
        return float("nan")
    n    = len(x)
    mx   = sum(x) / n
    my   = sum(y) / n
    num  = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den  = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    return num / den if den else float("nan")


def run_eda(records: list[dict], combo_label: str) -> None:
    logger.info("=" * 60)
    logger.info("EDA — %s  (%d samples)", _combo_label(combo_label), len(records))
    logger.info("=" * 60)

    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
        vals = [r[key] for r in records]
        logger.info("  %-22s %s", key + ":", _stats(vals))

    for cat in ("category", "complexity", "result_size", "intent", "answer_type"):
        counts: dict[str, int] = {}
        for r in records:
            counts[r[cat]] = counts.get(r[cat], 0) + 1
        logger.info("  %s: %s", cat, "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    logger.info("  --- Correlations with input_tokens ---")
    for feat in (
        "question_length", "question_word_count", "domain_id",
        "category_enc", "complexity_enc", "result_size_enc",
        "num_mcp_calls", "num_reasoning_steps", "total_sql_length", "total_result_length",
    ):
        x = [r[feat] for r in records]
        y = [r["input_tokens"] for r in records]
        logger.info("    %-30s r=%.3f", feat, _correlation(x, y))

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart → %s", path)


def _fmt_k(x, _=None) -> str:
    return f"{x/1000:.0f}k" if x >= 1000 else str(int(x))


def plot_token_distributions(all_records: dict[str, list[dict]]) -> None:
    platforms = list(all_records.keys())
    targets   = ["input_tokens", "output_tokens", "total_tokens"]
    titles    = ["Input Tokens",  "Output Tokens",  "Total Tokens"]

    fig, axes = plt.subplots(len(platforms), len(targets), figsize=(14, 4 * len(platforms)))
    if len(platforms) == 1:
        axes = [axes]
    fig.suptitle("Token Count Distributions by Platform", fontsize=14, fontweight="bold", y=1.01)

    for row, platform in enumerate(platforms):
        records = all_records[platform]
        color   = _combo_color(platform)
        label   = _combo_label(platform)

        for col, (target, title) in enumerate(zip(targets, titles)):
            ax   = axes[row][col]
            vals = [r[target] for r in records]
            if not vals or max(vals) == 0:
                ax.set_visible(False)
                continue

            log_min = math.floor(math.log10(max(min(vals), 1)))
            log_max = math.ceil(math.log10(max(vals)))
            bins    = np.logspace(log_min, log_max, 20)

            ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor="white")
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
            ax.set_xlabel("Tokens (log scale)")
            ax.set_ylabel("Count")

            mean_v = statistics.mean(vals)
            med_v  = statistics.median(vals)
            ax.axvline(mean_v, color="black",   linestyle="--", linewidth=1.2, label=f"mean={_fmt_k(mean_v)}")
            ax.axvline(med_v,  color="dimgray", linestyle=":",  linewidth=1.2, label=f"median={_fmt_k(med_v)}")
            ax.legend(fontsize=8)

            if row == 0:
                ax.set_title(title, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\nCount", fontweight="bold")

    plt.tight_layout()
    _save(fig, "01_token_distributions.png")


def plot_platform_comparison(all_records: dict[str, list[dict]]) -> None:
    targets   = ["input_tokens", "output_tokens", "total_tokens"]
    titles    = ["Input Tokens",  "Output Tokens",  "Total Tokens"]
    platforms = list(all_records.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Token Count Distribution — Cross-Platform Comparison", fontsize=14, fontweight="bold")

    for ax, target, title in zip(axes, targets, titles):
        data = [[r[target] for r in all_records[p]] for p in platforms]
        bp   = ax.boxplot(
            data,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        for patch, platform in zip(bp["boxes"], platforms):
            patch.set_facecolor(_combo_color(platform))
            patch.set_alpha(0.7)

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
        ax.set_xticks(range(1, len(platforms) + 1))
        ax.set_xticklabels([_combo_label(p) for p in platforms], rotation=15)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Tokens (log scale)" if target == "input_tokens" else "")

    plt.tight_layout()
    _save(fig, "02_platform_comparison.png")


def plot_token_components(all_records: dict[str, list[dict]]) -> None:
    components = [
        ("input_tokens",     "Input"),
        ("output_tokens",    "Output"),
        ("reasoning_tokens", "Reasoning"),
        ("cached_tokens",    "Cached"),
    ]
    platforms = list(all_records.keys())
    x     = np.arange(len(platforms))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Mean Token Components per Platform", fontsize=14, fontweight="bold")

    for i, (key, label) in enumerate(components):
        means  = [statistics.mean(r[key] for r in all_records[p]) for p in platforms]
        offset = (i - len(components) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width, label=label,
                        color=TOKEN_COLORS[key], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, means):
            if val > 200:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                        _fmt_k(val), ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([_combo_label(p) for p in platforms])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_ylabel("Mean Token Count")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    _save(fig, "03_token_components.png")


def plot_categorical_distributions(all_records: dict[str, list[dict]]) -> None:
    cats = [
        ("category",    ["Direct", "Generic"]),
        ("complexity",  ["Easy", "Medium", "Hard"]),
        ("intent",      ["lookup", "aggregate", "list", "comparison", "trend", "anomaly_detection"]),
        ("answer_type", ["single_number", "list", "chart", "table"]),
    ]
    platforms = list(all_records.keys())

    fig, axes = plt.subplots(
        len(platforms), len(cats), figsize=(16, 4 * len(platforms))
    )
    if len(platforms) == 1:
        axes = [axes]
    fig.suptitle("Categorical Feature Distributions by Platform", fontsize=14, fontweight="bold", y=1.01)

    for row, platform in enumerate(platforms):
        records = all_records[platform]
        color   = _combo_color(platform)

        for col, (cat, ordered_vals) in enumerate(cats):
            ax     = axes[row][col]
            counts = {}
            for r in records:
                v         = r[cat]
                counts[v] = counts.get(v, 0) + 1

            labels = [v for v in ordered_vals if counts.get(v, 0) > 0]
            vals   = [counts.get(v, 0) for v in labels]

            ax.barh(labels, vals, color=color, alpha=0.8, edgecolor="white")
            for v, val in zip(labels, vals):
                ax.text(val + 0.1, v, str(val), va="center", fontsize=9)
            ax.set_xlim(0, max(vals) * 1.3 if vals else 1)

            if row == 0:
                ax.set_title(cat.replace("_", " ").capitalize(), fontweight="bold")
            if col == 0:
                ax.set_ylabel(_combo_label(platform), fontweight="bold")
            ax.set_xlabel("Count")

    plt.tight_layout()
    _save(fig, "04_categorical_distributions.png")


def plot_correlation_heatmap(all_records: dict[str, list[dict]]) -> None:
    feature_keys = [
        "question_length", "question_word_count", "domain_id",
        "category_enc", "complexity_enc", "result_size_enc",
        "num_mcp_calls", "num_reasoning_steps",
        "total_sql_length", "total_result_length",
        "reasoning_tokens", "cached_tokens",
    ]
    feature_labels = [
        "question_length", "question_words", "domain_id",
        "category_enc", "complexity_enc", "result_size_enc",
        "num_mcp_calls", "num_reasoning",
        "sql_length", "result_length",
        "reasoning_tokens", "cached_tokens",
    ]
    target_keys = ["input_tokens", "output_tokens"]
    platforms   = list(all_records.keys())

    fig, axes = plt.subplots(1, len(platforms), figsize=(5 * len(platforms), 6))
    if len(platforms) == 1:
        axes = [axes]
    fig.suptitle("Feature Correlations with Token Counts", fontsize=14, fontweight="bold")

    for ax, platform in zip(axes, platforms):
        records = all_records[platform]
        matrix  = []
        for fk in feature_keys:
            row_corrs = []
            for tk in target_keys:
                x = [r[fk] for r in records]
                y = [r[tk] for r in records]
                row_corrs.append(_correlation(x, y))
            matrix.append(row_corrs)

        mat         = np.array(matrix, dtype=float)
        mat_display = np.where(np.isnan(mat), 0.0, mat)

        sns.heatmap(
            mat_display, ax=ax,
            xticklabels=["input_tokens", "output_tokens"],
            yticklabels=feature_labels,
            vmin=-1, vmax=1, center=0,
            cmap="RdBu_r", annot=True, fmt=".2f",
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 8},
            cbar=(platform == platforms[-1]),
        )
        ax.set_title(_combo_label(platform), fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png")


def plot_top_predictors_scatter(all_records: dict[str, list[dict]]) -> None:
    candidate_features = [
        "question_length", "question_word_count", "domain_id",
        "category_enc", "complexity_enc", "result_size_enc",
        "num_mcp_calls", "num_reasoning_steps",
        "total_sql_length", "total_result_length",
    ]
    targets   = ["input_tokens", "output_tokens"]
    platforms = list(all_records.keys())

    fig, axes = plt.subplots(
        len(platforms), len(targets), figsize=(10, 4.5 * len(platforms))
    )
    if len(platforms) == 1:
        axes = [axes]
    fig.suptitle("Top Predictor vs Token Count (per Platform)", fontsize=14, fontweight="bold", y=1.01)

    for row, platform in enumerate(platforms):
        records = all_records[platform]
        color   = _combo_color(platform)

        for col, target in enumerate(targets):
            ax = axes[row][col]

            best_feat, best_r = candidate_features[0], 0.0
            for fk in candidate_features:
                x_vals = [r[fk] for r in records]
                y_vals = [r[target] for r in records]
                r_val  = _correlation(x_vals, y_vals)
                if not math.isnan(r_val) and abs(r_val) > abs(best_r):
                    best_feat, best_r = fk, r_val

            x = np.array([r[best_feat] for r in records], dtype=float)
            y = np.array([r[target]    for r in records], dtype=float)

            ax.scatter(x, y, color=color, alpha=0.65, edgecolors="white", linewidths=0.5, s=60)

            if len(x) > 2 and not math.isnan(best_r):
                m, b   = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

            ax.set_xlabel(best_feat.replace("_", " "), fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
            label = "input tokens" if target == "input_tokens" else "output tokens"
            ax.set_title(
                f"{_combo_label(platform)} — {label}\n(r = {best_r:.2f})",
                fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    _save(fig, "06_top_predictors_scatter.png")


def plot_complexity_breakdown(all_records: dict[str, list[dict]]) -> None:
    platforms         = list(all_records.keys())
    complexity_levels = ["Easy", "Medium", "Hard"]

    fig, axes = plt.subplots(len(platforms), 2, figsize=(12, 4.5 * len(platforms)))
    if len(platforms) == 1:
        axes = [axes]
    fig.suptitle("Token Counts by Complexity Level", fontsize=14, fontweight="bold", y=1.01)

    for row, platform in enumerate(platforms):
        records = all_records[platform]
        color   = _combo_color(platform)

        for col, target in enumerate(["input_tokens", "output_tokens"]):
            ax     = axes[row][col]
            groups = [
                [r[target] for r in records if r["complexity"] == lvl]
                for lvl in complexity_levels
            ]
            non_empty = [(lvl, g) for lvl, g in zip(complexity_levels, groups) if g]
            if not non_empty:
                ax.set_visible(False)
                continue

            lvls, grps = zip(*non_empty)
            bp = ax.boxplot(
                grps,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.65)

            ax.set_xticks(range(1, len(lvls) + 1))
            ax.set_xticklabels(lvls)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
            ax.set_xlabel("Complexity")
            label = "Input Tokens" if target == "input_tokens" else "Output Tokens"
            ax.set_title(
                f"{_combo_label(platform)} — {label}",
                fontsize=9, fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel("Tokens")

    plt.tight_layout()
    _save(fig, "07_complexity_breakdown.png")


def plot_eda(all_records: dict[str, list[dict]]) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating EDA charts → %s", CHARTS_DIR)
    plot_token_distributions(all_records)
    plot_platform_comparison(all_records)
    plot_token_components(all_records)
    plot_categorical_distributions(all_records)
    plot_correlation_heatmap(all_records)
    plot_top_predictors_scatter(all_records)
    plot_complexity_breakdown(all_records)
    logger.info("All charts saved.")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Ridge regression on log1p(tokens) with:
    - StandardScaler on numeric features
    - OneHotEncoder (ignore unknown) on categorical features
    """
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                       NUMERIC_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORIC_FEATS),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", Ridge())])


def train_and_evaluate(records: list[dict], target: str, combo_label: str) -> Pipeline:
    """
    Tune Ridge alpha via GridSearchCV, evaluate with stratified CV,
    and refit on all data. Returns the fitted pipeline.
    """
    df = pd.DataFrame(records)
    X  = df[FEATURE_NAMES]
    y_log = np.log1p(df[target].values.astype(float))
    y_raw = df[target].values.astype(float)
    n     = len(records)

    n_splits = min(5, max(2, n // 3))
    alpha_grid = [0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]

    pipeline = build_pipeline()

    if n >= n_splits * 2:
        strat_labels = (y_raw > np.median(y_raw)).astype(int)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline,
            {"model__alpha": alpha_grid},
            cv=cv.split(X, strat_labels),
            scoring="neg_mean_absolute_error",
            refit=True,
        )
        grid.fit(X, y_log)
        best_alpha = grid.best_params_["model__alpha"]
        best_model = grid.best_estimator_

        # Report CV MAE in raw token space
        fold_maes = []
        for train_idx, test_idx in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, strat_labels):
            best_model.fit(X.iloc[train_idx], y_log[train_idx])
            y_pred = np.expm1(best_model.predict(X.iloc[test_idx])).clip(0)
            fold_maes.append(mean_absolute_error(y_raw[test_idx], y_pred))

        mae_mean = np.mean(fold_maes)
        logger.info(
            "  [%s] %-15s  n=%d  alpha=%s  CV-MAE=%.0f",
            combo_label, target, n, best_alpha, mae_mean,
        )
    else:
        best_alpha = 1.0
        best_model = pipeline
        logger.warning(
            "  [%s] %-15s  n=%d  too few for CV — using alpha=1.0", combo_label, target, n
        )

    # Refit on all data
    best_model.fit(X, y_log)
    return best_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory  : %s", DATA_DIR)
    logger.info("Models directory: %s", MODELS_DIR)
    logger.info("Charts directory: %s", CHARTS_DIR)

    all_records: dict[str, list[dict]] = {}

    for (server, dataset) in COMBO_DATA:
        combo_label = f"{server}+{dataset}"
        logger.info("")
        records = load_combo_data(server, dataset)

        if len(records) < 3:
            logger.warning(
                "Only %d records for (%s, %s) — skipping (need ≥ 3)", len(records), server, dataset
            )
            continue

        all_records[combo_label] = records
        run_eda(records, combo_label)

        logger.info("Training models for (%s, %s)...", server, dataset)
        for target in ("input_tokens", "output_tokens"):
            pipeline = train_and_evaluate(records, target, combo_label)
            out_path = MODELS_DIR / f"{server}_{dataset}_{target}_model.joblib"
            joblib.dump(pipeline, out_path)
            logger.info("  Saved → %s", out_path)

    logger.info("")
    if all_records:
        plot_eda(all_records)

    logger.info("")
    logger.info("Done. Run `uvicorn backend.main:app` to start the API.")


if __name__ == "__main__":
    main()
