"""
inference.py — Feature extraction, question classification, and per-platform
token/cost prediction for the MCP Cost Estimator.

Each platform has two scikit-learn Pipeline models:
    {platform}_input_tokens_model.joblib
    {platform}_output_tokens_model.joblib

Features (must match train_model.py FEATURE_NAMES):
    question_length  — character count of the user question
    question_word_count — word count of the user question
    domain_id        — 0=banking, 1=supply_chain, 2=healthcare, 3=general
    category_enc     — 0=Direct, 1=Generic
    complexity_enc   — 0=Easy, 1=Medium, 2=Hard
    result_size_enc  — 0=small, 1=medium, 2=large
    intent           — categorical: lookup, aggregate, list, comparison, trend, anomaly_detection
    answer_type      — categorical: single_number, list, chart, table

Models expect a pandas DataFrame as input (required by ColumnTransformer).
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing: separate input / output rates (per token)
# ---------------------------------------------------------------------------

INPUT_PRICING: dict[str, float] = {
    "gpt-5.4":       2.50  / 1_000_000,
    "gpt-4o":        2.50  / 1_000_000,
    "gpt-4-turbo":   10.00 / 1_000_000,
    "gpt-3.5-turbo": 0.50  / 1_000_000,
}
OUTPUT_PRICING: dict[str, float] = {
    "gpt-5.4":       15.00 / 1_000_000,
    "gpt-4o":        10.00 / 1_000_000,
    "gpt-4-turbo":   30.00 / 1_000_000,
    "gpt-3.5-turbo":  1.50 / 1_000_000,
}

# ---------------------------------------------------------------------------
# Feature encodings (must match train_model.py)
# ---------------------------------------------------------------------------

DOMAIN_MAP: dict[str, int] = {
    "banking": 0, "supply_chain": 1, "healthcare": 2, "general": 3,
}

CATEGORY_MAP:    dict[str, int] = {"Direct": 0,  "Generic": 1}
COMPLEXITY_MAP:  dict[str, int] = {"Easy": 0,    "Medium": 1, "Hard": 2}
RESULT_SIZE_MAP: dict[str, int] = {"small": 0,   "medium": 1, "large": 2}

NUMERIC_FEATS = [
    "question_length", "question_word_count", "domain_id",
    "category_enc", "complexity_enc", "result_size_enc",
]
CATEGORIC_FEATS = ["intent", "answer_type"]
FEATURE_NAMES   = NUMERIC_FEATS + CATEGORIC_FEATS

# Dataset → available servers (must mirror train_model.py COMBO_DATA)
DATASET_SERVERS: dict[str, list[str]] = {
    "unitus": ["sql_server", "tursio", "supabase"],
    "umcu":   ["snowflake",  "tursio", "supabase"],
    "tpch":   ["tursio", "supabase"],
}

# All (server, dataset) combos — model files: {server}_{dataset}_{target}_model.joblib
_COMBOS: list[tuple[str, str]] = [
    ("sql_server", "unitus"),
    ("supabase",   "unitus"),
    ("tursio",     "unitus"),
    ("snowflake",  "umcu"),
    ("supabase",   "umcu"),
    ("tursio",     "umcu"),
    ("supabase",   "tpch"),
    ("tursio",     "tpch"),
]

# ---------------------------------------------------------------------------
# Domain keyword classifier (always derived from the raw question text)
# ---------------------------------------------------------------------------

def _classify_domain(question: str) -> str:
    q = question.lower()
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
# LLM classification prompt
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = (
    "Classify this analytics question into the following categories.\n\n"
    "Question: {question}\n\n"
    "Respond with JSON only — no extra text:\n"
    '{{'
    '"category": "<Direct|Generic>", '
    '"complexity": "<Easy|Medium|Hard>", '
    '"result_size": "<small|medium|large>", '
    '"intent": "<lookup|aggregate|list|comparison|trend|anomaly_detection>", '
    '"answer_type": "<single_number|list|chart|table>"'
    '}}\n\n'
    "Guidelines:\n"
    "- category: Direct if the question maps directly to database fields/values; "
    "Generic if it requires broader reasoning, patterns, or domain knowledge\n"
    "- complexity: Easy for simple single-table lookups, Medium for filtered/conditional "
    "queries or basic aggregations, Hard for multi-step aggregations, joins, or comparisons\n"
    "- result_size: small (1–5 data points), medium (5–50), large (50+)\n"
    "- intent: lookup (retrieve a specific value), aggregate (count/sum/avg), "
    "list (enumerate records), comparison (side-by-side values), "
    "trend (change over time), anomaly_detection (unusual patterns/outliers)\n"
    "- answer_type: single_number (one KPI), list (rows of items), "
    "chart (time-series or bar chart), table (multi-column breakdown)"
)


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------


class CostEstimator:
    def __init__(self) -> None:
        self.models: dict[tuple[str, str], dict[str, object]] = {}
        self._openai_client: OpenAI | None = None
        self._load_models()
        self._init_openai()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        base_path = os.path.join(os.path.dirname(__file__), "models")
        for (server, dataset) in _COMBOS:
            combo_models: dict[str, object] = {}
            for target in ("input_tokens", "output_tokens"):
                model_path = os.path.join(
                    base_path, f"{server}_{dataset}_{target}_model.joblib"
                )
                if os.path.exists(model_path):
                    combo_models[target] = joblib.load(model_path)
                    logger.debug("Loaded %s model for (%s, %s)", target, server, dataset)
                else:
                    logger.warning("Model not found: %s", model_path)

            if combo_models:
                self.models[(server, dataset)] = combo_models

        complete = [k for k, m in self.models.items() if len(m) == 2]
        logger.info("Models loaded for combos: %s", complete)

    def _init_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self._openai_client = OpenAI(api_key=api_key)
            logger.debug("OpenAI client initialised")
        else:
            logger.warning(
                "OPENAI_API_KEY not set — feature inference will use keyword fallback"
            )

    # ------------------------------------------------------------------
    # Question classification  → (category, complexity, result_size, intent, answer_type)
    # ------------------------------------------------------------------

    def infer_features(self, question: str) -> tuple[str, str, str, str, str]:
        """
        Classify a question into (category, complexity, result_size, intent, answer_type).
        Tries OpenAI first; falls back to keyword matching.
        """
        if self._openai_client is not None:
            try:
                return self._infer_features_llm(question)
            except Exception as exc:
                logger.warning(
                    "LLM feature inference failed (%s) — falling back to keywords", exc
                )

        logger.debug("Using keyword fallback for feature inference")
        return self._infer_features_fallback(question)

    def _infer_features_llm(self, question: str) -> tuple[str, str, str, str, str]:
        response = self._openai_client.chat.completions.create(  # type: ignore[union-attr]
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(question=question)}],
        )
        raw = response.choices[0].message.content.strip()
        logger.debug("LLM raw response: %s", raw)

        if raw.startswith("```"):
            parts = raw.split("```", 2)
            raw   = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result: dict = json.loads(raw.strip())

        category    = result.get("category",    "Direct")
        complexity  = result.get("complexity",  "Medium")
        result_size = result.get("result_size", "small")
        intent      = result.get("intent",      "lookup")
        answer_type = result.get("answer_type", "single_number")

        # Validate and fall back to safe defaults
        category    = category    if category    in CATEGORY_MAP    else "Direct"
        complexity  = complexity  if complexity  in COMPLEXITY_MAP  else "Medium"
        result_size = result_size if result_size in RESULT_SIZE_MAP else "small"
        intent      = intent      if intent      in (
            "lookup", "aggregate", "list", "comparison", "trend", "anomaly_detection"
        ) else "lookup"
        answer_type = answer_type if answer_type in (
            "single_number", "list", "chart", "table"
        ) else "single_number"

        logger.info(
            "LLM features — category=%s complexity=%s result_size=%s intent=%s answer_type=%s",
            category, complexity, result_size, intent, answer_type,
        )
        return category, complexity, result_size, intent, answer_type

    def _infer_features_fallback(self, question: str) -> tuple[str, str, str, str, str]:
        q = question.lower()

        # category — Generic if requires reasoning beyond direct field lookup
        category = "Generic"
        if not any(w in q for w in [
            "why", "what if", "could", "should", "would", "is there",
            "pattern", "anomaly", "unusual", "risk", "suspect",
        ]):
            category = "Direct"

        # complexity
        complexity = "Easy"
        if any(w in q for w in [
            "join", "aggregate", "group by", "having", "subquery",
            "percentage", "ratio", "breakdown", "compare", "versus", "vs",
            "dormant", "inactive", "anomaly", "pattern", "outlier",
            "per month", "per year", "by month", "by year",
        ]):
            complexity = "Hard"
        elif any(w in q for w in [
            "where", "filter", "between", "greater", "less",
            "after", "before", "last", "recent", "top", "total",
            "sum", "average", "mean", "trend",
        ]):
            complexity = "Medium"

        # result_size
        result_size = "small"
        if any(w in q for w in [
            "breakdown", "distribution", "by month", "by year",
            "over time", "trend", "all", "list all", "every", "each",
        ]):
            result_size = "medium"
        elif any(w in q for w in ["full", "complete", "entire", "all records"]):
            result_size = "large"

        # intent
        intent = "lookup"
        if any(w in q for w in ["anomaly", "unusual", "outlier", "dormant", "inactive", "suspicious"]):
            intent = "anomaly_detection"
        elif any(w in q for w in ["trend", "over time", "by month", "by year", "growth", "change"]):
            intent = "trend"
        elif any(w in q for w in ["compare", "difference", "versus", "vs", "higher", "lower"]):
            intent = "comparison"
        elif any(w in q for w in ["list", "show all", "get all", "find all", "which"]):
            intent = "list"
        elif any(w in q for w in ["how many", "count", "total", "sum", "average", "mean", "what is the"]):
            intent = "aggregate"

        # answer_type
        if intent == "trend" or any(w in q for w in ["chart", "plot", "graph", "visualize"]):
            answer_type = "chart"
        elif any(w in q for w in ["table", "breakdown", "distribution", "by ", "per "]):
            answer_type = "table"
        elif intent in ("list", "anomaly_detection") or result_size != "small":
            answer_type = "list"
        else:
            answer_type = "single_number"

        logger.info(
            "Keyword features — category=%s complexity=%s result_size=%s intent=%s answer_type=%s",
            category, complexity, result_size, intent, answer_type,
        )
        return category, complexity, result_size, intent, answer_type

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate(self, question: str, gpt_model: str, dataset: str) -> dict:
        """
        Return predicted token counts and costs for all servers that have
        a trained model for the requested dataset.

        Response structure:
            {
              "inferred_features": {domain, category, complexity, result_size, intent, answer_type},
              "estimates": {
                "<server>": {input_tokens, output_tokens, total_tokens, cost_usd}
              }
            }
        """
        category, complexity, result_size, intent, answer_type = self.infer_features(question)
        domain    = _classify_domain(question)
        domain_id = DOMAIN_MAP.get(domain, 3)

        in_price  = INPUT_PRICING.get(gpt_model,  INPUT_PRICING["gpt-4o"])
        out_price = OUTPUT_PRICING.get(gpt_model, OUTPUT_PRICING["gpt-4o"])

        estimates: dict[str, dict] = {}
        for (server, ds), combo_models in self.models.items():
            if ds != dataset:
                continue

            input_model  = combo_models.get("input_tokens")
            output_model = combo_models.get("output_tokens")

            if input_model is None or output_model is None:
                logger.warning("Incomplete models for (%s, %s) — skipping", server, ds)
                continue

            x_df = pd.DataFrame([{
                "question_length":     len(question),
                "question_word_count": len(question.split()),
                "domain_id":           domain_id,
                "category_enc":        CATEGORY_MAP[category],
                "complexity_enc":      COMPLEXITY_MAP[complexity],
                "result_size_enc":     RESULT_SIZE_MAP[result_size],
                "intent":              intent,
                "answer_type":         answer_type,
            }])

            input_tokens  = max(0, int(np.expm1(input_model.predict(x_df)[0])))   # type: ignore[union-attr]
            output_tokens = max(0, int(np.expm1(output_model.predict(x_df)[0])))  # type: ignore[union-attr]
            total_tokens  = input_tokens + output_tokens
            cost_usd      = round(input_tokens * in_price + output_tokens * out_price, 6)

            estimates[server] = {
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "total_tokens":  total_tokens,
                "cost_usd":      cost_usd,
            }
            logger.debug(
                "(%s, %s) — in=%d out=%d total=%d cost=$%.6f",
                server, dataset, input_tokens, output_tokens, total_tokens, cost_usd,
            )

        return {
            "inferred_features": {
                "domain":      domain,
                "category":    category,
                "complexity":  complexity,
                "result_size": result_size,
                "intent":      intent,
                "answer_type": answer_type,
            },
            "estimates": estimates,
        }
