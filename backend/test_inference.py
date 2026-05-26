def estimate(self, question: str, gpt_model: str) -> dict:
    """
    Return predicted input/output token counts and USD costs per platform.

    Response structure:
        {
          "inferred_features": {domain, complexity, intent},
          "estimates": {
            "<platform>": {
              "input_tokens": int,
              "output_tokens": int,
              "total_tokens": int,
              "cost_usd": float
            }
          }
        }
    """
    domain, complexity, intent = self.infer_features(question)

    x_features = np.array([[
        len(question),
        len(question.split()),
        DOMAIN_MAP[domain],
        COMPLEXITY_MAP[complexity],
        INTENT_MAP[intent],
    ]])

    if gpt_model not in PRICING:
        logger.warning(
            "Unknown gpt_model '%s' — defaulting to gpt-4o pricing", gpt_model
        )
    cost_per_token = PRICING.get(gpt_model, PRICING["gpt-4o"]) / 1000

    estimates: dict[str, dict] = {}
    for platform, platform_models in self.models.items():
        input_model = platform_models.get("input_tokens")
        output_model = platform_models.get("output_tokens")

        if input_model is None or output_model is None:
            logger.warning(
                "Incomplete models for platform '%s' — skipping", platform
            )
            continue

        input_tokens = max(0, int(input_model.predict(x_features)[0]))
        output_tokens = max(0, int(output_model.predict(x_features)[0]))
        total_tokens = input_tokens + output_tokens
        cost_usd = round(total_tokens * cost_per_token, 6)

        estimates[platform] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }
        logger.debug(
            "Platform '%s' — in=%d out=%d total=%d cost=$%.6f",
            platform, input_tokens, output_tokens, total_tokens, cost_usd,
        )

    return {
        "inferred_features": {
            "domain": domain,
            "complexity": complexity,
            "intent": intent,
        },
        "estimates": estimates,
    }