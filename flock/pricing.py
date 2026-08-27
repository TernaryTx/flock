from __future__ import annotations

from datetime import date

# Dollars per million tokens, list price: https://www.anthropic.com/pricing
RATES = {
    'claude-sonnet-5': {'input': 3.00, 'output': 15.00},
    'claude-sonnet-4-6': {'input': 3.00, 'output': 15.00},
    'claude-haiku-4-5': {'input': 0.80, 'output': 4.00},
}

# Introductory pricing, which applies instead of the list price until it ends.
INTRO_RATES = {'claude-sonnet-5': {'input': 2.00, 'output': 10.00}}
INTRO_RATES_END = date(2026, 8, 31)


def token_cost(
        usage: dict,
        model: str,
        intro: bool = False,
        cache_write_multiplier: float = 1.25,
        cache_read_multiplier: float = 0.10,
) -> float:
    """Price one response's token usage at non-batch rates.

    Args:
        usage: Token counts from a response, as the API names them.
        model: Which model answered.
        intro: Use introductory rates where the model has them.
        cache_write_multiplier: What writing the cache costs against input rate.
        cache_read_multiplier: What reading it back costs against input rate.

    Returns:
        Dollars for this response.

    Raises:
        KeyError: If the model has no rates here, which is a model this cannot
            cost rather than one to cost wrongly.
    """
    rates = INTRO_RATES[model] if intro and model in INTRO_RATES else RATES[model]
    billed_input = float(usage.get('input_tokens', 0))
    billed_input += usage.get(
        'cache_creation_input_tokens', 0,
    ) * cache_write_multiplier
    billed_input += usage.get(
        'cache_read_input_tokens', 0,
    ) * cache_read_multiplier
    cost = billed_input / 1e6 * rates['input']
    return cost + usage.get('output_tokens', 0) / 1e6 * rates['output']
