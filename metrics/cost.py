def calculate_cost(usage: dict, input_price_per_million: float = 0.15, output_price_per_million: float = 0.60) -> float:
    input_cost=usage["input_tokens"]*input_price_per_million/1000000
    output_cost=usage["output_tokens"]*output_price_per_million/1000000
    return input_cost+output_cost