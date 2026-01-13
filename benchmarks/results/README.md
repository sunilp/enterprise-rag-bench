# Benchmark Results

Results from running `python benchmarks/run_benchmarks.py` are saved here.

## Running Benchmarks

1. Prepare your documents in `benchmarks/financial_docs/`
2. Create `eval_cases.json` with question-answer pairs
3. Configure embedding and LLM providers in the benchmark script
4. Run: `python benchmarks/run_benchmarks.py`

## Output Format

Results are saved as JSON with per-pattern, per-case metrics:
- Faithfulness, relevance, groundedness, context precision
- Latency (ms), token usage, cost
- Composite score (weighted average)

## Reproducing Results

All benchmark code is deterministic given the same:
- Document corpus
- Eval cases
- Embedding model
- LLM model and temperature (0)
- Top-k parameter
