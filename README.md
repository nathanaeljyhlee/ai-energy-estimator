# ai-energy-estimator

Estimate the energy and water footprint of your Claude Code usage.

## Screenshot

![AI Energy Estimator dashboard](<screenshot.png>)

## How it works

1. Reads Claude Code session transcripts from `~/.claude/projects/` (JSONL files written automatically by Claude Code)
2. Extracts per-message token usage and model info from each assistant message
3. Applies energy coefficients derived from public research (Epoch AI benchmarks + Anthropic pricing signals) to compute GPU compute energy per token, then scales by a 2x infrastructure multiplier

All processing is local — no data leaves your machine.

> **Requires Claude Code** to be installed and to have existing session history in `~/.claude/projects/`.

## Energy rates

Rates are in Wh per million tokens (GPU compute only), then multiplied by 2x for full infrastructure.

| Model | Input (Wh/MTok) | Output (Wh/MTok) | Input (J/token) | Output (J/token) |
|-------|-----------------|------------------|-----------------|------------------|
| Opus  | 50              | 250              | 0.180           | 0.900            |
| Sonnet| 30              | 150              | 0.108           | 0.540            |
| Haiku | 10              | 50               | 0.036           | 0.180            |

**Infrastructure multiplier:** 2x GPU energy, per MIT Technology Review (2025): "doubling the GPU energy gives an approximate estimate of the entire operation's energy demands."

**Water:** 1.7 mL/Wh (Li et al. 2023, UC Riverside — midpoint of 1.3–2.0 mL/Wh range for data center cooling + power generation).

## Quickstart

```bash
pip install -r requirements.txt && python app.py
```

The dashboard opens at `http://127.0.0.1:7861` in your browser.

## Sources

1. **MIT Technology Review (2025/05/20)** — AI energy usage and climate footprint
   https://www.technologyreview.com/2025/05/20/1116327/ais-energy-use-is-surging-faster-than-anyone-expected/

2. **Epoch AI (2025)** — Inference compute trends and per-query energy estimates
   https://epochai.org/blog/inference-compute-trends

3. **Li et al. (2023)** — Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models (arXiv:2304.03271)
   https://arxiv.org/abs/2304.03271

4. **ratherlegit/environmental-impact-tracker** — Model-specific Wh/MTok coefficients derived from public benchmarks and pricing signals
   https://github.com/ratherlegit/environmental-impact-tracker

5. **Patterson et al. (2021)** — Carbon and the Cloud (arXiv:2104.10350)
   https://arxiv.org/abs/2104.10350

## Disclaimer

These are estimates derived from public benchmarks and pricing signals, not official Anthropic energy figures. Actual energy consumption will vary based on data center location, hardware generation, utilization rates, and other factors not captured in these models.

## License

MIT License — see [LICENSE](LICENSE)
