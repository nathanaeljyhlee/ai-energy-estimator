"""
Claude Code AI Energy Estimator
================================
A Gradio dashboard that reads Claude Code session transcripts from
~/.claude/projects/ and estimates the energy and water footprint of
your Claude Code usage.

Per-message model and token data is extracted from JSONL transcripts,
then energy coefficients derived from public research (Epoch AI,
ratherlegit/environmental-impact-tracker) are applied per model tier.

Run with: python app.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Auto-detect local UTC offset from the system timezone
_local_utc_offset = datetime.now(timezone.utc).astimezone().utcoffset() or timedelta(0)

# --- Energy constants (Wh per million tokens) ---
# Source: ratherlegit/environmental-impact-tracker (Epoch AI + Anthropic pricing signals)
ENERGY_RATES: dict[str, dict[str, float]] = {
    "opus": {"input_wh_per_mtok": 50, "output_wh_per_mtok": 250},
    "sonnet": {"input_wh_per_mtok": 30, "output_wh_per_mtok": 150},
    "haiku": {"input_wh_per_mtok": 10, "output_wh_per_mtok": 50},
}

JOULES_PER_TOKEN: dict[str, dict[str, float]] = {
    tier: {
        "input": rates["input_wh_per_mtok"] * 3600 / 1_000_000,
        "output": rates["output_wh_per_mtok"] * 3600 / 1_000_000,
    }
    for tier, rates in ENERGY_RATES.items()
}

WATER_ML_PER_WH = 1.7
INFRA_MULTIPLIER = 2.0

# Simple in-memory cache: (result, timestamp)
_cache: dict[str, tuple] = {}
CACHE_TTL = 300  # seconds


def classify_model(model_str: str) -> str:
    lower = model_str.lower()
    if "opus" in lower:
        return "opus"
    if "haiku" in lower:
        return "haiku"
    return "sonnet"


CACHE_READ_DISCOUNT = 0.1  # Cache reads cost ~10% of fresh input (matches Anthropic pricing ratio)


def estimate_energy_joules(
    input_tokens: int,
    output_tokens: int,
    model_tier: str,
    cache_read_tokens: int = 0,
) -> float:
    rates = JOULES_PER_TOKEN.get(model_tier, JOULES_PER_TOKEN["sonnet"])
    gpu_joules = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_tokens * rates["input"] * CACHE_READ_DISCOUNT
    )
    return gpu_joules * INFRA_MULTIPLIER


def joules_to_wh(joules: float) -> float:
    return joules / 3600


def joules_to_water_ml(joules: float) -> float:
    return joules_to_wh(joules) * WATER_ML_PER_WH


def parse_all_sessions(force_refresh: bool = False) -> list[dict]:
    """Parse all JSONL session transcripts. Cached for 5 minutes."""
    cache_key = "sessions"
    now = time.time()
    if not force_refresh and cache_key in _cache:
        result, ts = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return result

    sessions: list[dict] = []
    if not PROJECTS_DIR.exists():
        _cache[cache_key] = (sessions, now)
        return sessions

    for jsonl_path in PROJECTS_DIR.rglob("*.jsonl"):
        # Skip subagent transcripts to avoid double-counting
        if "subagents" in jsonl_path.parts:
            continue
        try:
            session = _parse_single_session(jsonl_path)
            if session and session["total_tokens"] > 0:
                sessions.append(session)
        except Exception:
            continue

    _cache[cache_key] = (sessions, now)
    return sessions


def _parse_single_session(jsonl_path: Path) -> dict | None:
    """Extract aggregated token/energy data from one JSONL session file."""
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_create_tokens = 0
    user_messages = 0
    assistant_messages = 0
    session_date = None
    model_counts: dict[str, int] = defaultdict(int)

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_type = row.get("type")

            # Get session date from first user message timestamp
            if row_type == "user" and session_date is None:
                ts = row.get("timestamp")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        local_dt = dt + _local_utc_offset
                        session_date = local_dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                user_messages += 1
                continue

            if row_type == "user":
                user_messages += 1
                continue

            if row_type == "assistant":
                assistant_messages += 1
                msg = row.get("message", {})
                usage = msg.get("usage")
                model = msg.get("model", "")
                if model and model != "<synthetic>":
                    tier = classify_model(model)
                    model_counts[tier] += 1
                if usage:
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
                    cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                    cache_create_tokens += usage.get("cache_creation_input_tokens", 0)

    if session_date is None:
        return None

    # Compute energy weighted by model usage per message
    total_model_msgs = sum(model_counts.values()) or 1
    energy_j = 0.0
    for tier, count in model_counts.items():
        weight = count / total_model_msgs
        tier_input = input_tokens * weight
        tier_output = output_tokens * weight
        tier_cache = cache_read_tokens * weight
        energy_j += estimate_energy_joules(
            int(tier_input), int(tier_output), tier, cache_read_tokens=int(tier_cache)
        )

    # Fallback: if no model detected, use sonnet
    if not model_counts:
        energy_j = estimate_energy_joules(
            input_tokens, output_tokens, "sonnet", cache_read_tokens=cache_read_tokens
        )

    # Dominant model for display
    dominant_model = max(model_counts, key=model_counts.get) if model_counts else "sonnet"

    return {
        "date": session_date,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_create_tokens": cache_create_tokens,
        "total_tokens": input_tokens + output_tokens,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "messages": user_messages + assistant_messages,
        "energy_joules": energy_j,
        "energy_wh": joules_to_wh(energy_j),
        "water_ml": joules_to_water_ml(energy_j),
        "dominant_model": dominant_model,
        "model_counts": dict(model_counts),
        "session_file": str(jsonl_path),
    }


def aggregate_daily(sessions: list[dict]) -> dict[str, dict]:
    daily: dict[str, dict] = defaultdict(lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cache_read_tokens": 0, "cache_create_tokens": 0,
        "sessions": 0, "messages": 0, "user_messages": 0,
        "energy_joules": 0.0, "energy_wh": 0.0, "water_ml": 0.0,
        "model_sessions": defaultdict(int),
    })
    for s in sessions:
        d = s["date"]
        daily[d]["input_tokens"] += s["input_tokens"]
        daily[d]["output_tokens"] += s["output_tokens"]
        daily[d]["total_tokens"] += s["total_tokens"]
        daily[d]["cache_read_tokens"] += s["cache_read_tokens"]
        daily[d]["cache_create_tokens"] += s["cache_create_tokens"]
        daily[d]["sessions"] += 1
        daily[d]["messages"] += s["messages"]
        daily[d]["user_messages"] += s["user_messages"]
        daily[d]["energy_joules"] += s["energy_joules"]
        daily[d]["energy_wh"] += s["energy_wh"]
        daily[d]["water_ml"] += s["water_ml"]
        daily[d]["model_sessions"][s["dominant_model"]] += 1
    return dict(sorted(daily.items()))


def aggregate_by_model(sessions: list[dict]) -> dict[str, dict]:
    by_model: dict[str, dict] = defaultdict(lambda: {
        "sessions": 0, "total_tokens": 0, "energy_joules": 0.0,
    })
    for s in sessions:
        m = s["dominant_model"]
        by_model[m]["sessions"] += 1
        by_model[m]["total_tokens"] += s["total_tokens"]
        by_model[m]["energy_joules"] += s["energy_joules"]
    return dict(by_model)


def format_joules(j: float) -> str:
    if j >= 1_000_000:
        return f"{j / 1_000_000:.1f} MJ"
    if j >= 1_000:
        return f"{j / 1_000:.1f} kJ"
    return f"{j:.0f} J"


def format_water(ml: float) -> str:
    if ml >= 1_000:
        return f"{ml / 1_000:.2f} L"
    return f"{ml:.1f} mL"


def _progress_bar(value: float, max_val: float, color: str, label: str) -> str:
    """Horizontal progress bar with labeled ticks."""
    pct = min(100, (value / max_val) * 100) if max_val > 0 else 0
    return f"""
    <div style="position:relative;background:#e5e7eb;border-radius:6px;height:28px;overflow:hidden;margin:6px 0;">
      <div style="background:{color};height:100%;width:{pct:.1f}%;border-radius:6px;transition:width 0.3s;"></div>
      <div style="position:absolute;top:0;left:8px;line-height:28px;font-size:0.78rem;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.3);">
        {label}
      </div>
    </div>"""


def _fill_icon(icon: str, fill_fraction: float, size: str = "3rem") -> str:
    """Single icon with partial fill using CSS clip. fill_fraction 0-1."""
    fill = max(0, min(1, fill_fraction))
    clip_pct = 100 - (fill * 100)
    return f"""
    <div style="position:relative;display:inline-block;width:{size};height:{size};font-size:{size};line-height:1;">
      <span style="opacity:0.15;">{icon}</span>
      <span style="position:absolute;top:0;left:0;clip-path:inset({clip_pct:.0f}% 0 0 0);">{icon}</span>
    </div>"""


def _fill_icons_row(icon: str, value: float, max_icons: int = 10, size: str = "2.2rem") -> str:
    """Row of icons where full ones are solid, last one is partially filled, rest are ghosted."""
    full = int(value)
    frac = value - full
    display_full = min(full, max_icons)
    icons = []
    for _ in range(display_full):
        icons.append(f'<span style="font-size:{size};line-height:1;display:inline-block;margin:0 1px;">{icon}</span>')
    if frac > 0.05 and display_full < max_icons:
        icons.append(_fill_icon(icon, frac, size))
    # Ghost remaining
    remaining = max_icons - display_full - (1 if frac > 0.05 and display_full < max_icons else 0)
    for _ in range(max(0, remaining)):
        icons.append(f'<span style="font-size:{size};line-height:1;display:inline-block;margin:0 1px;opacity:0.12;">{icon}</span>')
    return "".join(icons)


def build_equivalences_html(total_energy_j: float, total_water_ml: float) -> str:
    e_bike_miles = total_energy_j / 89_000
    microwave_minutes = total_energy_j / 60_000
    kwh = total_energy_j / 3_600_000
    phone_charges = kwh / 0.012
    glasses_water = total_water_ml / 240

    # --- E-bike: distance bar with landmark comparisons ---
    landmarks = [
        (0.5, "Around the block"),
        (2, "To the coffee shop"),
        (5, "Across town"),
        (10, "To the next town"),
        (26.2, "A marathon"),
        (50, "A day trip"),
        (100, "Century ride"),
    ]
    # Find the best-fit landmark for the scale
    bar_max = 5  # default
    for dist, _ in landmarks:
        if e_bike_miles <= dist * 1.3:
            bar_max = dist
            break
    else:
        bar_max = max(e_bike_miles * 1.2, 100)

    landmark_label = ""
    for dist, name in landmarks:
        if e_bike_miles <= dist * 1.1:
            landmark_label = name
            break
    if not landmark_label:
        landmark_label = f"{e_bike_miles:.0f} miles"

    bike_bar = _progress_bar(e_bike_miles, bar_max, "#16a34a", f"{e_bike_miles:.1f} mi")
    bike_context = f'<div style="font-size:0.75rem;color:#888;">~ {landmark_label} ({bar_max:.0f} mi scale)</div>'

    # --- Microwave: time bar with relatable durations ---
    time_comparisons = [
        (0.5, "Reheating coffee"),
        (2, "Heating soup"),
        (5, "Cooking a frozen meal"),
        (10, "Baking a potato"),
        (30, "Defrosting a chicken"),
        (60, "Running for an hour"),
    ]
    time_max = 5
    for mins, _ in time_comparisons:
        if microwave_minutes <= mins * 1.3:
            time_max = mins
            break
    else:
        time_max = max(microwave_minutes * 1.2, 60)

    time_label = ""
    for mins, name in time_comparisons:
        if microwave_minutes <= mins * 1.1:
            time_label = name
            break
    if not time_label:
        time_label = f"{microwave_minutes:.0f} min"

    if microwave_minutes >= 60:
        time_display = f"{microwave_minutes / 60:.1f} hr"
    else:
        time_display = f"{microwave_minutes:.1f} min"

    micro_bar = _progress_bar(microwave_minutes, time_max, "#d97706", time_display)
    micro_context = f'<div style="font-size:0.75rem;color:#888;">~ {time_label} ({time_max:.0f} min scale)</div>'

    # --- Phone charges: fill icons (max 10 phones) ---
    phone_scale = max(phone_charges, 1)
    if phone_scale <= 10:
        phone_row = _fill_icons_row("🔋", phone_charges, max_icons=10)
    else:
        # Scale down: each icon = N charges
        unit = phone_scale / 10
        phone_row = _fill_icons_row("🔋", phone_charges / unit, max_icons=10)
        phone_row += f'<div style="font-size:0.72rem;color:#888;margin-top:2px;">1 icon = {unit:.1f} charges</div>'

    # --- Water: fill icons (max 10 glasses) ---
    water_scale = max(glasses_water, 1)
    if water_scale <= 10:
        water_row = _fill_icons_row("🥛", glasses_water, max_icons=10)
    else:
        unit = water_scale / 10
        water_row = _fill_icons_row("🥛", glasses_water / unit, max_icons=10)
        water_row += f'<div style="font-size:0.72rem;color:#888;margin-top:2px;">1 icon = {unit:.1f} glasses</div>'

    html = f"""
<div style="font-family:sans-serif;">
  <h3 style="margin-bottom:16px;font-size:1.1rem;color:#333;">What does that energy look like?</h3>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div style="background:#f8f9fa;border-radius:10px;padding:14px 16px;">
      <div style="font-weight:600;font-size:0.85rem;color:#555;margin-bottom:2px;text-transform:uppercase;letter-spacing:.05em;">
        E-bike Distance
      </div>
      {bike_bar}
      {bike_context}
    </div>
    <div style="background:#f8f9fa;border-radius:10px;padding:14px 16px;">
      <div style="font-weight:600;font-size:0.85rem;color:#555;margin-bottom:2px;text-transform:uppercase;letter-spacing:.05em;">
        Microwave Time
      </div>
      {micro_bar}
      {micro_context}
    </div>
    <div style="background:#f8f9fa;border-radius:10px;padding:14px 16px;">
      <div style="font-weight:600;font-size:0.85rem;color:#555;margin-bottom:2px;text-transform:uppercase;letter-spacing:.05em;">
        Phone Charges &nbsp;<span style="font-weight:400;color:#888;">({phone_charges:.1f})</span>
      </div>
      <div style="margin-top:6px;line-height:1;">{phone_row}</div>
    </div>
    <div style="background:#f8f9fa;border-radius:10px;padding:14px 16px;">
      <div style="font-weight:600;font-size:0.85rem;color:#555;margin-bottom:2px;text-transform:uppercase;letter-spacing:.05em;">
        Glasses of Water &nbsp;<span style="font-weight:400;color:#888;">({glasses_water:.1f})</span>
      </div>
      <div style="margin-top:6px;line-height:1;">{water_row}</div>
    </div>
  </div>
</div>
"""
    return html


def build_metrics_html(total_tokens: int, total_sessions: int, total_energy_j: float, total_water_ml: float) -> str:
    metrics = [
        ("Total Tokens", f"{total_tokens:,}", "#4f46e5"),
        ("Sessions", f"{total_sessions:,}", "#0891b2"),
        ("Energy", format_joules(total_energy_j), "#d97706"),
        ("Water", format_water(total_water_ml), "#059669"),
    ]
    cards = "".join(
        f"""<div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 22px;
                        box-shadow:0 1px 4px rgba(0,0,0,.06);flex:1;min-width:140px;">
              <div style="font-size:0.78rem;color:#6b7280;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;">{label}</div>
              <div style="font-size:1.75rem;font-weight:700;color:{color};">{value}</div>
            </div>"""
        for label, value, color in metrics
    )
    return f'<div style="display:flex;gap:14px;flex-wrap:wrap;font-family:sans-serif;margin-bottom:8px;">{cards}</div>'


def make_bar_chart(x: list, y: list, title: str, color: str = "#4f46e5", y_label: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=color))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fff",
        yaxis_title=y_label,
        height=260,
        font=dict(family="sans-serif", size=11),
        bargap=0.25,
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def make_pie_chart(labels: list, values: list, title: str) -> go.Figure:
    colors = {"opus": "#7c3aed", "sonnet": "#2563eb", "haiku": "#0891b2"}
    marker_colors = [colors.get(l, "#94a3b8") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=marker_colors),
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        font=dict(family="sans-serif", size=11),
        showlegend=True,
    )
    return fig


def build_table_html(filtered_dates: list, filtered_daily: dict) -> str:
    if not filtered_dates:
        return "<p>No data.</p>"

    rows_html = ""
    for d in filtered_dates:
        v = filtered_daily[d]
        model_str = ", ".join(
            f"{m}:{c}" for m, c in sorted(v["model_sessions"].items(), key=lambda x: -x[1])
        )
        rows_html += f"""
        <tr style="border-bottom:1px solid #e5e7eb;">
          <td style="padding:7px 10px;white-space:nowrap;">{d}</td>
          <td style="padding:7px 10px;text-align:right;">{v['input_tokens']:,}</td>
          <td style="padding:7px 10px;text-align:right;">{v['output_tokens']:,}</td>
          <td style="padding:7px 10px;text-align:right;font-weight:600;">{v['total_tokens']:,}</td>
          <td style="padding:7px 10px;text-align:right;">{v['cache_read_tokens']:,}</td>
          <td style="padding:7px 10px;text-align:right;">{v['sessions']}</td>
          <td style="padding:7px 10px;">{model_str}</td>
          <td style="padding:7px 10px;text-align:right;">{v['energy_joules']:.1f}</td>
          <td style="padding:7px 10px;text-align:right;">{v['energy_wh']:.3f}</td>
          <td style="padding:7px 10px;text-align:right;">{v['water_ml']:.1f}</td>
        </tr>"""

    return f"""
<div style="overflow-x:auto;font-family:sans-serif;font-size:0.82rem;">
<table style="width:100%;border-collapse:collapse;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
  <thead>
    <tr style="background:#f3f4f6;font-weight:600;color:#374151;">
      <th style="padding:8px 10px;text-align:left;">Date</th>
      <th style="padding:8px 10px;text-align:right;">Input Tok</th>
      <th style="padding:8px 10px;text-align:right;">Output Tok</th>
      <th style="padding:8px 10px;text-align:right;">Total Tok</th>
      <th style="padding:8px 10px;text-align:right;">Cache Read</th>
      <th style="padding:8px 10px;text-align:right;">Sessions</th>
      <th style="padding:8px 10px;text-align:left;">Models</th>
      <th style="padding:8px 10px;text-align:right;">Energy (J)</th>
      <th style="padding:8px 10px;text-align:right;">Energy (Wh)</th>
      <th style="padding:8px 10px;text-align:right;">Water (mL)</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>"""


METHODOLOGY_HTML = """
<div style="font-family:sans-serif;font-size:0.85rem;line-height:1.65;color:#374151;max-width:860px;">
<p><strong>Data source:</strong> Live JSONL session transcripts from <code>~/.claude/projects/</code>.
Each assistant message includes the exact model name and per-message token usage (input, output, cache read, cache creation).
This dashboard aggregates across all messages in all sessions.</p>

<p><strong>Energy rates (GPU compute, per token):</strong></p>
<table style="border-collapse:collapse;margin-bottom:12px;font-size:0.82rem;">
  <thead><tr style="background:#f3f4f6;">
    <th style="padding:6px 12px;border:1px solid #ddd;">Model Tier</th>
    <th style="padding:6px 12px;border:1px solid #ddd;">Input (J/token)</th>
    <th style="padding:6px 12px;border:1px solid #ddd;">Output (J/token)</th>
    <th style="padding:6px 12px;border:1px solid #ddd;">Source</th>
  </tr></thead>
  <tbody>
    <tr><td style="padding:5px 12px;border:1px solid #ddd;">Opus</td><td style="padding:5px 12px;border:1px solid #ddd;">0.180</td><td style="padding:5px 12px;border:1px solid #ddd;">0.900</td><td style="padding:5px 12px;border:1px solid #ddd;">ratherlegit (Epoch AI + pricing signals)</td></tr>
    <tr><td style="padding:5px 12px;border:1px solid #ddd;">Sonnet</td><td style="padding:5px 12px;border:1px solid #ddd;">0.108</td><td style="padding:5px 12px;border:1px solid #ddd;">0.540</td><td style="padding:5px 12px;border:1px solid #ddd;">ratherlegit (Epoch AI + pricing signals)</td></tr>
    <tr><td style="padding:5px 12px;border:1px solid #ddd;">Haiku</td><td style="padding:5px 12px;border:1px solid #ddd;">0.036</td><td style="padding:5px 12px;border:1px solid #ddd;">0.180</td><td style="padding:5px 12px;border:1px solid #ddd;">ratherlegit (Epoch AI + pricing signals)</td></tr>
  </tbody>
</table>

<p><strong>Infrastructure multiplier:</strong> 2x GPU energy (MIT Technology Review, 2025:
"doubling the GPU energy gives an approximate estimate of the entire operation's energy demands")</p>

<p><strong>Water usage:</strong> 1.7 mL per Wh (Li et al. 2023, UC Riverside — midpoint of 1.3–2.0 range for data center cooling + power generation)</p>

<p><strong>Cross-check:</strong> Epoch AI (Feb 2025) estimated ~1,080 J per ChatGPT query. Your avg assistant message is ~194 output tokens,
implying ~5.6 J/output-token at full infrastructure cost. The rates above yield ~1.08 J/output-token (Sonnet, with 2x multiplier) —
lower because they're derived from compute benchmarks, not metered datacenter power. True energy is likely between these bounds.</p>

<p><strong>Cross-check (Google, Aug 2025):</strong> Google disclosed median Gemini text prompt = 0.24 Wh, 0.26 mL water.
This is the only first-party per-query energy disclosure from a major AI company. It implies ~1.08 mL/Wh water usage efficiency,
lower than the 1.7 mL/Wh midpoint used here.</p>

<p><strong>Cache read handling:</strong> Cache read tokens are counted at 10% of fresh input energy cost, matching
Anthropic's pricing ratio for cached vs. uncached input. Claude Code sessions are heavily cache-dependent,
so this materially affects totals.</p>

<p><strong>Model detection:</strong> Each session's model is detected from the <code>model</code> field in assistant messages.
Energy is weighted proportionally if multiple models are used within one session (e.g., Haiku subagents within a Sonnet session).</p>

<p><strong>What's NOT counted:</strong> Subagent JSONL files (in <code>subagents/</code> subdirectories) are excluded to avoid
double-counting — their token usage is already included in the parent session's assistant messages.</p>

<p><strong>Sources:</strong><br>
- MIT Technology Review (2025/05/20) — AI energy usage &amp; climate footprint<br>
- Epoch AI (2025) — inference compute trends<br>
- Li et al. (2023) — Making AI Less Thirsty (arXiv:2304.03271)<br>
- ratherlegit/environmental-impact-tracker — Claude-specific energy coefficients<br>
- Patterson et al. (2021) — Carbon and the Cloud (arXiv:2104.10350)
</p>
</div>
"""


def run_dashboard(time_range: str, force_refresh: bool = False):
    """Core logic: parse data, apply filter, return all UI components."""
    sessions = parse_all_sessions(force_refresh=force_refresh)

    if not sessions:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"No session data found in {PROJECTS_DIR}", showarrow=False, font=dict(size=14))],
            height=260,
        )
        no_data_html = f"<p style='color:#b91c1c;font-family:sans-serif;'>No session data found in <code>{PROJECTS_DIR}</code>.</p>"
        return (
            no_data_html,           # metrics
            empty_fig,              # daily tokens
            empty_fig,              # daily energy
            empty_fig,              # daily sessions
            empty_fig,              # model mix
            no_data_html,           # equivalences
            no_data_html,           # table
        )

    daily = aggregate_daily(sessions)
    dates = sorted(daily.keys())

    filter_map = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "All time": 9999}
    days_back = filter_map.get(time_range, 30)
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    filtered_dates = [d for d in dates if d >= cutoff]
    filtered_daily = {d: daily[d] for d in filtered_dates}
    filtered_sessions = [s for s in sessions if s["date"] >= cutoff]

    if not filtered_daily:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text="No data in selected time range", showarrow=False, font=dict(size=14))],
            height=260,
        )
        no_data_html = "<p style='color:#b91c1c;font-family:sans-serif;'>No data in selected time range.</p>"
        return (no_data_html, empty_fig, empty_fig, empty_fig, empty_fig, no_data_html, no_data_html)

    # Aggregates
    total_tokens = sum(v["total_tokens"] for v in filtered_daily.values())
    total_energy_j = sum(v["energy_joules"] for v in filtered_daily.values())
    total_water = sum(v["water_ml"] for v in filtered_daily.values())
    total_sessions_count = sum(v["sessions"] for v in filtered_daily.values())

    metrics_html = build_metrics_html(total_tokens, total_sessions_count, total_energy_j, total_water)

    # Charts
    chart_tokens = make_bar_chart(
        filtered_dates,
        [filtered_daily[d]["total_tokens"] for d in filtered_dates],
        "Daily Tokens",
        color="#4f46e5",
        y_label="Tokens",
    )
    chart_energy = make_bar_chart(
        filtered_dates,
        [round(filtered_daily[d]["energy_wh"], 3) for d in filtered_dates],
        "Daily Energy (Wh)",
        color="#d97706",
        y_label="Wh",
    )
    chart_sessions = make_bar_chart(
        filtered_dates,
        [filtered_daily[d]["sessions"] for d in filtered_dates],
        "Daily Sessions",
        color="#0891b2",
        y_label="Sessions",
    )

    model_data = aggregate_by_model(filtered_sessions)
    if model_data:
        chart_models = make_pie_chart(
            list(model_data.keys()),
            [v["sessions"] for v in model_data.values()],
            "Model Mix (sessions)",
        )
    else:
        chart_models = go.Figure()
        chart_models.update_layout(
            annotations=[dict(text="No model data", showarrow=False)],
            height=260,
        )

    # Equivalences with icon grids
    equiv_html = build_equivalences_html(total_energy_j, total_water)

    # Table
    table_html = build_table_html(filtered_dates, filtered_daily)

    return (
        metrics_html,
        chart_tokens,
        chart_energy,
        chart_sessions,
        chart_models,
        equiv_html,
        table_html,
    )


def refresh_and_run(time_range: str):
    return run_dashboard(time_range, force_refresh=True)


def build_ui():
    with gr.Blocks(
        title="Claude Code Usage & Energy",
    ) as demo:
        gr.HTML("""
        <div style="font-family:sans-serif;border-bottom:1px solid #e5e7eb;padding-bottom:16px;margin-bottom:8px;">
          <h1 style="font-size:1.5rem;font-weight:700;margin:0 0 4px;">Claude Code Usage &amp; Energy Dashboard</h1>
          <p style="font-size:0.82rem;color:#6b7280;margin:0;">
            Source: ~/.claude/projects/**/*.jsonl (live session transcripts)
            &nbsp;|&nbsp; Energy estimates from Epoch AI + ratherlegit methodology
            &nbsp;|&nbsp; <em>DISCLAIMER: Estimates only, not official Anthropic figures.</em>
          </p>
        </div>
        """)

        with gr.Row():
            time_range = gr.Dropdown(
                choices=["Last 7 days", "Last 14 days", "Last 30 days", "All time"],
                value="Last 30 days",
                label="Time range",
                scale=3,
            )
            refresh_btn = gr.Button("Refresh data", variant="secondary", scale=1)

        # Metrics
        metrics_html = gr.HTML()

        # Charts row 1
        with gr.Row():
            chart_tokens = gr.Plot(label="Daily Tokens")
            chart_energy = gr.Plot(label="Daily Energy (Wh)")

        # Charts row 2
        with gr.Row():
            chart_sessions = gr.Plot(label="Daily Sessions")
            chart_models = gr.Plot(label="Model Mix")

        # Equivalences
        equiv_html = gr.HTML()

        # Expandable sections
        with gr.Accordion("Daily Breakdown", open=False):
            table_html = gr.HTML()

        with gr.Accordion("Methodology & Sources", open=False):
            gr.HTML(METHODOLOGY_HTML)

        # Wire up outputs
        outputs = [metrics_html, chart_tokens, chart_energy, chart_sessions, chart_models, equiv_html, table_html]

        time_range.change(fn=run_dashboard, inputs=[time_range], outputs=outputs)
        refresh_btn.click(fn=refresh_and_run, inputs=[time_range], outputs=outputs)

        # Load initial data on page open
        demo.load(fn=run_dashboard, inputs=[time_range], outputs=outputs)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Default(
            primary_hue="indigo",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .gradio-container { max-width: 1200px !important; margin: 0 auto; }
        .accordion-header { font-weight: 600 !important; }
        footer { display: none !important; }
        """,
    )
