"""
Test the new prompt (v3) against races WITHOUT changing the live predictor.

Supports both historical (backtest) and live races.

Usage:
    # Historical (uses PuntingForm SP odds)
    python experiments/test_new_prompt.py "Sandown-Hillside" 5 "28-Jan-2026"

    # Live / today's races (uses Ladbrokes live odds)
    python experiments/test_new_prompt.py "Randwick" 3 "29-Jan-2026" --live

    # Test different prompt variants
    python experiments/test_new_prompt.py "Rosehill" 1 "31-Jan-2026" -V A
    python experiments/test_new_prompt.py "Rosehill" 1 "31-Jan-2026" -V B  (default)
    python experiments/test_new_prompt.py "Rosehill" 1 "31-Jan-2026" -V C
    python experiments/test_new_prompt.py "Rosehill" 1 "31-Jan-2026" -V D
"""
import sys
sys.path.insert(0, '/Users/danielsamus/punt-legacy-ai')

import anthropic
import json
import re

# ============================================================
# VARIANT A: "Minimal" — bare bones, almost no instructions
# ============================================================
PROMPT_A = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race.

Rules:
- Compare each horse's speed ratings at SIMILAR distance and conditions to today's race
- More recent ratings matter more
- The horse(es) with the best ratings at similar distance/conditions are your contenders
- Barrier trials (marked TRIAL) don't count
- If a horse has 0 race runs, they are UNKNOWN
- Pick 0 if too many unknowns or no clear standouts

Secondary: Jockey/trainer A/E ratios, weight changes, first/second-up patterns.

Tags: "The one to beat" (max 1, clear best). For other picks, tag however you like.

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "your choice",
      "analysis": "1-3 sentences",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "Brief overview"
}
```

tipsheet_pick = true only if you'd genuinely bet on this horse at these odds."""

# ============================================================
# VARIANT B: "Current test prompt" (v3) - ratings-focused
# ============================================================
PROMPT_B = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race:

- **"The one to beat"** (0-1 pick) - Your top pick. Clear standout with best chance to win
- **"Value bet"** (0-2 picks) - Genuine winning chance, odds better than form suggests

Always list "The one to beat" first if selected.

**Pick 0 contenders (no bet) when:**
- Multiple runners in the field has ZERO race runs (first starters/only trials) - can't compare unknowns
- Field is too even with no standouts

**Lower confidence (but still pick) when:**
- Many horses have limited form (1-3 runs) - assess with caution, still make picks but comment on it in the summary

## Key Analysis

**Primary factor:** Normalized speed ratings (100 = average for distance/condition) from RACE runs (not trials) at SIMILAR distance and conditions. More recent runs are more relevant.
Compare ratings WITHIN the field.

**How to evaluate contenders:**
Compare ratings at similar distance/conditions WITHIN the field. The horses with the best recent ratings at similar distance/conditions are your contenders. A favourite with the best ratings and consistent form is a genuine pick. A roughie with a standout rating at similar distance/conditions to the race being predicted can be a value pick. For first/second-up horses, check if their past first/second-up ratings are better or worse than their mid-prep ratings.

**Secondary factors:** Jockey/trainer A/E ratios (>1.0 = beats market expectations), barrier, weight changes, first/second-up prep patterns.

Use win/place odds to assess value - is the horse better/worse than the market thinks?

**Do not over-weight:** Career win/place record, finishing margins, or perceived "class". These are already reflected in the speed ratings. Focus on the numbers.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form - horses don't always try
- If a horse has 0 race runs, they are UNKNOWN - could be brilliant or useless
- Limited form (1-3 runs) is still usable data - don't skip these races, but comment on limited form

## Output

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "Value bet",
      "analysis": "1-3 sentences referencing RACE form",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "Brief race overview if picks made, or reason for 0 picks"
}
```

**tipsheet_pick = true** when you would genuinely bet on this horse at these odds:
- Speed ratings clearly support this horse vs the field at similar distances/conditions
- The odds represent real value
- You're confident in the pick, not just filling a slot"""

# ============================================================
# VARIANT C: "Ratings only" — pure speed rating focus
# ============================================================
PROMPT_C = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race based ONLY on normalized speed ratings.

Rules:
- Compare each horse's speed ratings at SIMILAR distance and conditions to today's race
- More recent ratings matter more
- The horse(es) with the best ratings at similar distance/conditions are your contenders
- Ignore everything else: jockey, trainer, barrier, weight, career record
- Barrier trials (marked TRIAL) don't count
- If a horse has 0 race runs, they are UNKNOWN
- Pick 0 if too many unknowns or no clear standouts

Tags: "The one to beat" (max 1, clear best) or "Value bet" (odds better than ratings suggest).

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "Value bet",
      "analysis": "1-3 sentences on ratings only",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "Brief overview"
}
```

tipsheet_pick = true only if ratings clearly support this horse AND you'd genuinely bet at these odds."""

# ============================================================
# VARIANT D: "Strict value" — ratings + strict value filter
# ============================================================
PROMPT_D = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. Be HIGHLY SELECTIVE — only pick when there's a clear edge.

- **"The one to beat"** (0-1 pick) - Clear standout. Best ratings at similar distance/conditions AND consistent form.
- **"Value bet"** (0-2 picks) - Standout rating at similar distance/conditions AND odds are at least 20% higher than you'd expect.

**Pick 0 contenders when:**
- Multiple runners have ZERO race runs
- Field is competitive with several horses on similar ratings
- No horse has a clear ratings edge over the rest
- The best horse is already fairly priced (no value)

**Analysis method:**
Compare normalized speed ratings (100 = average) from RACE runs (not trials) at SIMILAR distance and conditions. More recent = more relevant. Compare WITHIN the field.

For first/second-up horses, check past first/second-up ratings vs mid-prep ratings.

Secondary: Jockey/trainer A/E ratios, barrier, weight changes.

**Do not over-weight:** Career win/place record, finishing margins, perceived "class" — already in the ratings.

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "Value bet",
      "analysis": "1-3 sentences",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "Brief overview or reason for 0 picks"
}
```

tipsheet_pick = true ONLY when:
- Ratings clearly separate this horse from the field
- The odds represent genuine value
- You are confident — if in doubt, tipsheet_pick = false"""

# ============================================================
# VARIANT E: "OG free-form" — original contender-based prompt (commit 2aad842)
# ============================================================
PROMPT_E = """You are an expert horse racing analyst specializing in Australian thoroughbred racing.

Your task is to identify the CONTENDERS (0-3 horses that could realistically win) and give your thoughts on each.

## Your Approach

**Step 1: Identify contenders.** Which horses could realistically WIN this race? Could be 0, 1, 2, or 3 picks.

**Step 2: Rank them.** Who's the best horse? Who else has a genuine chance?

**Step 3: Give your view.** For each contender, explain why they can win and your thoughts on the price.

## Key Analysis

Compare each horse's speed ratings at SIMILAR distance and conditions to today's race. More recent ratings matter more. The horse(es) with the best ratings at similar distance/conditions are your contenders.

**Condition matching:** Prioritise ratings on similar conditions to today's race (±1 step, e.g. if today is G4, focus on G3-S5 ratings).Also prioritise similar distance (±0-20%). If a horse's best ratings are only on very different conditions or distances, treat them with caution.

**Secondary factors:** Jockey/trainer A/E ratios, weight changes, first/second-up patterns, barrier.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form
- If a horse has 0 race runs, they are UNKNOWN
- Pick 0 if too many unknowns or no clear standouts
- If many horses have limited form (1-3 runs), still make picks but mention it in the summary

## Output Format

Return 0-3 contenders as JSON:

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "best" or your choice of tag,
      "analysis": "2-3 sentences: why this horse can win, and your thoughts on the price. Be natural - give your honest view on value.",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "1-2 sentences summarizing the race overall, or reason for 0 picks"
}
```

**Tags:**
- "best" = Your top pick, most likely winner (max 1)
- For other picks, tag however you like - be descriptive

**tipsheet_pick = true** Star your confident picks where:
- Speed ratings at similar conditions/distance are clearly superior to the field
- The odds represent genuine value
- You'd put your own money on it at those odds.

## Important Rules

1. Only include horses that could realistically WIN based on ratings at similar conditions / distance to the race being predicted
2. Don't use explicit percentages or rigid language
3. Be honest: if no horse stands out, pick 0"""

# ============================================================
# VARIANT F: "Live predictor" — current production prompt from core/predictor.py
# ============================================================
from core.predictor import SYSTEM_PROMPT as LIVE_SYSTEM_PROMPT
PROMPT_F = LIVE_SYSTEM_PROMPT

# ============================================================
# Variant lookup
# ============================================================
VARIANTS = {
    'A': ('Minimal', PROMPT_A),
    'B': ('Current (v3)', PROMPT_B),
    'C': ('Ratings only', PROMPT_C),
    'D': ('Strict value', PROMPT_D),
    'E': ('OG free-form', PROMPT_E),
    'F': ('Live predictor', PROMPT_F),
}


def run_new_prompt_prediction(prompt_text, verbose=False, system_prompt=None):
    if system_prompt is None:
        system_prompt = PROMPT_B
    client = anthropic.Anthropic()
    if verbose:
        user_msg = f"Analyze this race and pick your contenders (0-3). Provide a brief verdict for EVERY runner explaining their ratings at similar distance/conditions and whether they're a contender or not.\n\n{prompt_text}\n\nRespond with JSON only. Use this format:\n{{\n  \"runners\": [{{\"horse\": \"Name\", \"tab_no\": N, \"odds\": N, \"verdict\": \"2-3 sentences on ratings\"}}],\n  \"contenders\": [{{\"horse\": \"Name\", \"tab_no\": N, \"odds\": N, \"tag\": \"The one to beat\"|\"Value bet\", \"analysis\": \"1-3 sentences\", \"tipsheet_pick\": true|false}}],\n  \"summary\": \"Brief overview\"\n}}"
    else:
        user_msg = f"Analyze this race and pick your contenders (0-3).\n\n{prompt_text}\n\nRespond with JSON only."
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def get_live_data(track, race_num, date):
    """Get race data using the live pipeline (Ladbrokes odds)."""
    from core.race_data import RaceDataPipeline
    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_num, date)
    if error:
        return None, error
    return race_data, None


def display_results(raw):
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        data = json.loads(json_match.group()) if json_match else {}

        # Per-runner verdicts (verbose mode)
        runners = data.get('runners', [])
        if runners:
            print(f"\n{'='*60}")
            print("  RUNNER ANALYSIS")
            print(f"{'='*60}")
            for rv in runners:
                print(f"\n  {rv.get('tab_no', '?')}. {rv['horse']} (${rv.get('odds', '?')})")
                print(f"     {rv['verdict']}")

        # Contenders
        contenders = data.get('contenders', [])
        print(f"\n{'='*60}")
        print("  CONTENDERS")
        print(f"{'='*60}")

        if not contenders:
            print(f"\n  NO CONTENDERS")
        else:
            for c in contenders:
                tipsheet = " ⭐ TIPSHEET" if c.get('tipsheet_pick') else ""
                print(f"\n  {c['horse']} (#{c['tab_no']}){tipsheet}")
                print(f"    ${c['odds']:.2f}")
                print(f'    "{c.get("tag") or c.get("chance", "")}"')
                print(f"    {c['analysis']}")

        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"\n  {data.get('summary', 'No summary')}")

    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw: {raw[:500]}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    track = sys.argv[1]
    race_num = int(sys.argv[2])
    date = sys.argv[3]
    live_mode = "--live" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Parse variant flag
    variant_key = 'B'  # default
    for i, arg in enumerate(sys.argv):
        if arg in ('-V', '--variant') and i + 1 < len(sys.argv):
            variant_key = sys.argv[i + 1].upper()
    if variant_key not in VARIANTS:
        print(f"Unknown variant '{variant_key}'. Choose from: {', '.join(VARIANTS.keys())}")
        sys.exit(1)
    variant_name, variant_prompt = VARIANTS[variant_key]

    print(f"\n{'='*60}")
    print(f"  NEW PROMPT TEST: {track} R{race_num} - {date}")
    print(f"  Mode: {'LIVE (Ladbrokes odds)' if live_mode else 'BACKTEST (PuntingForm SP)'}")
    print(f"  Variant: {variant_key} ({variant_name})")
    print(f"{'='*60}")

    if live_mode:
        race_data, error = get_live_data(track, race_num, date)
        if error:
            print(f"Error: {error}")
            sys.exit(1)

        print(f"\nDistance: {race_data.distance}m | Condition: {race_data.condition}")
        print(f"Runners: {len(race_data.runners)}")
        print(f"\nFIELD:")
        for r in sorted(race_data.runners, key=lambda x: x.tab_no):
            odds_str = f"${r.odds:.2f}" if r.odds else "N/A"
            print(f"  {r.tab_no}. {r.name} - {odds_str}")

        prompt_text = race_data.to_prompt_text()

    else:
        from experiments.backtest import get_backtest_prompt
        result = get_backtest_prompt(track, race_num, date)

        if result[0] is None:
            print(f"Error: {result[1]}")
            sys.exit(1)

        prompt_text, runners, race_info = result

        print(f"\nDistance: {race_info['distance']}m | Condition: {race_info['condition']}")
        print(f"Class: {race_info['name']}")
        print(f"Form: {race_info['with_form']}/{race_info['total']} have race runs")
        print(f"\nFIELD:")
        for r in runners:
            print(f"  {r['tab']}. {r['name']} - SP ${r['sp']:.2f}")

    print(f"\n{'='*60}")
    print(f"  RUNNING VARIANT {variant_key} ({variant_name})...")
    print(f"{'='*60}")

    raw = run_new_prompt_prediction(prompt_text, verbose=verbose, system_prompt=variant_prompt)
    display_results(raw)
