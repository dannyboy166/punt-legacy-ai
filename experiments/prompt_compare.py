"""
Prompt Comparison Tool

Runs two different prompts (LIVE vs TEST) on the same race data
and outputs a side-by-side HTML comparison.

Usage:
    # Single race
    python experiments/prompt_compare.py --track "Caulfield Heath" --race 1 --date "04-Mar-2026"

    # Entire meeting (parallel)
    python experiments/prompt_compare.py --track "Caulfield Heath" --date "04-Mar-2026" --all
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from dotenv import load_dotenv

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.race_data import RaceDataPipeline, RaceData
from core.normalize import normalize_horse_name
from api.puntingform import PuntingFormAPI

load_dotenv()

# =============================================================================
# PROMPTS
# =============================================================================

# LIVE PROMPT - Current production prompt + bonus bet extras
LIVE_SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout
- **"Each-way chance"** - Good ratings at similar DISTANCE and CONDITIONS vs the field, place odds $1.80+
- **"Value bet"** - Odds better than their form suggests

**Pick 0 contenders (no bet) when:**
- A lot of field has no race form (only trials) - you can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

## Key Analysis

Focus on **normalized speed ratings** from RACE runs (not trials) at similar distance and conditions to the race being predicted. More recent runs are more relevant. **Speed ratings matter more than last start wins or career win/place stats.**

**First-up/Second-up horses:** Check their past runs at the same prep stage (Prep=1 in form table for first-up runs, Prep=2 for second-up). Some horses perform better/worse when fresh - their career first-up record and past first-up ratings tell you this.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form - horses don't always try
- If a horse has 0 race runs, they are UNKNOWN (first starter)
- If 50%+ of field has no race form, pick 0 contenders - too many unknowns to assess

You also have: prep run number, barrier, weight, speedmap/pace data, win/place odds, jockey/trainer A/E ratios, first-up/second-up records, gear changes.

Also include brief notes for non-selected runners explaining why they weren't picked. Focus on:
- Speed ratings at similar distance and condition vs contenders
- Poor jockey/trainer A/E ratios if relevant
- Significant weight changes from recent runs, bad barrier draw
- Lack of form at this distance/condition
Avoid generic career stats like "poor strike rate" - be specific to today's race.

## Output

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "Each-way chance" | "Value bet",
      "analysis": "2-3 sentences referencing RACE form",
      "tipsheet_pick": true | false
    }
  ],
  "other_chances": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "rating": "101.2 at 1200m S5",
      "issue": "Wide barrier + poor trainer A/E (0.67)"
    }
  ],
  "less_likely": ["Horse A", "Horse B"],
  "summary": "Brief overview or reason for 0 picks"
}
```

**other_chances**: Horses with competitive ratings that COULD win but have issues (barrier, jockey/trainer A/E, weight, limited form). Good for bonus bets.
**less_likely**: Horses with weaker ratings or clear issues - just list names.

**tipsheet_pick = true** when you would genuinely bet on this horse yourself:
- Speed ratings clearly support this horse vs the field at this distance and condition
- The odds represent real value
- You're confident in the pick (requires most of the field to have sufficient form data)"""


# TEST PROMPT - New version with improvements
TEST_SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout
- **"Each-way chance"** - Good ratings at similar DISTANCE and CONDITIONS vs the field, place odds $1.80+
- **"Value bet"** - Odds better than their form suggests

**Pick 0 contenders (no bet) when:**
- A lot of field has no race form (only trials) - you can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

## Key Analysis

Focus on **normalized speed ratings** from RACE runs (not trials) at SIMILAR DISTANCE and CONDITIONS to the race being predicted. **Speed ratings matter more than last start wins or career win/place stats.**

**Relevance guide:**
- Exact condition match (S5→S5) + ±10% distance = strongest comparison
- Close condition (±2 levels) + ±20% distance = good comparison
- Weight recent runs more heavily

**First-up/Second-up horses:** Check their past run ratings at the same prep stage (Prep=1 for first-up, Prep=2 for second-up).

**Critical:**
- Barrier trials (marked TRIAL) don't count as form - horses don't always try
- If a horse has 0 race runs, they are UNKNOWN (first starter)
- If 50%+ of field has no race form, pick 0 contenders - too many unknowns to assess

You also have: win/place odds, jockey/trainer A/E ratios, career record, first-up/second-up records, prep run number, barrier, weight, speedmap/pace data, gear changes.

Also include brief notes for non-selected runners explaining why they weren't picked. Focus on:
- Speed ratings at similar distance AND condition vs contenders
- Poor jockey/trainer A/E ratios if relevant
- Significant weight changes from recent runs, bad barrier draw
- Lack of form at this distance/condition
Avoid generic career stats like "poor strike rate" - be specific to today's race.

## Output

First, briefly note which 2-4 horses have the best speed ratings at similar distance and condition. Then output your JSON.

```json
{
  "top_by_ratings": "Horse A (1.02 at 1200m S5), Horse B (1.01 at 1100m G4), ...",
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "Each-way chance" | "Value bet",
      "analysis": "2-3 sentences referencing RACE form",
      "tipsheet_pick": true | false
    }
  ],
  "other_chances": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "rating": "101.2 at 1200m S5",
      "issue": "Wide barrier + poor trainer A/E (0.67)"
    }
  ],
  "less_likely": ["Horse A", "Horse B"],
  "summary": "Brief overview or reason for 0 picks"
}
```

**other_chances**: Horses with competitive ratings that COULD win but have issues (barrier, jockey/trainer A/E, weight, limited form). Good for bonus bets.
**less_likely**: Horses with weaker ratings or clear issues - just list names.

**tipsheet_pick = true** when you would genuinely bet on this horse yourself:
- Speed ratings clearly support this horse vs the field at this distance AND condition
- The odds represent real value
- You're confident in the pick (requires most of the field to have sufficient form data)"""


USER_PROMPT_TEMPLATE = """Analyze this race and pick your contenders (0-3).

{race_data}

Respond with JSON only."""


# =============================================================================
# PREDICTION RUNNER
# =============================================================================

@dataclass
class ComparisonResult:
    """Result from running both prompts on a race."""
    track: str
    race_number: int
    distance: int
    condition: str
    condition_num: int

    live_raw: str
    live_contenders: list
    live_summary: str
    live_runner_notes: dict
    live_other_chances: list
    live_less_likely: list

    test_raw: str
    test_contenders: list
    test_summary: str
    test_runner_notes: dict
    test_top_by_ratings: str
    test_other_chances: list
    test_less_likely: list

    error: Optional[str] = None


def run_prediction(race_data: RaceData, system_prompt: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Run a single prediction with given prompt."""
    client = anthropic.Anthropic()

    race_text = race_data.to_prompt_text()
    user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text

    # Parse JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except json.JSONDecodeError:
        data = {}

    return {
        "raw": raw,
        "contenders": data.get("contenders", []),
        "summary": data.get("summary", ""),
        "runner_notes": data.get("runner_notes", {}),
        "top_by_ratings": data.get("top_by_ratings", ""),
        "other_chances": data.get("other_chances", []),
        "less_likely": data.get("less_likely", []),
    }


def compare_race(track: str, race_number: int, date: str, allow_finished: bool = False) -> ComparisonResult:
    """Run both prompts on same race and return comparison."""
    pipeline = RaceDataPipeline()

    # Get race data (same for both)
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=allow_finished)

    if error:
        return ComparisonResult(
            track=track,
            race_number=race_number,
            distance=0,
            condition="",
            condition_num=0,
            live_raw="",
            live_contenders=[],
            live_summary="",
            live_runner_notes={},
            live_other_chances=[],
            live_less_likely=[],
            test_raw="",
            test_contenders=[],
            test_summary="",
            test_runner_notes={},
            test_top_by_ratings="",
            test_other_chances=[],
            test_less_likely=[],
            error=error,
        )

    # Run both predictions in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        live_future = executor.submit(run_prediction, race_data, LIVE_SYSTEM_PROMPT)
        test_future = executor.submit(run_prediction, race_data, TEST_SYSTEM_PROMPT)

        live_result = live_future.result()
        test_result = test_future.result()

    return ComparisonResult(
        track=track,
        race_number=race_number,
        distance=race_data.distance,
        condition=race_data.condition,
        condition_num=race_data.condition_num,
        live_raw=live_result["raw"],
        live_contenders=live_result["contenders"],
        live_summary=live_result["summary"],
        live_runner_notes=live_result["runner_notes"],
        live_other_chances=live_result["other_chances"],
        live_less_likely=live_result["less_likely"],
        test_raw=test_result["raw"],
        test_contenders=test_result["contenders"],
        test_summary=test_result["summary"],
        test_runner_notes=test_result["runner_notes"],
        test_top_by_ratings=test_result["top_by_ratings"],
        test_other_chances=test_result["other_chances"],
        test_less_likely=test_result["less_likely"],
    )


def compare_meeting(track: str, date: str, allow_finished: bool = False) -> list[ComparisonResult]:
    """Run comparison on all races at a meeting."""
    pf_api = PuntingFormAPI()
    meetings = pf_api.get_meetings(date)

    # Find meeting
    meeting_id = None
    for m in meetings:
        if m.get("track", {}).get("name", "").lower() == track.lower():
            meeting_id = m.get("meetingId")
            break

    if not meeting_id:
        print(f"Meeting not found: {track} on {date}")
        return []

    # Get race count
    form_data = pf_api.get_form(meeting_id, 1, 10)
    if not form_data:
        print(f"Could not get form data for {track}")
        return []

    # Check up to 12 races, will skip ones that error
    race_numbers = list(range(1, 13))

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(compare_race, track, r, date, allow_finished): r
            for r in race_numbers
        }

        for future in as_completed(futures):
            race_num = futures[future]
            try:
                result = future.result()
                if not result.error:
                    results.append(result)
                    print(f"  R{result.race_number}: Done")
                elif "not found" not in result.error.lower():
                    print(f"  R{race_num}: {result.error}")
            except Exception as e:
                print(f"  R{race_num}: Error - {e}")

    # Sort by race number
    results.sort(key=lambda x: x.race_number)
    return results


# =============================================================================
# HTML OUTPUT
# =============================================================================

def generate_html(results: list[ComparisonResult], track: str, date: str) -> str:
    """Generate HTML comparison page."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prompt Comparison: {track} - {date}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .race {{
            margin-bottom: 40px;
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
        }}
        .race-header {{
            background: #0f3460;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .race-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #fff;
        }}
        .race-info {{
            color: #888;
            font-size: 0.9em;
        }}
        .columns {{
            display: grid;
            grid-template-columns: 1fr 1fr;
        }}
        .column {{
            padding: 20px;
            border-right: 1px solid #0f3460;
        }}
        .column:last-child {{ border-right: none; }}
        .column-header {{
            font-size: 0.85em;
            font-weight: bold;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }}
        .live {{ background: #1a1a2e; }}
        .test {{ background: #1e2a3a; }}
        .contender {{
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
        }}
        .contender-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .horse-name {{
            font-weight: bold;
            font-size: 1.1em;
            color: #fff;
        }}
        .odds {{
            color: #4ade80;
            font-weight: bold;
        }}
        .tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-bottom: 8px;
        }}
        .tag-beat {{ background: #854d0e; color: #fef08a; }}
        .tag-eachway {{ background: #1e40af; color: #93c5fd; }}
        .tag-value {{ background: #166534; color: #86efac; }}
        .analysis {{
            color: #ccc;
            font-size: 0.9em;
            line-height: 1.5;
        }}
        .tipsheet {{
            color: #fbbf24;
            font-size: 0.85em;
            margin-top: 8px;
        }}
        .summary {{
            color: #888;
            font-style: italic;
            padding: 10px 0;
            font-size: 0.9em;
        }}
        .top-ratings {{
            background: #0d2137;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 15px;
            font-size: 0.85em;
            color: #67e8f9;
        }}
        .top-ratings-label {{
            font-weight: bold;
            color: #888;
            margin-bottom: 5px;
        }}
        .runner-notes {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #0f3460;
        }}
        .runner-notes-header {{
            font-size: 0.8em;
            color: #666;
            margin-bottom: 8px;
        }}
        .runner-note {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 4px;
        }}
        .runner-note strong {{
            color: #aaa;
        }}
        .no-picks {{
            color: #888;
            font-style: italic;
            padding: 20px 0;
        }}
        .other-chances {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #0f3460;
        }}
        .other-chances-header {{
            font-size: 0.85em;
            font-weight: bold;
            color: #f59e0b;
            margin-bottom: 10px;
        }}
        .other-chance {{
            background: #1e3a5f;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid #f59e0b;
        }}
        .other-chance-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .other-chance-horse {{
            font-weight: bold;
            color: #fbbf24;
        }}
        .other-chance-odds {{
            color: #4ade80;
            font-size: 0.9em;
        }}
        .other-chance-rating {{
            color: #67e8f9;
            font-size: 0.85em;
        }}
        .other-chance-issue {{
            color: #f87171;
            font-size: 0.85em;
            margin-top: 4px;
        }}
        .no-chance {{
            margin-top: 10px;
            padding: 8px 12px;
            background: #1a1a2e;
            border-radius: 6px;
            font-size: 0.8em;
            color: #666;
        }}
        .no-chance-label {{
            color: #888;
            margin-right: 8px;
        }}
        .diff-marker {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-left: 8px;
        }}
        .diff-same {{ background: #4ade80; }}
        .diff-different {{ background: #f87171; }}
        .stats {{
            background: #0f3460;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #fff;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>Prompt Comparison</h1>
    <p class="subtitle">{track} - {date}</p>
"""

    # Calculate stats
    total_races = len(results)
    same_top_pick = 0
    live_picks = 0
    test_picks = 0

    for r in results:
        if r.live_contenders:
            live_picks += len(r.live_contenders)
        if r.test_contenders:
            test_picks += len(r.test_contenders)

        # Check if top pick is the same
        if r.live_contenders and r.test_contenders:
            live_top = r.live_contenders[0].get("horse", "").lower()
            test_top = r.test_contenders[0].get("horse", "").lower()
            if live_top == test_top:
                same_top_pick += 1

    html += f"""
    <div class="stats">
        <div>
            <div class="stat-value">{total_races}</div>
            <div class="stat-label">Races</div>
        </div>
        <div>
            <div class="stat-value">{same_top_pick}/{total_races}</div>
            <div class="stat-label">Same Top Pick</div>
        </div>
        <div>
            <div class="stat-value">{live_picks}</div>
            <div class="stat-label">Live Picks</div>
        </div>
        <div>
            <div class="stat-value">{test_picks}</div>
            <div class="stat-label">Test Picks</div>
        </div>
    </div>
"""

    def get_tag_class(tag: str) -> str:
        if "beat" in tag.lower():
            return "tag-beat"
        elif "each" in tag.lower():
            return "tag-eachway"
        else:
            return "tag-value"

    def render_other_chances(other_chances: list, less_likely: list) -> str:
        if not other_chances and not less_likely:
            return ""

        html = '<div class="other-chances">'

        if other_chances:
            html += '<div class="other-chances-header">🎯 Other Chances (Bonus Bet Ideas)</div>'
            for oc in other_chances[:4]:  # Show top 4
                odds = oc.get('odds', 0)
                odds_str = f"${odds:.2f}" if odds else "N/A"
                html += f"""
                <div class="other-chance">
                    <div class="other-chance-header">
                        <span class="other-chance-horse">#{oc.get('tab_no', '?')}. {oc.get('horse', 'Unknown')}</span>
                        <span class="other-chance-odds">{odds_str}</span>
                    </div>
                    <div class="other-chance-rating">📊 {oc.get('rating', 'N/A')}</div>
                    <div class="other-chance-issue">⚠️ {oc.get('issue', 'Unknown issue')}</div>
                </div>
                """

        if less_likely:
            names = ", ".join(less_likely[:6])  # Show first 6
            html += f'<div class="no-chance"><span class="no-chance-label">Less likely:</span>{names}</div>'

        html += '</div>'
        return html

    def render_contenders(contenders: list, runner_notes: dict) -> str:
        if not contenders:
            return '<div class="no-picks">No contenders selected</div>'

        html = ""
        for c in contenders:
            tag_class = get_tag_class(c.get("tag", ""))
            tipsheet = "⭐ Tipsheet Pick" if c.get("tipsheet_pick") else ""

            html += f"""
            <div class="contender">
                <div class="contender-header">
                    <span class="horse-name">#{c.get('tab_no', '?')}. {c.get('horse', 'Unknown')}</span>
                    <span class="odds">${c.get('odds', 0):.2f}</span>
                </div>
                <span class="tag {tag_class}">{c.get('tag', 'Contender')}</span>
                <div class="analysis">{c.get('analysis', '')}</div>
                {f'<div class="tipsheet">{tipsheet}</div>' if tipsheet else ''}
            </div>
            """

        # Add runner notes
        if runner_notes:
            html += '<div class="runner-notes">'
            html += '<div class="runner-notes-header">Other runners:</div>'
            for horse, note in list(runner_notes.items())[:5]:
                html += f'<div class="runner-note"><strong>{horse}:</strong> {note}</div>'
            html += '</div>'

        return html

    for r in results:
        # Check if top pick is same
        same_pick = False
        if r.live_contenders and r.test_contenders:
            live_top = r.live_contenders[0].get("horse", "").lower()
            test_top = r.test_contenders[0].get("horse", "").lower()
            same_pick = live_top == test_top

        diff_class = "diff-same" if same_pick else "diff-different"

        html += f"""
    <div class="race">
        <div class="race-header">
            <span class="race-title">Race {r.race_number} <span class="diff-marker {diff_class}"></span></span>
            <span class="race-info">{r.distance}m | {r.condition}{r.condition_num}</span>
        </div>
        <div class="columns">
            <div class="column live">
                <div class="column-header">Live Prompt</div>
                {render_contenders(r.live_contenders, {})}
                {render_other_chances(r.live_other_chances, r.live_less_likely)}
                <div class="summary">{r.live_summary}</div>
            </div>
            <div class="column test">
                <div class="column-header">Test Prompt</div>
                {f'<div class="top-ratings"><div class="top-ratings-label">Top by ratings:</div>{r.test_top_by_ratings}</div>' if r.test_top_by_ratings else ''}
                {render_contenders(r.test_contenders, {})}
                {render_other_chances(r.test_other_chances, r.test_less_likely)}
                <div class="summary">{r.test_summary}</div>
            </div>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""
    return html


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare two prompts on same race data")
    parser.add_argument("--track", required=True, help="Track name")
    parser.add_argument("--race", type=int, help="Race number (omit for all races)")
    parser.add_argument("--date", required=True, help="Date (dd-MMM-yyyy)")
    parser.add_argument("--all", action="store_true", help="Run all races at meeting")
    parser.add_argument("--past", action="store_true", help="Allow finished races (for backtesting)")
    parser.add_argument("--output", help="Output HTML file path")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PROMPT COMPARISON: {args.track} - {args.date}")
    print(f"{'='*60}\n")

    if args.all or not args.race:
        print("Running comparison on all races...")
        if args.past:
            print("(Using past race mode - SP odds)")
        results = compare_meeting(args.track, args.date, allow_finished=args.past)
    else:
        print(f"Running comparison on Race {args.race}...")
        if args.past:
            print("(Using past race mode - SP odds)")
        result = compare_race(args.track, args.race, args.date, allow_finished=args.past)
        results = [result] if not result.error else []
        if result.error:
            print(f"Error: {result.error}")

    if not results:
        print("No results to display")
        return

    # Generate HTML
    html = generate_html(results, args.track, args.date)

    # Save to file
    output_file = args.output or f"comparison_{args.track.lower().replace(' ', '_')}_{args.date.replace('-', '')}.html"
    output_dir = os.path.join(os.path.dirname(__file__), "html_reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"  Output saved to: {output_path}")
    print(f"{'='*60}\n")

    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
