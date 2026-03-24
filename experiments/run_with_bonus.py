"""
Run live predictor with bonus bet extras.

Just runs core/predictor.py prompt + adds other_chances output.

Usage:
    python experiments/run_with_bonus.py --track "Rosehill" --date "21-Feb-2026" --all --past
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.race_data import RaceDataPipeline, RaceData
from api.puntingform import PuntingFormAPI

load_dotenv()

# Read current live prompt from core/predictor.py and add bonus bet output
SYSTEM_PROMPT = """You are an expert horse racing analyst.

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
      "issue": "Brief reason this horse has issues vs contenders"
    }
  ],
  "less_likely": ["Horse A", "Horse B"],
  "summary": "Brief overview or reason for 0 picks"
}
```

**other_chances**: Horses with competitive ratings that COULD win but have issues preventing them being top contenders. Good value for bonus bets.
**less_likely**: Horses with weaker ratings or clear issues - just list names.

**tipsheet_pick = true** when you would genuinely bet on this horse yourself:
- Speed ratings clearly support this horse vs the field at this distance and condition
- The odds represent real value
- You're confident in the pick (requires most of the field to have sufficient form data)"""


USER_PROMPT = """Analyze this race and pick your contenders (0-3).

{race_data}

Respond with JSON only."""


@dataclass
class RaceResult:
    track: str
    race_number: int
    distance: int
    condition: str
    condition_num: int
    contenders: list
    other_chances: list
    less_likely: list
    summary: str
    raw: str
    error: Optional[str] = None


def run_prediction(race_data: RaceData) -> dict:
    client = anthropic.Anthropic()
    race_text = race_data.to_prompt_text()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT.format(race_data=race_text)}],
    )

    raw = response.content[0].text

    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        data = json.loads(json_match.group()) if json_match else {}
    except json.JSONDecodeError:
        data = {}

    return {
        "raw": raw,
        "contenders": data.get("contenders", []),
        "other_chances": data.get("other_chances", []),
        "less_likely": data.get("less_likely", []),
        "summary": data.get("summary", ""),
    }


def analyze_race(track: str, race_number: int, date: str, allow_finished: bool = False) -> RaceResult:
    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=allow_finished)

    if error:
        return RaceResult(
            track=track, race_number=race_number, distance=0, condition="", condition_num=0,
            contenders=[], other_chances=[], less_likely=[], summary="", raw="", error=error
        )

    result = run_prediction(race_data)

    return RaceResult(
        track=track,
        race_number=race_number,
        distance=race_data.distance,
        condition=race_data.condition,
        condition_num=race_data.condition_num,
        contenders=result["contenders"],
        other_chances=result["other_chances"],
        less_likely=result["less_likely"],
        summary=result["summary"],
        raw=result["raw"],
    )


def analyze_meeting(track: str, date: str, allow_finished: bool = False) -> list[RaceResult]:
    pf_api = PuntingFormAPI()
    meetings = pf_api.get_meetings(date)

    meeting = next((m for m in meetings if m.get("track", {}).get("name", "").lower() == track.lower()), None)
    if not meeting:
        print(f"Meeting not found: {track} on {date}")
        return []

    race_numbers = list(range(1, 13))
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_race, track, r, date, allow_finished): r for r in race_numbers}

        for future in as_completed(futures):
            race_num = futures[future]
            try:
                result = future.result()
                if not result.error:
                    results.append(result)
                    print(f"  R{result.race_number}: Done")
            except Exception as e:
                print(f"  R{race_num}: Error - {e}")

    results.sort(key=lambda x: x.race_number)
    return results


def generate_html(results: list[RaceResult], track: str, date: str) -> str:
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Live Predictor + Bonus: {track} - {date}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ text-align: center; color: #fff; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .race {{ margin-bottom: 30px; background: #16213e; border-radius: 12px; overflow: hidden; max-width: 800px; margin-left: auto; margin-right: auto; }}
        .race-header {{ background: #0f3460; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }}
        .race-title {{ font-size: 1.3em; font-weight: bold; color: #fff; }}
        .race-info {{ color: #888; font-size: 0.9em; }}
        .race-body {{ padding: 20px; }}
        .contender {{ background: #0f3460; border-radius: 8px; padding: 15px; margin-bottom: 12px; }}
        .contender-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .horse-name {{ font-weight: bold; font-size: 1.1em; color: #fff; }}
        .odds {{ color: #4ade80; font-weight: bold; }}
        .tag {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-bottom: 8px; }}
        .tag-beat {{ background: #854d0e; color: #fef08a; }}
        .tag-eachway {{ background: #1e40af; color: #93c5fd; }}
        .tag-value {{ background: #166534; color: #86efac; }}
        .analysis {{ color: #ccc; font-size: 0.9em; line-height: 1.5; }}
        .tipsheet {{ color: #fbbf24; font-size: 0.85em; margin-top: 8px; }}
        .summary {{ color: #888; font-style: italic; padding: 10px 0; font-size: 0.9em; }}
        .section-header {{ font-size: 0.85em; font-weight: bold; color: #f59e0b; margin: 15px 0 10px 0; padding-top: 15px; border-top: 1px solid #0f3460; }}
        .other-chance {{ background: #1e3a5f; border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; border-left: 3px solid #f59e0b; }}
        .other-chance-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
        .other-chance-horse {{ font-weight: bold; color: #fbbf24; }}
        .other-chance-odds {{ color: #4ade80; font-size: 0.9em; }}
        .other-chance-rating {{ color: #67e8f9; font-size: 0.85em; }}
        .other-chance-issue {{ color: #f87171; font-size: 0.85em; margin-top: 4px; }}
        .less-likely {{ margin-top: 10px; padding: 8px 12px; background: #1a1a2e; border-radius: 6px; font-size: 0.8em; color: #666; }}
        .stats {{ background: #0f3460; border-radius: 12px; padding: 20px; margin-bottom: 30px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center; max-width: 600px; margin-left: auto; margin-right: auto; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #fff; }}
        .stat-label {{ font-size: 0.85em; color: #888; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>Live Predictor + Bonus Picks</h1>
    <p class="subtitle">{track} - {date}</p>
"""

    total_contenders = sum(len(r.contenders) for r in results)
    total_other = sum(len(r.other_chances) for r in results)

    html += f"""
    <div class="stats">
        <div><div class="stat-value">{len(results)}</div><div class="stat-label">Races</div></div>
        <div><div class="stat-value">{total_contenders}</div><div class="stat-label">Contenders</div></div>
        <div><div class="stat-value">{total_other}</div><div class="stat-label">Bonus Picks</div></div>
    </div>
"""

    def get_tag_class(tag):
        if "beat" in tag.lower(): return "tag-beat"
        elif "each" in tag.lower(): return "tag-eachway"
        return "tag-value"

    for r in results:
        html += f"""
    <div class="race">
        <div class="race-header">
            <span class="race-title">Race {r.race_number}</span>
            <span class="race-info">{r.distance}m | {r.condition}{r.condition_num}</span>
        </div>
        <div class="race-body">
"""

        if r.contenders:
            for c in r.contenders:
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
        else:
            html += '<div class="summary">No contenders selected</div>'

        if r.other_chances:
            html += '<div class="section-header">🎯 Bonus Bet Ideas</div>'
            for oc in r.other_chances:
                odds = oc.get('odds', 0)
                html += f"""
            <div class="other-chance">
                <div class="other-chance-header">
                    <span class="other-chance-horse">#{oc.get('tab_no', '?')}. {oc.get('horse', 'Unknown')}</span>
                    <span class="other-chance-odds">${odds:.2f}</span>
                </div>
                <div class="other-chance-rating">📊 {oc.get('rating', 'N/A')}</div>
                <div class="other-chance-issue">⚠️ {oc.get('issue', '')}</div>
            </div>
"""

        if r.less_likely:
            html += f'<div class="less-likely"><strong>Less likely:</strong> {", ".join(r.less_likely[:6])}</div>'

        html += f'<div class="summary">{r.summary}</div></div></div>'

    html += "</body></html>"
    return html


def main():
    parser = argparse.ArgumentParser(description="Run live predictor with bonus bet extras")
    parser.add_argument("--track", required=True)
    parser.add_argument("--race", type=int)
    parser.add_argument("--date", required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--past", action="store_true")
    parser.add_argument("--output", help="Output file")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LIVE PREDICTOR + BONUS: {args.track} - {args.date}")
    print(f"{'='*60}\n")

    if args.all or not args.race:
        print("Running all races...")
        results = analyze_meeting(args.track, args.date, allow_finished=args.past)
    else:
        print(f"Running Race {args.race}...")
        result = analyze_race(args.track, args.race, args.date, allow_finished=args.past)
        results = [result] if not result.error else []

    if not results:
        print("No results")
        return

    html = generate_html(results, args.track, args.date)
    output_file = args.output or f"live_bonus_{args.track.lower().replace(' ', '_')}_{args.date.replace('-', '')}.html"
    output_dir = os.path.join(os.path.dirname(__file__), "html_reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
