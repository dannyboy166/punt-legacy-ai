"""
Test a simplified predictor prompt - just ask about speed ratings.
"""

import argparse
import os
import json
import re
from dotenv import load_dotenv
import anthropic

from core.race_data import RaceDataPipeline

load_dotenv()

SIMPLE_SYSTEM_PROMPT = """You are a horse racing analyst.

Pick 0-2 horses where:
1. Their speed ratings at similar distances/conditions are among the best in the field
2. The odds represent genuine value (price is worth the risk)

Only pick when BOTH criteria are met. Skip races where no value exists.

Output JSON:
```json
{
  "picks": [
    {
      "horse": "Name",
      "tab_no": number,
      "odds": number,
      "analysis": "Why ratings + odds = value"
    }
  ],
  "summary": "Brief overview"
}
```

Pick 0 if:
- 50%+ have no race form (only trials)
- Best rated horses are too short (no value)
- Field is too even (no standouts)"""


def run_simple_prediction(track: str, race_number: int, date: str, allow_finished: bool = False):
    """Run prediction with simple prompt."""

    # Get race data
    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=allow_finished)

    if error:
        print(f"Error getting race data: {error}")
        return

    # Format race data
    race_text = race_data.to_prompt_text()

    # Call Claude
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    print(f"\n{'='*60}")
    print(f"  {track.upper()} RACE {race_number} - {date}")
    print(f"  {race_data.distance}m | {race_data.condition}")
    print(f"{'='*60}\n")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.2,
        system=SIMPLE_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Analyze this race:\n\n{race_text}\n\nRespond with JSON only."}
        ],
    )

    raw = response.content[0].text

    # Parse JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
        else:
            print("No JSON found")
            print(raw)
            return
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(raw)
        return

    # Display results
    picks = data.get("picks", [])

    if not picks:
        print("  NO PICKS")
        print(f"  Reason: {data.get('summary', 'N/A')}")
    else:
        for i, pick in enumerate(picks, 1):
            print(f"  {i}. {pick['horse']} (#{pick.get('tab_no', '?')}) @ ${pick.get('odds', '?')}")
            print(f"     {pick.get('analysis', '')}")
            print()

    print(f"  Summary: {data.get('summary', '')}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test simple speed rating prompt")
    parser.add_argument("--track", required=True, help="Track name")
    parser.add_argument("--race", type=int, required=True, help="Race number")
    parser.add_argument("--date", required=True, help="Date (dd-MMM-yyyy)")
    parser.add_argument("--allow-finished", action="store_true", help="Allow past/finished races (uses SP odds)")

    args = parser.parse_args()
    run_simple_prediction(args.track, args.race, args.date, allow_finished=args.allow_finished)
