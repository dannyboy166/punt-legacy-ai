"""
Experimental Bet Type Predictor (v2).

A simplified predictor that:
- Picks 0-3 contenders per race (not forced to always pick)
- Skips races where 50%+ of field has no race form (only trials)
- Uses 3 fixed tags for clarity
- Focuses on normalized speed ratings from RACE runs (not trials)

Tags:
- "The one to beat" - Clear standout
- "Each-way chance" - Could win, should place, place odds worth it
- "Value bet" - Odds better than form suggests

When it returns 0 picks:
- Too many first starters (50%+ with no race form)
- Field too even, no standouts
- Insufficient data to assess

Usage:
    # Single race (live)
    python experiments/bet_type_predictor.py --track "Randwick" --race 4 --date "19-Jan-2026"

    # All races at a track
    python experiments/bet_type_predictor.py --track "Randwick" --race 1 --date "19-Jan-2026" --all

For backtesting finished races, use experiments/backtest.py instead.
"""

import os
import sys
import json
import re
import argparse
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from dotenv import load_dotenv

from core.race_data import RaceDataPipeline, RaceData
from core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


@dataclass
class Contender:
    """A single contender pick."""
    horse: str
    tab_no: int
    odds: float
    place_odds: float
    tag: str  # "The one to beat", "Each-way chance", "Value bet"
    analysis: str


@dataclass
class BetRecommendation:
    """The AI's recommendation for a race."""
    contenders: list[Contender]  # 0-3 contenders
    summary: str  # Brief race overview
    raw_response: str = ""


SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout
- **"Each-way chance"** - Could win, should place, place odds worth it
- **"Value bet"** - Odds better than their form suggests

**Pick 0 contenders (no bet) when:**
- A lot of field has no race form (only trials) - you can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

## Key Analysis

Focus on **normalized speed ratings** from RACE runs (not trials) at similar distance and conditions. More recent runs are more relevant.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form - horses don't always try
- If a horse has 0 race runs, they are UNKNOWN - could be brilliant or useless
- If 50%+ of field has no race form, pick 0 contenders - too many unknowns to assess

You also have: win/place odds, jockey/trainer A/E ratios, career record, first-up/second-up records, prep run number, barrier, weight.

## Output

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "place_odds": number,
      "tag": "The one to beat" | "Each-way chance" | "Value bet",
      "analysis": "1-2 sentences referencing RACE form"
    }
  ],
  "summary": "Brief overview or reason for 0 picks"
}
```
"""


USER_PROMPT_TEMPLATE = """Analyze this race and pick your contenders (0-3).

{race_data}

Respond with JSON only."""


class BetTypePredictor:
    """Experimental predictor that recommends bet types based on form analysis."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze(self, race_data: RaceData) -> BetRecommendation:
        """Analyze a race and get bet recommendation."""

        race_text = race_data.to_prompt_text()
        user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

        logger.info(f"Analyzing {race_data.track} R{race_data.race_number} for bet type...")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )

            raw_response = response.content[0].text
            return self._parse_response(raw_response)

        except Exception as e:
            logger.error(f"API error: {e}")
            return BetRecommendation(
                contenders=[],
                summary=f"Error: {e}",
                raw_response=""
            )

    def _parse_response(self, raw_response: str) -> BetRecommendation:
        """Parse Claude's JSON response."""

        try:
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")

            contenders = []
            for c in data.get("contenders", []):
                contenders.append(Contender(
                    horse=c.get("horse", ""),
                    tab_no=c.get("tab_no", 0),
                    odds=c.get("odds", 0),
                    place_odds=c.get("place_odds", 0),
                    tag=c.get("tag", ""),
                    analysis=c.get("analysis", "")
                ))

            return BetRecommendation(
                contenders=contenders,
                summary=data.get("summary", ""),
                raw_response=raw_response
            )

        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return BetRecommendation(
                contenders=[],
                summary=f"Parse error: {raw_response[:200]}",
                raw_response=raw_response
            )


def print_recommendation(rec: BetRecommendation, track: str, race_num: int):
    """Pretty print a bet recommendation."""

    print("\n" + "=" * 60)
    print(f"  {track} RACE {race_num}")
    print("=" * 60)

    if not rec.contenders:
        print("\n  ❌ NO CONTENDERS")
        print(f"\n  {rec.summary}")
    else:
        print(f"\n  {len(rec.contenders)} CONTENDER(S):\n")
        for c in rec.contenders:
            print(f"  {c.horse} (#{c.tab_no})")
            print(f"    ${c.odds:.2f} win / ${c.place_odds:.2f} place")
            print(f"    \"{c.tag}\"")
            print(f"    {c.analysis}\n")

        print(f"  SUMMARY: {rec.summary}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experimental Bet Type Predictor")
    parser.add_argument("--track", required=True, help="Track name")
    parser.add_argument("--race", type=int, required=True, help="Race number")
    parser.add_argument("--date", required=True, help="Date (dd-MMM-yyyy)")
    parser.add_argument("--all", action="store_true", help="Analyze all races at track")

    args = parser.parse_args()

    # Get race data
    pipeline = RaceDataPipeline()
    predictor = BetTypePredictor()

    if args.all:
        # Analyze all races at the track
        print(f"\nAnalyzing all races at {args.track} on {args.date}...")

        # We'd need to get race count first - for now just do races 1-10
        for race_num in range(1, 11):
            race_data, error = pipeline.get_race_data(args.track, race_num, args.date)
            if error:
                print(f"\nRace {race_num}: {error}")
                continue

            rec = predictor.analyze(race_data)
            print_recommendation(rec, args.track, race_num)
    else:
        # Single race
        race_data, error = pipeline.get_race_data(args.track, args.race, args.date)

        if error:
            print(f"Error: {error}")
            return

        rec = predictor.analyze(race_data)
        print_recommendation(rec, args.track, args.race)


if __name__ == "__main__":
    main()
