"""
Experimental Bet Type Predictor.

This is a research module to test a different approach:
Instead of "pick the contenders", we ask "would you bet? if so, what type?"

Bet types:
- WIN: Clear standout horse with good value
- EACH-WAY: Horse likely to place but not necessarily win
- EXACTA: Two horses clearly above the rest
- TRIFECTA: Three horses clearly above the rest
- QUINELLA: Two horses above rest, either could win
- NO BET: Race too open, not enough form, or no value

Usage:
    python experiments/bet_type_predictor.py --track "Randwick" --race 4 --date "19-Jan-2026"
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
class BetRecommendation:
    """The AI's bet recommendation for a race."""

    would_bet: bool
    bet_type: Optional[str]  # WIN, EACH_WAY, EXACTA, TRIFECTA, QUINELLA, or None

    # Selections (depends on bet type)
    selections: list[dict]  # [{horse, tab_no, odds, role}] - role = "win", "place", "1st leg", etc.

    # Reasoning
    reasoning: str  # Why this bet type (or why no bet)
    form_analysis: str  # Key form factors that led to decision

    # Confidence
    confidence: int  # 1-10

    # Raw data
    raw_response: str = ""


SYSTEM_PROMPT = """You are an expert horse racing analyst and betting strategist.

Your task: Analyze this race and decide IF you would bet, and if so, WHAT TYPE of bet.

## Bet Types to Consider

1. **WIN**: Use when ONE horse clearly stands out
   - Superior speed ratings at similar distance/conditions
   - Strong recent form, good prep pattern
   - Suitable conditions for this horse

2. **EACH-WAY**: Use when a horse should place but winning is uncertain
   - Consistent placer with good speed ratings
   - Place odds offer value ($1.80+)
   - Might face one or two superior horses but will run top 3

3. **EXACTA**: Use when TWO horses are clearly above the rest
   - Both have superior ratings at similar conditions
   - Clear gap between these two and the rest of field
   - Order uncertain - could go either way

4. **QUINELLA**: Similar to exacta but when order truly doesn't matter
   - Two standouts, genuinely can't split them
   - Better value than exacta if odds are similar

5. **TRIFECTA**: Use when THREE horses are clearly above the rest
   - Top 3 are obvious, order is the question
   - Clear form gap to 4th and beyond

6. **NO BET**: Use when you SHOULDN'T bet
   - Too many first-uppers with no form
   - Field too even, no standouts
   - Insufficient data to make confident assessment
   - No value in the market

## How to Analyze

Focus on these factors (in order of importance):

1. **Speed Ratings at Similar Conditions**: Ratings from runs at similar distance (within 200m) and track condition are MOST predictive
2. **Recent Form**: Last 2-3 runs matter more than older runs
3. **Prep Pattern**: Where is horse in its campaign? First-up, second-up, third-up?
4. **Class Level**: Is horse up or down in class from recent runs?
5. **Distance Suitability**: Has horse proven at this distance?
6. **Track/Condition Form**: How does horse perform on this track condition?

DO NOT just look at odds to decide bet type. A $2 favorite could be NO BET if form doesn't support it.

## Output Format

```json
{
  "would_bet": true/false,
  "bet_type": "WIN" | "EACH_WAY" | "EXACTA" | "QUINELLA" | "TRIFECTA" | null,
  "selections": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "role": "win" | "place" | "1st leg" | "2nd leg" | "3rd leg"
    }
  ],
  "reasoning": "2-3 sentences on why this bet type (or why no bet)",
  "form_analysis": "Key form factors: mention specific runs, ratings, conditions that support your decision",
  "confidence": number (1-10)
}
```

## Guidelines

- Be HONEST - if you wouldn't bet, say so
- Back up decisions with SPECIFIC form references
- Don't force exotic bets - WIN is fine if one horse stands out
- EACH-WAY requires $1.80+ place odds
- For exotics, you need CLEAR separation between your picks and the rest
- Confidence should reflect how certain you are in the bet type AND selections
"""


USER_PROMPT_TEMPLATE = """Analyze this race and tell me: Would you bet? If so, what type of bet?

{race_data}

Remember: Focus on form analysis, not just odds. Only recommend a bet if the form genuinely supports it.

Respond with valid JSON only."""


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
                would_bet=False,
                bet_type=None,
                selections=[],
                reasoning=f"Error: {e}",
                form_analysis="",
                confidence=0,
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

            return BetRecommendation(
                would_bet=data.get("would_bet", False),
                bet_type=data.get("bet_type"),
                selections=data.get("selections", []),
                reasoning=data.get("reasoning", ""),
                form_analysis=data.get("form_analysis", ""),
                confidence=data.get("confidence", 5),
                raw_response=raw_response
            )

        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return BetRecommendation(
                would_bet=False,
                bet_type=None,
                selections=[],
                reasoning=f"Parse error: {raw_response[:200]}",
                form_analysis="",
                confidence=0,
                raw_response=raw_response
            )


def print_recommendation(rec: BetRecommendation, track: str, race_num: int):
    """Pretty print a bet recommendation."""

    print("\n" + "=" * 60)
    print(f"  {track} RACE {race_num}")
    print("=" * 60)

    if not rec.would_bet:
        print("\n  ❌ NO BET")
        print(f"\n  Reason: {rec.reasoning}")
    else:
        print(f"\n  ✅ BET TYPE: {rec.bet_type}")
        print(f"  Confidence: {rec.confidence}/10")

        print("\n  SELECTIONS:")
        for sel in rec.selections:
            role = sel.get('role', '')
            print(f"    • {sel['horse']} (#{sel['tab_no']}) @ ${sel['odds']:.2f} [{role}]")

        print(f"\n  REASONING: {rec.reasoning}")

    print(f"\n  FORM ANALYSIS: {rec.form_analysis}")
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
