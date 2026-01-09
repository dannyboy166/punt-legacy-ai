"""
Claude AI Predictor for Horse Racing.

Uses Claude API to analyze race data and pick winners.

Usage:
    from core.predictor import Predictor

    predictor = Predictor()
    prediction = predictor.predict(race_data)

    print(f"Top pick: {prediction.top_pick} @ ${prediction.top_pick_odds}")
    print(f"Win probability: {prediction.win_probability}%")
    print(f"BET: {'YES' if prediction.bet_recommendation else 'NO'}")
    print(f"Reasoning: {prediction.reasoning}")
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from core.race_data import RaceData
from core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

# Default model
DEFAULT_MODEL = "claude-sonnet-4-20250514"


# =============================================================================
# PREDICTION RESULT
# =============================================================================

@dataclass
class Contender:
    """A horse identified as a genuine winning chance."""

    horse: str
    tab_no: int
    odds: float
    chance: str  # "best", "solid", "each-way"
    analysis: str  # Natural language analysis of the horse and price

    def to_dict(self) -> dict:
        return {
            "horse": self.horse,
            "tab_no": self.tab_no,
            "odds": self.odds,
            "chance": self.chance,
            "analysis": self.analysis,
        }


@dataclass
class PredictionOutput:
    """Output from Claude predictor."""

    # 1-3 contenders
    contenders: list[Contender] = field(default_factory=list)

    # Overall summary
    summary: str = ""

    # Race context
    track: str = ""
    race_number: int = 0

    # Metadata
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "contenders": [c.to_dict() for c in self.contenders],
            "summary": self.summary,
            "track": self.track,
            "race_number": self.race_number,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert horse racing analyst specializing in Australian thoroughbred racing.

Your task is to identify the CONTENDERS (1-3 horses that could realistically win) and give your thoughts on each.

## Your Approach

**Step 1: Identify contenders.** Which horses could realistically WIN this race? Usually 1-3 horses.

**Step 2: Rank them.** Who's the best horse? Who else has a genuine chance?

**Step 3: Give your view.** For each contender, explain why they can win and your thoughts on the price.

## Key Concepts

**Speed Ratings**: Normalized performance measure.
- Rating = 1.000 means average performance for that distance/condition
- Rating > 1.000 means faster than expected, < 1.000 means slower
- **CRITICAL: Compare ratings WITHIN THIS FIELD** - the highest-rated horse is best in the field

**Prep Run (1st up, 2nd up, etc.)**: The "Prep" column shows which run in the prep.
- Prep=1 means first-up, Prep=2 means second-up
- Look for patterns: does the horse improve 2nd-up? Are they better fresh?

**A/E (Actual vs Expected)**: Measures if jockey/trainer outperforms market expectations.
- A/E > 1.0 = consistently beats market (positive signal)
- A/E < 1.0 = underperforms market (negative signal)

## Output Format

Return 1-3 contenders. You MUST respond with a JSON object in this exact format:

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "chance": "best" / "solid" / "each-way",
      "analysis": "2-3 sentences: why this horse can win, and your thoughts on the price. Be natural - you can say things like 'looks good value', 'short price for what you're getting', 'worth a small each-way', 'standout on the ratings', etc. Don't use explicit percentages."
    }
  ],
  "summary": "1-2 sentences summarizing the race overall"
}
```

**Chance levels:**
- "best" = Most likely winner
- "solid" = Genuine winning chance
- "each-way" = Could win if things go right

## Important Rules

1. Only include horses that could realistically WIN - not place hopes or longshot lottery tickets
2. Maximum 3 contenders per race
3. Be natural in your analysis - give your honest view on each horse and the price
4. You can mention if you like the value, if it's short, if it's worth a small bet, etc. - use your judgment
5. Don't use explicit percentages or rigid "bet/no bet" language
6. Be honest: some races only have 1 real contender, others have 2-3"""


USER_PROMPT_TEMPLATE = """Analyze this race and identify the contenders.

{race_data}

Remember:
- Identify 1-3 horses that could realistically WIN
- For each, say if it's a bet or not based on value
- Respond with valid JSON only"""


# =============================================================================
# PREDICTOR
# =============================================================================

class Predictor:
    """
    Claude AI predictor for horse racing value bets.

    Analyzes race data and identifies betting opportunities where
    the estimated probability exceeds the implied probability from odds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize predictor.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def predict(
        self,
        race_data: RaceData,
        custom_instructions: Optional[str] = None,
    ) -> PredictionOutput:
        """
        Analyze race and predict value bets.

        Args:
            race_data: Complete race data from RaceDataPipeline
            custom_instructions: Optional additional instructions for Claude

        Returns:
            PredictionOutput with selection and reasoning
        """
        # Format race data for prompt
        race_text = race_data.to_prompt_text()

        # Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

        if custom_instructions:
            user_prompt += f"\n\nAdditional instructions: {custom_instructions}"

        # Call Claude
        logger.info(f"Calling Claude for {race_data.track} R{race_data.race_number}")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            raw_response = response.content[0].text
            logger.debug(f"Raw response: {raw_response}")

        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return PredictionOutput(
                reasoning=f"API error: {str(e)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        # Parse response
        return self._parse_response(
            raw_response,
            race_data,
        )

    def _parse_response(
        self,
        raw_response: str,
        race_data: RaceData,
    ) -> PredictionOutput:
        """Parse Claude's JSON response into PredictionOutput."""

        # Try to extract JSON from response
        try:
            # Handle case where response might have text before/after JSON
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return PredictionOutput(
                reasoning=f"Failed to parse response: {raw_response[:200]}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
                raw_response=raw_response,
            )

        # Parse contenders
        contenders = []
        for c in data.get("contenders", []):
            horse = c.get("horse")
            tab_no = c.get("tab_no")
            odds = c.get("odds")

            # Look up odds from race data if not provided
            if horse and not odds:
                for runner in race_data.runners:
                    if runner.name.lower() == horse.lower() or runner.tab_no == tab_no:
                        odds = runner.odds
                        tab_no = runner.tab_no
                        horse = runner.name  # Canonical name
                        break

            if horse and tab_no and odds:
                contenders.append(Contender(
                    horse=horse,
                    tab_no=tab_no,
                    odds=odds,
                    chance=c.get("chance", "solid"),
                    analysis=c.get("analysis", ""),
                ))

        return PredictionOutput(
            contenders=contenders,
            summary=data.get("summary", ""),
            track=race_data.track,
            race_number=race_data.race_number,
            model=self.model,
            raw_response=raw_response,
        )

    def predict_meeting(
        self,
        races: list[RaceData],
        custom_instructions: Optional[str] = None,
    ) -> list[PredictionOutput]:
        """
        Analyze all races at a meeting.

        Args:
            races: List of RaceData for all races
            custom_instructions: Optional additional instructions

        Returns:
            List of PredictionOutput for each race
        """
        predictions = []

        for race in races:
            prediction = self.predict(race, custom_instructions)
            predictions.append(prediction)

            if prediction.contenders:
                top = prediction.contenders[0]
                logger.info(
                    f"{race.track} R{race.race_number}: "
                    f"{top.horse} @ ${top.odds} ({top.chance})"
                )

        return predictions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_race(
    track: str,
    race_number: int,
    date: str,
    custom_instructions: Optional[str] = None,
) -> PredictionOutput:
    """
    Convenience function to analyze a single race.

    Args:
        track: Track name
        race_number: Race number
        date: Date in PuntingForm format (dd-MMM-yyyy)
        custom_instructions: Optional additional instructions

    Returns:
        PredictionOutput
    """
    from core.race_data import RaceDataPipeline

    # Get race data
    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_number, date)

    if error:
        return PredictionOutput(
            reasoning=f"Failed to get race data: {error}",
            track=track,
            race_number=race_number,
        )

    # Run prediction
    predictor = Predictor()
    return predictor.predict(race_data, custom_instructions)
