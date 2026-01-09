"""
Claude AI Predictor for Horse Racing.

Uses Claude API to analyze race data and identify value bets.

Usage:
    from core.predictor import Predictor

    predictor = Predictor()
    prediction = predictor.predict(race_data)

    if prediction.has_value_bet:
        print(f"Value bet: {prediction.selection} @ ${prediction.odds}")
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
class PredictionOutput:
    """Output from Claude predictor."""

    # Did Claude find a value bet?
    has_value_bet: bool

    # Selection details (if has_value_bet)
    selection: Optional[str] = None
    tab_no: Optional[int] = None
    odds: Optional[float] = None
    estimated_probability: Optional[float] = None
    implied_probability: Optional[float] = None
    edge: Optional[float] = None  # estimated - implied
    confidence: Optional[str] = None  # "high", "medium", "low"

    # Reasoning
    reasoning: str = ""
    key_factors: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    # Race context
    track: str = ""
    race_number: int = 0

    # Metadata
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "has_value_bet": self.has_value_bet,
            "selection": self.selection,
            "tab_no": self.tab_no,
            "odds": self.odds,
            "estimated_probability": self.estimated_probability,
            "implied_probability": self.implied_probability,
            "edge": round(self.edge, 1) if self.edge else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "concerns": self.concerns,
            "track": self.track,
            "race_number": self.race_number,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert horse racing analyst specializing in Australian thoroughbred racing.

Your task is to analyze race data and identify VALUE BETS - horses where the true probability of winning is HIGHER than what the odds imply.

## Key Concepts

**Implied Probability**: The probability the market assigns based on odds.
- Formula: 100 / odds
- Example: $4.00 odds = 25% implied probability

**Value Bet**: When your estimated probability > implied probability.
- Example: You estimate 30% chance, odds imply 25% = VALUE (+5% edge)
- You should NOT bet if: You estimate 20% chance, odds imply 25% = NO VALUE

**Speed Ratings**: Normalized performance measure.
- Rating = 1.000 means average performance for that distance/condition
- Rating > 1.000 means faster than expected
- Rating < 1.000 means slower than expected
- **CRITICAL: Compare ratings WITHIN THIS FIELD, not against absolute standards**
- Even if all horses are below 1.0, the highest-rated horse is still best in the field
- Someone HAS to win - focus on relative rankings, not absolute numbers

**Prep Run (1st up, 2nd up, etc.)**: The "Prep" column shows which run in the prep each form run was.
- Prep=1 means first-up (resuming from a spell)
- Prep=2 means second-up
- Look for patterns: does the horse improve 2nd-up? Are they better fresh?
- First-up/second-up career records show historical performance at these stages

**A/E (Actual vs Expected)**: Measures if jockey/trainer outperforms market expectations.
- A/E > 1.0 = consistently beats market (positive signal)
- A/E < 1.0 = underperforms market (negative signal)

## Analysis Framework

1. **Form Analysis**: Look at each horse's recent runs
   - Are they improving or declining? (compare ratings across runs)
   - Do they perform better at certain distances/conditions?
   - How does today's race compare to their past races?
   - **Check prep patterns**: Do their 1st-up or 2nd-up runs show better/worse ratings?

2. **Class Assessment**: Is the horse rising, dropping, or staying at same level?
   - Dropping class = positive if form warrants it
   - Rising class = concerning unless progressive improvement shown

3. **Condition Suitability**: Does today's track suit the horse?
   - Filter their form by similar conditions and compare ratings
   - Some horses improve dramatically on wet tracks

4. **Spell/Prep Analysis**:
   - If horse is FIRST UP: check their first-up career record and past 1st-up ratings
   - If horse is SECOND UP: check their second-up career record and past 2nd-up ratings
   - Some horses improve sharply 2nd-up, others are best fresh

5. **Pace Scenario**: Who benefits from the expected pace?
   - Hot pace (3+ leaders) often helps backmarkers
   - Soft pace (0-1 leaders) often helps on-pace runners

6. **Barrier and Weight**: Consider their impact
   - Wide barriers disadvantage in short races
   - Weight matters more in longer races

## Output Format

You MUST respond with a JSON object in this exact format:

```json
{
  "has_value_bet": true/false,
  "selection": "Horse Name" or null,
  "tab_no": number or null,
  "estimated_probability": number (0-100) or null,
  "confidence": "high"/"medium"/"low" or null,
  "reasoning": "2-3 sentence summary of why this is/isn't a value bet",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "concerns": ["concern 1", "concern 2"]
}
```

## Important Rules

1. Only recommend a bet if you genuinely believe there's VALUE (edge > 5%)
2. If no value exists, set has_value_bet to false and explain why
3. Be conservative - it's better to pass than force a bad bet
4. Focus on the DATA, not gut feeling
5. Consider the full field, not just the favourite
6. If odds are missing for key runners, be more cautious"""


USER_PROMPT_TEMPLATE = """Analyze this race and identify if there's a value bet.

{race_data}

Remember:
- Only recommend a bet if estimated probability > implied probability (from odds)
- Respond with valid JSON only
- Be conservative - passing on a race is fine if no clear value exists"""


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
                has_value_bet=False,
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
                has_value_bet=False,
                reasoning=f"Failed to parse response: {raw_response[:200]}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
                raw_response=raw_response,
            )

        # Build output
        has_value = data.get("has_value_bet", False)

        # Calculate edge if we have the data
        est_prob = data.get("estimated_probability")
        selection = data.get("selection")

        # Find odds for selection
        odds = None
        implied_prob = None
        tab_no = data.get("tab_no")

        if selection:
            for runner in race_data.runners:
                if runner.name.lower() == selection.lower() or runner.tab_no == tab_no:
                    odds = runner.odds
                    implied_prob = runner.implied_prob
                    tab_no = runner.tab_no
                    break

        edge = None
        if est_prob and implied_prob:
            edge = est_prob - implied_prob

        return PredictionOutput(
            has_value_bet=has_value,
            selection=selection if has_value else None,
            tab_no=tab_no if has_value else None,
            odds=odds,
            estimated_probability=est_prob,
            implied_probability=implied_prob,
            edge=edge,
            confidence=data.get("confidence"),
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            concerns=data.get("concerns", []),
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

            if prediction.has_value_bet:
                logger.info(
                    f"Value bet found: {race.track} R{race.race_number} - "
                    f"{prediction.selection} @ ${prediction.odds} "
                    f"(est: {prediction.estimated_probability}%, "
                    f"implied: {prediction.implied_probability}%)"
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
            has_value_bet=False,
            reasoning=f"Failed to get race data: {error}",
            track=track,
            race_number=race_number,
        )

    # Run prediction
    predictor = Predictor()
    return predictor.predict(race_data, custom_instructions)
