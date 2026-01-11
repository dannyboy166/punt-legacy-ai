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
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from core.race_data import RaceData
from core.logging import get_logger
from core.normalize import normalize_horse_name

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
    tag: str  # Natural description like "The one to beat", "Value pick", etc.
    analysis: str  # Natural language analysis of the horse and price

    def to_dict(self) -> dict:
        return {
            "horse": self.horse,
            "tab_no": self.tab_no,
            "odds": self.odds,
            "tag": self.tag,
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

Your task is to identify 1-3 horses that you think will WIN this race and give your thoughts on each.

## Understanding the Data

**Speed Ratings** (MOST IMPORTANT): Normalized performance measure (1.000 = average for that distance/condition).
- Compare ratings WITHIN THIS FIELD - the highest-rated horse has faster normalized speed relative to competitors
- Prioritize runs at similar distance and similar track condition
- More recent runs are more relevant than older runs

**Prep Run**: The "Prep" column shows which run in the current preparation (1 = first-up, 2 = second-up, etc.)
- Horses marked **FIRST UP** or **SECOND UP** show their career record in that state

**A/E (Actual vs Expected)**: Measures if jockey/trainer outperforms market expectations (>1.0 = beats market, <1.0 = underperforms)

**Other factors to consider**: Class changes, weight, barrier, track/distance form

## Output Format

Return 1-3 contenders as JSON:

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "short phrase - e.g. The one to beat, Value pick, Main danger",
      "analysis": "2-3 sentences: why this horse can win, and your thoughts on the price."
    }
  ],
  "summary": "1-2 sentences summarizing the race"
}
```

## Guidelines

- Pick based on who you think wins
- Quality over quantity - if only 1 horse stands out, just pick 1
- Only suggest each-way if place odds are $1.80+
- Your summary should align with your picks"""


USER_PROMPT_TEMPLATE = """Analyze this race and identify the contenders.

{race_data}

Respond with valid JSON only."""


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

        # Call Claude with retry logic
        logger.info(f"Calling Claude for {race_data.track} R{race_data.race_number}")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.2,  # Low temp for consistent predictions
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                )

                raw_response = response.content[0].text
                logger.debug(f"Raw response: {raw_response}")
                break  # Success, exit retry loop

            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"Connection error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code >= 500 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {e.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                break  # Don't retry client errors (4xx)
            except Exception as e:
                last_error = e
                break  # Don't retry unknown errors

        else:
            # All retries exhausted
            logger.error(f"Claude API error after {max_retries} attempts: {str(last_error)}")
            return PredictionOutput(
                summary=f"API error: {str(last_error)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        if last_error and 'raw_response' not in dir():
            logger.error(f"Claude API error: {str(last_error)}")
            return PredictionOutput(
                summary=f"API error: {str(last_error)}",
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
                summary=f"Failed to parse response: {raw_response[:200]}",
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

            # Check if odds is a valid number
            if odds is not None:
                try:
                    odds = float(odds)
                except (ValueError, TypeError):
                    odds = None  # Invalid odds like "Not available"

            # Look up odds from race data if not provided or invalid
            if horse and not odds:
                normalized_horse = normalize_horse_name(horse)
                for runner in race_data.runners:
                    # Match by normalized name or tab number
                    if normalize_horse_name(runner.name) == normalized_horse or runner.tab_no == tab_no:
                        odds = runner.odds
                        tab_no = runner.tab_no
                        horse = runner.name  # Canonical name
                        break

                if not odds:
                    logger.warning(f"Could not find odds for {horse} (tab {tab_no}) in race data")

            if horse and tab_no and odds:
                contenders.append(Contender(
                    horse=horse,
                    tab_no=tab_no,
                    odds=odds,
                    tag=c.get("tag", "Contender"),
                    analysis=c.get("analysis", ""),
                ))
            elif horse and tab_no and not odds:
                logger.warning(f"Skipping contender {horse}: no odds available")
            elif horse:
                logger.warning(f"Skipping contender {horse}: missing tab_no={tab_no}")

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
                    f"{top.horse} @ ${top.odds} ({top.tag})"
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
            summary=f"Failed to get race data: {error}",
            track=track,
            race_number=race_number,
        )

    # Run prediction
    predictor = Predictor()
    return predictor.predict(race_data, custom_instructions)
