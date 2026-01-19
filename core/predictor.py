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
    confidence: int = 5  # 1-10 scale (10 = very confident)

    def to_dict(self) -> dict:
        return {
            "horse": self.horse,
            "tab_no": self.tab_no,
            "odds": self.odds,
            "tag": self.tag,
            "analysis": self.analysis,
            "confidence": self.confidence,
        }


@dataclass
class PromoBonusPick:
    """A horse picked specifically for bonus bets or promo plays."""

    horse: str
    tab_no: int
    odds: float
    pick_type: str  # "bonus_bet" or "promo_play"
    analysis: str  # Why this horse suits this type of bet

    def to_dict(self) -> dict:
        return {
            "horse": self.horse,
            "tab_no": self.tab_no,
            "odds": self.odds,
            "pick_type": self.pick_type,
            "analysis": self.analysis,
        }


@dataclass
class PredictionOutput:
    """Output from Claude predictor."""

    # Mode: "normal" or "promo_bonus"
    mode: str = "normal"

    # 1-3 contenders (used in normal mode)
    contenders: list[Contender] = field(default_factory=list)

    # Promo/Bonus picks (used in promo_bonus mode)
    bonus_pick: Optional[PromoBonusPick] = None
    promo_pick: Optional[PromoBonusPick] = None

    # Overall summary
    summary: str = ""

    # Race-level confidence (1-10) - how confident in predictions overall
    race_confidence: int = 5
    confidence_reason: str = ""  # Why confidence is high/low

    # Race context
    track: str = ""
    race_number: int = 0

    # Metadata
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: str = ""

    def to_dict(self) -> dict:
        result = {
            "mode": self.mode,
            "contenders": [c.to_dict() for c in self.contenders],
            "summary": self.summary,
            "race_confidence": self.race_confidence,
            "confidence_reason": self.confidence_reason,
            "track": self.track,
            "race_number": self.race_number,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }
        # Include promo/bonus picks if present
        if self.bonus_pick:
            result["bonus_pick"] = self.bonus_pick.to_dict()
        if self.promo_pick:
            result["promo_pick"] = self.promo_pick.to_dict()
        return result


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert horse racing analyst specializing in Australian thoroughbred racing.

Your task is to identify 1-3 horses that you think will WIN this race and give your thoughts on each.

## Understanding the Data

**Speed Ratings** (MOST IMPORTANT): Normalized performance measure (1.000 = average for that distance/condition).
- CRITICAL: Focus on ratings from runs at SIMILAR DISTANCE (within 200m) and SIMILAR CONDITIONS to today's race
- A horse with a 1.005 rating at today's distance/conditions is more relevant than a 1.015 at a different distance
- Compare ratings WITHIN THIS FIELD - highest-rated at relevant conditions is the key
- More recent runs (last 2-3) are more predictive than older runs

**Prep Run**: The "Prep" column shows which run in the current preparation (1 = first-up, 2 = second-up, etc.)
- Horses marked **FIRST UP** or **SECOND UP** show their career record in that state
- Second-up horses with good first-up records often improve

**A/E (Actual vs Expected)**: Measures if jockey/trainer outperforms market expectations (>1.0 = beats market, <1.0 = underperforms)

**Other factors to consider**: Class changes, weight, barrier, track/distance specialist

**Race Warnings**: Pay attention to any warnings about the race (e.g., many first-uppers, limited form data). These make predictions harder - adjust your confidence accordingly.

## Output Format

Return 1-3 contenders as JSON:

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat OR Value pick (use these two tags)",
      "analysis": "2-3 sentences: why this horse can win, referencing specific form at similar distance/conditions.",
      "confidence": number (1-10, how confident you are in THIS horse winning)
    }
  ],
  "race_confidence": number (1-10, overall confidence in your predictions for this race),
  "confidence_reason": "Brief reason for confidence level (e.g., 'Strong form data, clear top pick' or 'Many first-uppers, limited form makes this unpredictable')",
  "summary": "1-2 sentences summarizing the race"
}
```

## Tag Definitions
- **"The one to beat"**: The horse with the best form/ratings at similar distance and conditions. Your top pick.
- **"Value pick"**: A horse with solid form at relevant conditions whose odds are longer than expected. Must have genuine winning chance with form to back it up.

## Confidence Scale
- 8-10: Very confident - clear form standouts at today's distance/conditions
- 5-7: Moderate - decent form indicators but some uncertainty
- 1-4: Low confidence - many first-uppers, limited form, wide-open race

## Guidelines

- Pick based on who you think wins, prioritizing form at SIMILAR distance and conditions
- Quality over quantity - if only 1 horse stands out, just pick 1
- "Value pick" must have genuine form support, not just longer odds
- Your analysis should reference specific runs at similar conditions
- Be honest about confidence - low confidence races are harder to pick"""


PROMO_BONUS_SYSTEM_PROMPT = """You are an expert horse racing analyst specializing in Australian thoroughbred racing.

Your task is to identify picks for bonus bets and/or promo plays in this race (if genuine value exists):

1. **BONUS BET PICK**: A horse to use a bonus bet on (free bet where you only keep profits)
   - Target odds of $4.00-$8.00 for best value (high enough for profit, not so high it never wins)
   - MUST have solid form at similar distance/conditions - never pick a horse just for high odds
   - Look for horses with competitive speed ratings that are slightly overlooked by the market

2. **PROMO PICK**: A horse with strong, consistent form that you're confident will run TOP 3
   - Reliability and consistency matter more than odds
   - Look for horses with proven form at THIS distance and conditions
   - Focus on place chance (top 3) - this pick is about reliability, not winning

## Understanding the Data

**Speed Ratings** (MOST IMPORTANT): Normalized performance measure (1.000 = average for that distance/condition).
- CRITICAL: Focus on ratings from runs at SIMILAR DISTANCE (within 200m) and SIMILAR CONDITIONS
- Compare ratings WITHIN THIS FIELD at relevant conditions
- More recent runs (last 2-3) are more predictive than older runs

**Prep Run**: The "Prep" column shows which run in the current preparation (1 = first-up, 2 = second-up, etc.)
- Horses marked **FIRST UP** or **SECOND UP** show their career record in that state

**A/E (Actual vs Expected)**: Measures if jockey/trainer outperforms market expectations (>1.0 = beats market, <1.0 = underperforms)

**Other factors to consider**: Class changes, weight, barrier, track/distance form, place odds

**Race Warnings**: Pay attention to any warnings about the race (e.g., many first-uppers, limited form data). These make predictions harder - adjust your confidence accordingly.

## Output Format

Return both picks as JSON:

```json
{
  "bonus_pick": {
    "horse": "Horse Name",
    "tab_no": number,
    "odds": number,
    "analysis": "2-3 sentences: reference SPECIFIC form at similar distance/conditions that supports this pick",
    "confidence": number (1-10)
  },
  "promo_pick": {
    "horse": "Horse Name",
    "tab_no": number,
    "odds": number,
    "analysis": "2-3 sentences: why this horse will run top 3, referencing consistent form",
    "confidence": number (1-10)
  },
  "race_confidence": number (1-10, overall confidence in your predictions),
  "confidence_reason": "Brief reason for confidence level",
  "summary": "1-2 sentences summarizing both picks"
}
```

## Guidelines

- Only provide picks you genuinely believe have a good chance
- You may provide both picks, just one (bonus_pick OR promo_pick), or neither if no good options exist
- Don't force a pick if there's no genuine value - it's OK to leave one blank
- They can be the same horse if one horse fits both criteria well
- For bonus_pick: $4-8 odds range with PROVEN form at similar conditions (no hopeless longshots!)
- For promo_pick: must have consistent form showing top 3 capability
- Reference specific past runs in your analysis
- Be honest about confidence - low confidence races are harder to pick"""


USER_PROMPT_TEMPLATE = """Analyze this race and identify the contenders.

{race_data}

Respond with valid JSON only."""


PROMO_BONUS_USER_PROMPT_TEMPLATE = """Analyze this race and identify a bonus bet pick and a promo pick.

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
        mode: str = "normal",
    ) -> PredictionOutput:
        """
        Analyze race and predict value bets.

        Args:
            race_data: Complete race data from RaceDataPipeline
            custom_instructions: Optional additional instructions for Claude
            mode: Prediction mode - "normal" or "promo_bonus"

        Returns:
            PredictionOutput with selection and reasoning
        """
        # Validate mode
        if mode not in ("normal", "promo_bonus"):
            logger.warning(f"Invalid mode '{mode}', defaulting to 'normal'")
            mode = "normal"

        # Format race data for prompt
        race_text = race_data.to_prompt_text()

        # Select prompt based on mode
        if mode == "promo_bonus":
            system_prompt = PROMO_BONUS_SYSTEM_PROMPT
            user_prompt = PROMO_BONUS_USER_PROMPT_TEMPLATE.format(race_data=race_text)
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

        if custom_instructions:
            user_prompt += f"\n\nAdditional instructions: {custom_instructions}"

        # Call Claude with retry logic
        logger.info(f"Calling Claude for {race_data.track} R{race_data.race_number} (mode={mode})")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.2,  # Low temp for consistent predictions
                    system=system_prompt,
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
                mode=mode,
                summary=f"API error: {str(last_error)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        if last_error and 'raw_response' not in dir():
            logger.error(f"Claude API error: {str(last_error)}")
            return PredictionOutput(
                mode=mode,
                summary=f"API error: {str(last_error)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        # Parse response based on mode
        return self._parse_response(
            raw_response,
            race_data,
            mode,
        )

    def _parse_response(
        self,
        raw_response: str,
        race_data: RaceData,
        mode: str = "normal",
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
                mode=mode,
                summary=f"Failed to parse response: {raw_response[:200]}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
                raw_response=raw_response,
            )

        # Parse based on mode
        if mode == "promo_bonus":
            return self._parse_promo_bonus_response(data, race_data, raw_response)
        else:
            return self._parse_normal_response(data, race_data, raw_response)

    def _parse_normal_response(
        self,
        data: dict,
        race_data: RaceData,
        raw_response: str,
    ) -> PredictionOutput:
        """Parse normal mode response with contenders."""

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
                # Parse confidence (default to 5 if not provided)
                confidence = c.get("confidence", 5)
                try:
                    confidence = int(confidence)
                    confidence = max(1, min(10, confidence))  # Clamp to 1-10
                except (ValueError, TypeError):
                    confidence = 5

                contenders.append(Contender(
                    horse=horse,
                    tab_no=tab_no,
                    odds=odds,
                    tag=c.get("tag", "Contender"),
                    analysis=c.get("analysis", ""),
                    confidence=confidence,
                ))
            elif horse and tab_no and not odds:
                logger.warning(f"Skipping contender {horse}: no odds available")
            elif horse:
                logger.warning(f"Skipping contender {horse}: missing tab_no={tab_no}")

        # Parse race-level confidence
        race_confidence = data.get("race_confidence", 5)
        try:
            race_confidence = int(race_confidence)
            race_confidence = max(1, min(10, race_confidence))
        except (ValueError, TypeError):
            race_confidence = 5

        return PredictionOutput(
            mode="normal",
            contenders=contenders,
            summary=data.get("summary", ""),
            race_confidence=race_confidence,
            confidence_reason=data.get("confidence_reason", ""),
            track=race_data.track,
            race_number=race_data.race_number,
            model=self.model,
            raw_response=raw_response,
        )

    def _parse_promo_bonus_response(
        self,
        data: dict,
        race_data: RaceData,
        raw_response: str,
    ) -> PredictionOutput:
        """Parse promo/bonus mode response with bonus_pick and promo_pick."""

        bonus_pick = None
        promo_pick = None

        # Parse bonus_pick
        if "bonus_pick" in data and data["bonus_pick"]:
            bonus_pick = self._parse_single_pick(
                data["bonus_pick"],
                race_data,
                "bonus_bet",
            )

        # Parse promo_pick
        if "promo_pick" in data and data["promo_pick"]:
            promo_pick = self._parse_single_pick(
                data["promo_pick"],
                race_data,
                "promo_play",
            )

        # Parse race-level confidence
        race_confidence = data.get("race_confidence", 5)
        try:
            race_confidence = int(race_confidence)
            race_confidence = max(1, min(10, race_confidence))
        except (ValueError, TypeError):
            race_confidence = 5

        return PredictionOutput(
            mode="promo_bonus",
            bonus_pick=bonus_pick,
            promo_pick=promo_pick,
            summary=data.get("summary", ""),
            race_confidence=race_confidence,
            confidence_reason=data.get("confidence_reason", ""),
            track=race_data.track,
            race_number=race_data.race_number,
            model=self.model,
            raw_response=raw_response,
        )

    def _parse_single_pick(
        self,
        pick_data: dict,
        race_data: RaceData,
        pick_type: str,
    ) -> Optional[PromoBonusPick]:
        """Parse a single bonus/promo pick from JSON data."""

        horse = pick_data.get("horse")
        tab_no = pick_data.get("tab_no")
        odds = pick_data.get("odds")

        # Check if odds is a valid number
        if odds is not None:
            try:
                odds = float(odds)
            except (ValueError, TypeError):
                odds = None

        # Look up odds from race data if not provided or invalid
        if horse and not odds:
            normalized_horse = normalize_horse_name(horse)
            for runner in race_data.runners:
                if normalize_horse_name(runner.name) == normalized_horse or runner.tab_no == tab_no:
                    odds = runner.odds
                    tab_no = runner.tab_no
                    horse = runner.name  # Canonical name
                    break

            if not odds:
                logger.warning(f"Could not find odds for {horse} (tab {tab_no}) in race data")

        if horse and tab_no and odds:
            return PromoBonusPick(
                horse=horse,
                tab_no=tab_no,
                odds=odds,
                pick_type=pick_type,
                analysis=pick_data.get("analysis", ""),
            )
        elif horse:
            logger.warning(f"Skipping {pick_type} pick {horse}: missing data")

        return None

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
