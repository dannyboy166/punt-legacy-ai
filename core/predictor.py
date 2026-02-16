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
    tag: str  # "The one to beat" or "Value bet"
    analysis: str  # Natural language analysis of the horse and price
    place_odds: Optional[float] = None  # Place odds for each-way bets
    confidence: Optional[int] = None  # Deprecated - not used in new model
    tipsheet_pick: bool = False  # True if Claude would genuinely bet on this

    def to_dict(self) -> dict:
        result = {
            "horse": self.horse,
            "tab_no": self.tab_no,
            "odds": self.odds,
            "place_odds": self.place_odds,
            "tag": self.tag,
            "analysis": self.analysis,
            "tipsheet_pick": self.tipsheet_pick,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


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

    # Notes for non-selected runners (1 sentence each)
    runner_notes: dict = field(default_factory=dict)

    # Race-level confidence (1-10) - deprecated, not used in new model
    race_confidence: Optional[int] = None
    confidence_reason: Optional[str] = None

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
            "runner_notes": self.runner_notes,
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

SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout
- **"Each-way chance"** - Could win, should place, place odds good value
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

You also have: win/place odds, jockey/trainer A/E ratios, career record, first-up/second-up records, prep run number, barrier, weight, speedmap/pace data, gear changes.

Also include brief notes for non-selected runners explaining why they weren't picked (1 sentence each).

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
  "runner_notes": {
    "Horse Name": "1 sentence why not selected",
    "Another Horse": "1 sentence why not selected"
  },
  "summary": "Brief overview or reason for 0 picks"
}
```

**tipsheet_pick = true** when you would genuinely bet on this horse yourself:
- Speed ratings clearly support this horse vs the field at this distance/condition
- The odds represent real value
- You're confident in the pick (requires most of the field to have sufficient form data)"""


PROMO_BONUS_SYSTEM_PROMPT = """You are an expert horse racing analyst.

Identify a bonus bet pick and/or promo pick (if genuine value exists):

1. **BONUS BET**: Odds $5.00+ with genuine winning chance (you only keep profit, not stake)
2. **PROMO PICK**: Consistent, reliable speed ratings — confidence matters more than odds

**Skip both picks when:**
- Too many unknowns (50%+ have no race form)
- Field too even, no standouts
- Insufficient data for confident assessment

## Key Analysis

Focus on **normalized speed ratings** from RACE runs (not trials) at similar distance and conditions to the race being predicted. More recent runs are more relevant. **Speed ratings matter more than last start wins or career win/place stats.**

**First-up/Second-up horses:** Check their past runs at the same prep stage (Prep=1 in form table for first-up runs, Prep=2 for second-up). Some horses perform better/worse when fresh - their career first-up record and past first-up ratings tell you this.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form — horses don't always try
- If a horse has 0 race runs, they are UNKNOWN (first starter) — could be brilliant or useless

You also have: win/place odds, jockey/trainer A/E ratios, career record, first-up/second-up records, prep run number, barrier, weight, speedmap/pace data, gear changes.

## Output

```json
{
  "bonus_pick": {
    "horse": "Horse Name",
    "tab_no": number,
    "odds": number,
    "analysis": "1-2 sentences"
  },
  "promo_pick": {
    "horse": "Horse Name",
    "tab_no": number,
    "odds": number,
    "analysis": "1-2 sentences"
  },
  "summary": "Brief overview or reason for skipped picks"
}
```

Either pick can be null if no genuine value exists."""


USER_PROMPT_TEMPLATE = """Analyze this race and pick your contenders (0-3).

{race_data}

Respond with JSON only."""


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
        """Parse normal mode response with contenders (0-3)."""

        # Parse contenders (can be empty array)
        contenders = []
        for c in data.get("contenders", []):
            horse = c.get("horse")
            tab_no = c.get("tab_no")
            odds = c.get("odds")
            place_odds = c.get("place_odds")

            # Check if odds is a valid number
            if odds is not None:
                try:
                    odds = float(odds)
                except (ValueError, TypeError):
                    odds = None

            # Check if place_odds is a valid number
            if place_odds is not None:
                try:
                    place_odds = float(place_odds)
                except (ValueError, TypeError):
                    place_odds = None

            # Look up from race data if not provided or invalid
            if horse:
                normalized_horse = normalize_horse_name(horse)
                for runner in race_data.runners:
                    if normalize_horse_name(runner.name) == normalized_horse or runner.tab_no == tab_no:
                        if not odds:
                            odds = runner.odds
                        if not place_odds:
                            place_odds = runner.place_odds
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
                    place_odds=place_odds,
                    confidence=None,  # Not used in new model
                    tipsheet_pick=c.get("tipsheet_pick", False),
                ))
            elif horse and tab_no and not odds:
                logger.warning(f"Skipping contender {horse}: no odds available")
            elif horse:
                logger.warning(f"Skipping contender {horse}: missing tab_no={tab_no}")

        # Extract runner notes for non-selected runners
        runner_notes = data.get("runner_notes", {})

        # Return output (contenders can be empty)
        return PredictionOutput(
            mode="normal",
            contenders=contenders,
            summary=data.get("summary", ""),
            runner_notes=runner_notes,
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
