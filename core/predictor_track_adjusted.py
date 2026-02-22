"""
TEST Predictor: Track-Adjusted Ratings.

Shows the "Adj" column in form tables - ratings normalized by track speed.

This removes track speed bias by dividing the raw rating by the track's
overall rating from track_ratings.csv.

Usage:
    from core.predictor_track_adjusted import TrackAdjustedPredictor

    predictor = TrackAdjustedPredictor()
    prediction = predictor.predict(race_data)
"""

from core.predictor import (
    Predictor,
    PredictionOutput,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from core.race_data import RaceData
from core.logging import get_logger
from typing import Optional
import time
import anthropic

logger = get_logger(__name__)


class TrackAdjustedPredictor(Predictor):
    """
    Test predictor that shows track-adjusted ratings to Claude.

    Uses the exact same prompt as the live predictor, but includes the
    "Adj" column in form tables showing track-normalized ratings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize with same params as base Predictor."""
        super().__init__(api_key=api_key, model=model)

    def predict(
        self,
        race_data: RaceData,
        custom_instructions: Optional[str] = None,
        mode: str = "normal",
    ) -> PredictionOutput:
        """
        Analyze race showing track-adjusted ratings.

        Uses the same prompt as live predictor but with include_venue_adjusted=True.
        Only supports normal mode (not promo_bonus).
        """
        if mode != "normal":
            logger.warning(f"TrackAdjustedPredictor only supports normal mode, ignoring mode='{mode}'")
            mode = "normal"

        # Format race data WITH track-adjusted ratings column
        race_text = race_data.to_prompt_text(include_venue_adjusted=True)

        # Use SAME prompt as live predictor
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

        if custom_instructions:
            user_prompt += f"\n\nAdditional instructions: {custom_instructions}"

        # Call Claude
        logger.info(f"[TRACK-ADJ TEST] Calling Claude for {race_data.track} R{race_data.race_number}")

        max_retries = 3
        last_error = None
        raw_response = None

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.2,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                )

                raw_response = response.content[0].text
                logger.debug(f"Raw response: {raw_response}")
                break

            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
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
                break
            except Exception as e:
                last_error = e
                break

        if raw_response is None:
            logger.error(f"Claude API error after {max_retries} attempts: {str(last_error)}")
            return PredictionOutput(
                mode=mode,
                summary=f"API error: {str(last_error)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        # Use parent's parsing logic (same output format)
        return self._parse_response(raw_response, race_data, mode)
