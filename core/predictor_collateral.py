"""
TEST Predictor: Collateral Form Analysis.

Focuses on comparing horses through common tracks and opponents,
rather than relying on absolute speed ratings (which vary by track speed).

This is ideal for midweek/country racing where:
- Runners come from different circuits with varying track speeds
- Absolute ratings may not be comparable across tracks
- But horses that have raced at the SAME track can be compared directly

Usage:
    from core.predictor_collateral import CollateralPredictor

    predictor = CollateralPredictor()
    prediction = predictor.predict(race_data)
"""

from core.predictor import (
    Predictor,
    PredictionOutput,
    METRO_TRACKS,
    is_metro_track,
    DEFAULT_MODEL,
    USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT as LIVE_SYSTEM_PROMPT,
)
from core.race_data import RaceData
from core.logging import get_logger
from typing import Optional

logger = get_logger(__name__)


# =============================================================================
# COLLATERAL FORM SYSTEM PROMPT
# =============================================================================

# Same as live prompt, just add one sentence about comparing ratings at same tracks
COLLATERAL_SYSTEM_PROMPT = LIVE_SYSTEM_PROMPT.replace(
    "Focus on **normalized speed ratings**",
    "**If available, compare ratings at the same tracks to get relative speed between runners - this removes track speed bias.** Focus on **normalized speed ratings**"
)


# =============================================================================
# COLLATERAL PREDICTOR CLASS
# =============================================================================

class CollateralPredictor(Predictor):
    """
    Test predictor using collateral form analysis.

    Inherits from Predictor but uses a different system prompt
    focused on comparing horses through common tracks/opponents.
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
        Analyze race using collateral form analysis.

        Uses the collateral form system prompt instead of the default.
        Only supports normal mode (not promo_bonus).
        """
        if mode != "normal":
            logger.warning(f"CollateralPredictor only supports normal mode, ignoring mode='{mode}'")
            mode = "normal"

        # Format race data for prompt
        race_text = race_data.to_prompt_text()

        # Use collateral form prompt
        system_prompt = COLLATERAL_SYSTEM_PROMPT
        user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

        if custom_instructions:
            user_prompt += f"\n\nAdditional instructions: {custom_instructions}"

        # Call Claude
        logger.info(f"[COLLATERAL TEST] Calling Claude for {race_data.track} R{race_data.race_number}")

        import time
        import anthropic

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
