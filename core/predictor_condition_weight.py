"""
TEST Predictor: Condition-Weighted Ratings.

This test predictor uses the EXACT same:
- Model (claude-sonnet-4-20250514)
- Temperature (0.2)
- Max tokens (2000)
- Data format (race_data.to_prompt_text())
- Response parsing

The ONLY difference is the SYSTEM_PROMPT, which better explains:
- Ratings are normalized and comparable across conditions
- Prioritize runs at similar conditions (CStep within ±2)
- Still consider all form, don't filter
- Wet-tracker vs dry-tracker patterns

Usage:
    from core.predictor_condition_weight import ConditionWeightPredictor

    predictor = ConditionWeightPredictor()
    prediction = predictor.predict(race_data)
"""

from core.predictor import (
    Predictor,
    PredictionOutput,
    DEFAULT_MODEL,
    USER_PROMPT_TEMPLATE,
)
from core.race_data import RaceData
from core.logging import get_logger
from typing import Optional
import time
import anthropic

logger = get_logger(__name__)


# =============================================================================
# MODIFIED SYSTEM PROMPT - Only change from live predictor
# =============================================================================

CONDITION_WEIGHT_SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout
- **"Each-way chance"** - Good ratings at similar DISTANCE and CONDITIONS vs the field, place odds $1.80+
- **"Value bet"** - Odds better than their form suggests

**Pick 0 contenders (no bet) when:**
- A lot of field has no race form (only trials) - you can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

## How to Analyze

**Always compare to the field.** A 99 rating beats a field of 97s. A 97 rating loses to a field of 101s.

**Key factors to weigh together:**
- Speed ratings at similar distance (±10-20%) and condition (CStep within ±2)
- Prep patterns - first-up/second-up record AND their past ratings at that prep stage
- Weight carried today vs their best runs at this distance/condition
- Jockey/trainer A/E ratios - who consistently beats market expectations
- Barrier draw and racing pattern

**Justify your picks.** Don't just cite data - explain why this horse is better than the others in the field.

## Key Data Points

**Speed ratings are normalized** - a 102 on Heavy 8 equals 102 on Good 4. You can compare ratings across ALL runs directly.

**CStep column** = condition steps from today's track:
- CStep 0: Same condition as today
- CStep ±1-2: Similar conditions (prioritize these)
- CStep ±3+: Very different conditions

**Wet-tracker / dry-tracker patterns:** If a horse rates 105 on wet but 98 on dry, expect wet-like performance on wet days.

**Prep patterns:** Check Prep column (1=first-up, 2=second-up). Compare to their career first/second-up record and past ratings at same prep stage.

**Critical:**
- Barrier trials (TRIAL) don't count - horses don't always try
- 0 race runs = UNKNOWN first starter
- If 50%+ of field has no race form, pick 0 contenders

## Runner Notes

For non-selected runners, explain specifically:
- Their best rating at this distance/condition vs your picks
- Why they fall short (lower ratings, wrong distance, poor prep record, weight concerns)
Don't cite generic stats - be specific to today's race.

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
  "other_chances": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "rating": "101.2 at 1200m S5",
      "issue": "Brief reason this horse has issues vs contenders"
    }
  ],
  "less_likely": ["Horse A", "Horse B"],
  "runner_notes": {
    "Horse Name": "1 sentence: ratings at this distance/condition, weight change, barrier",
    "Another Horse": "1 sentence: specific reason vs contenders"
  },
  "summary": "Brief overview or reason for 0 picks"
}
```

**other_chances**: Horses with competitive ratings that COULD win but have issues preventing them being top contenders. Good value for bonus bets.
**less_likely**: Horses with weaker ratings or clear issues - just list names.

**tipsheet_pick = true** when you would genuinely bet on this horse yourself:
- Speed ratings clearly support this horse vs the field at this distance and condition
- The odds represent real value
- You're confident in the pick (requires most of the field to have sufficient form data)"""


class ConditionWeightPredictor(Predictor):
    """
    Test predictor with improved condition weighting explanation.

    Uses EXACTLY the same:
    - Model (claude-sonnet-4-20250514)
    - Temperature (0.2)
    - Max tokens (2000)
    - Data format (race_data.to_prompt_text())
    - Response parsing (_parse_response from base class)

    Only the SYSTEM_PROMPT is different.
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
        Analyze race with condition-weighted prompt.

        Uses EXACTLY the same data and parsing as live predictor.
        Only supports normal mode (not promo_bonus).
        """
        if mode != "normal":
            logger.warning(f"ConditionWeightPredictor only supports normal mode, ignoring mode='{mode}'")
            mode = "normal"

        # SAME data format as live predictor
        race_text = race_data.to_prompt_text()

        # DIFFERENT prompt - this is the only change
        system_prompt = CONDITION_WEIGHT_SYSTEM_PROMPT

        # SAME user prompt template as live predictor
        user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

        if custom_instructions:
            user_prompt += f"\n\nAdditional instructions: {custom_instructions}"

        # Call Claude with SAME settings as live predictor
        logger.info(f"[CONDITION-WEIGHT TEST] Calling Claude for {race_data.track} R{race_data.race_number}")

        max_retries = 3
        last_error = None
        raw_response = None

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,          # SAME model
                    max_tokens=2000,           # SAME max_tokens
                    temperature=0.2,           # SAME temperature
                    system=system_prompt,      # DIFFERENT - modified prompt
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

        else:
            logger.error(f"Claude API error after {max_retries} attempts: {str(last_error)}")
            return PredictionOutput(
                mode=mode,
                summary=f"API error: {str(last_error)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        if last_error and raw_response is None:
            logger.error(f"Claude API error: {str(last_error)}")
            return PredictionOutput(
                mode=mode,
                summary=f"API error: {str(last_error)}",
                track=race_data.track,
                race_number=race_data.race_number,
                model=self.model,
            )

        # SAME response parsing as live predictor
        return self._parse_response(
            raw_response,
            race_data,
            mode,
        )
