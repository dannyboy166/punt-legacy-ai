# Session Summary - 4 March 2026

## What We Built

### 1. Prompt Comparison Tool (`experiments/prompt_compare.py`)
- Runs two prompts (live vs test) on same race data
- Outputs side-by-side HTML comparison
- Usage: `python experiments/prompt_compare.py --track "Randwick" --date "28-Feb-2026" --all --past`

### 2. Live Predictor + Bonus Picks (`experiments/run_with_bonus.py`)
- Runs your live predictor with extra bonus bet output
- Outputs to `experiments/html_reports/`
- Usage: `python experiments/run_with_bonus.py --track "Sandown-Hillside" --date "25-Feb-2026" --all --past`

## Prompt Changes Made to `core/predictor.py`

### Changed:
1. Each-way definition: `"Good ratings at similar DISTANCE and CONDITIONS vs the field, place odds $1.80+"`
2. tipsheet_pick: `"at this distance and condition"` (was "distance/condition")

### Removed (de-emphasized A/E):
- Removed: `"- Poor jockey/trainer A/E ratios if relevant"`
- Removed A/E from runner_notes template

## TODO: Add Bonus Picks to Admin Predictor

To add `other_chances` (bonus picks) to the live admin predictor:

### 1. Update `core/predictor.py` SYSTEM_PROMPT

Add to JSON output section:
```
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
```

Add after contenders description:
```
**other_chances**: Horses with competitive ratings that COULD win but have issues preventing them being top contenders. Good value for bonus bets.
**less_likely**: Horses with weaker ratings or clear issues - just list names.
```

### 2. Update `core/predictor.py` parsing

In `_parse_normal_response()`, add:
```python
other_chances = data.get("other_chances", [])
less_likely = data.get("less_likely", [])
```

Add to `PredictionOutput` dataclass and `to_dict()`.

### 3. Update `server.py`

Include `other_chances` and `less_likely` in the `/predict` response.

### 4. Update Frontend (racing-tips-platform)

Add a "Bonus Bet Ideas" section below contenders in the prediction display.

## Files Created/Modified

- `experiments/prompt_compare.py` - NEW
- `experiments/run_with_bonus.py` - NEW
- `experiments/html_reports/` - NEW folder for HTML outputs
- `core/predictor.py` - Modified (A/E de-emphasized, each-way wording)
