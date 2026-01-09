# CLAUDE.md

## Project: Punt Legacy AI

AI-powered horse racing predictor product for Punt Legacy subscribers.

**Status:** Phase 4 Complete - Ready for Testing

---

## Project Vision

A subscription product where users can:
1. Get AI-powered race predictions
2. Customize which factors the AI emphasizes
3. Free tier (2-3 races/day) → Paid tiers for more

**Key principle:** Every calculation must be 100% correct and well-documented.

---

## Development Phases

### Phase 1: API Documentation & Foundation ✅ COMPLETE
- [x] Document ALL PuntingForm API endpoints (10 endpoints)
- [x] Document ALL Ladbrokes API endpoints
- [x] Document PuntingForm odds reliability issue
- [x] Build clean API wrappers with tests (41 tests passing)
- [x] Horse name normalization for cross-API matching

### Phase 2: Core Calculations ✅ COMPLETE
- [x] Speed rating calculation (with tests)
- [x] Time/margin calculations (with tests)
- [x] Normalization by distance/condition (with tests)

### Phase 3: Data Pipeline ✅ COMPLETE
- [x] Fetch race data from both APIs
- [x] Calculate per-run speed ratings
- [x] Format for AI prompt (markdown tables)
- [x] Merge PuntingForm + Ladbrokes data

### Phase 4: AI Integration ✅ COMPLETE
- [x] Build default predictor prompt
- [x] Claude API integration
- [x] Value bet identification logic
- [ ] Test on historical races
- [ ] Add user customization options (CURRENT)

### Phase 5: Product
- [ ] User accounts
- [ ] Usage tracking
- [ ] Billing integration

---

## Quick Start

```bash
# Run all tests
python3 -m pytest tests/ -v

# Test API connection
python3 -c "from api.puntingform import PuntingFormAPI; api = PuntingFormAPI(); print(api.get_meetings('08-Jan-2026'))"
```

---

## Project Structure

```
punt-legacy-ai/
├── CLAUDE.md              # This file - project guidance
├── api/                   # API clients
│   ├── puntingform.py     # PuntingForm API wrapper ✅
│   └── ladbrokes.py       # Ladbrokes API wrapper ✅
├── core/                  # Core logic
│   ├── normalize.py       # Horse/track name normalization ✅
│   ├── track_mapping.py   # Track name mapping between APIs ✅
│   ├── speed.py           # Speed rating calculations ✅
│   ├── race_data.py       # Data pipeline for Claude ✅
│   ├── predictor.py       # Claude AI predictor ✅
│   ├── results.py         # Prediction results & error handling ✅
│   ├── logging.py         # Structured logging ✅
│   └── normalization/     # Baseline data
│       ├── distance.csv   # Distance -> speed baselines ✅
│       └── condition.csv  # Condition -> speed multipliers ✅
├── tests/                 # Unit tests (163 tests passing)
│   ├── test_normalize.py      # Normalization tests ✅
│   ├── test_track_mapping.py  # Track mapping tests ✅
│   ├── test_speed.py          # Speed rating tests ✅
│   ├── test_race_data.py      # Data pipeline tests ✅
│   ├── test_api.py            # API client tests ✅
│   └── test_results.py        # Results system tests ✅
├── docs/                  # API documentation
│   ├── puntingform_api.md      # Full PF API docs ✅
│   ├── ladbrokes_api.md        # Full LB API docs ✅
│   └── puntingform_odds_issue.md  # Known issue ✅
└── data/                  # Example responses, test data
```

---

## Known Issues

### PuntingForm Odds Unreliable

**Status:** NOT FIXED (as of Jan 8, 2026)

PuntingForm's `bestPrice_Current` sometimes returns incorrect odds (up to 6x actual price).

**Solution:** Use `api.ladbrokes.LadbrokeAPI` for live odds instead.

See `docs/puntingform_odds_issue.md` for full details.

---

## APIs

### PuntingForm API
- **Base:** `https://api.puntingform.com.au/v2/`
- **Auth:** `?apiKey=YOUR_KEY`
- **Docs:** `docs/puntingform_api.md`
- **Use for:** Form history, career stats, A/E data, speedmaps, conditions

### Ladbrokes API
- **Base:** `https://api.ladbrokes.com.au/affiliates/v1/`
- **Auth:** Headers (`From`, `X-Partner`)
- **Docs:** `docs/ladbrokes_api.md`
- **Use for:** Live odds (accurate)

### Claude API
- **Key:** Stored in `.env` as `ANTHROPIC_API_KEY`

---

## Track Name Mapping

Different APIs use different names for the same tracks:

| PuntingForm | Ladbrokes |
|-------------|-----------|
| Sandown-Lakeside | Sandown |
| Yarra Glen | Yarra Valley |
| Murray Bridge GH | Murray Bridge |
| Geelong | Ladbrokes Geelong |
| Fannie Bay | Darwin |

**Tracks without Ladbrokes coverage:** Pioneer Park (Alice Springs), NZ tracks, HK tracks

```python
from api.ladbrokes import LadbrokeAPI

api = LadbrokeAPI()

# Use this method - handles track mapping automatically
odds, error = api.get_odds_for_pf_track("Sandown-Lakeside", 1)
if error:
    print(f"Skipped: {error}")  # e.g., "Tauranga is not covered by Ladbrokes"
else:
    print(f"Got odds for {len(odds)} runners")
```

---

## Error Handling

All predictions return structured results for frontend display:

```python
from core.results import PredictionResult, RaceStatus

# Successful prediction
result = PredictionResult.success(
    horse="Fast Horse",
    odds=3.50,
    track="Randwick",
    race_number=1,
)

# Failed prediction with clear reason
result = PredictionResult.track_not_supported(
    track="Tauranga",
    race_number=1,
)
# result.message = "Tauranga is not currently supported (no Ladbrokes coverage)"

# Check and display
if result.ok:
    print(f"Bet: {result.horse} @ ${result.odds}")
else:
    print(f"Skipped: {result.message}")
```

---

## Horse Name Matching

When linking data between PuntingForm and Ladbrokes, use `core.normalize`:

```python
from core.normalize import normalize_horse_name, horses_match

# Normalize names for dictionary lookup
name = normalize_horse_name("O'Brien's Star")  # "obriens star"

# Check if two names match
horses_match("O'Brien's Star", "OBRIENS STAR")  # True
```

---

## Related Projects

- `/Users/danielsamus/pfai-tracker` - Original research/prototyping (reference)
- `/Users/danielsamus/racing-tips-platform` - Punt Legacy website

---

## Speed Ratings

Speed ratings normalize performance across different distances and track conditions.

```python
from core.speed import calculate_speed_rating, calculate_run_rating

# Rating = actual_speed / expected_speed
# Rating > 1.0 = faster than expected
# Rating < 1.0 = slower than expected
# Rating = 1.0 = exactly average for distance/condition

rating = calculate_speed_rating(
    distance=1200,
    winner_time=71.5,
    margin=2.5,      # Lengths behind winner
    position=3,
    condition="G4"   # Good 4
)
# Returns ~0.99 (slightly below average)
```

**Baseline data** (from 30k+ Australian races):
- `core/normalization/distance.csv` - Expected speed by distance
- `core/normalization/condition.csv` - Speed multiplier by track condition

---

## Prep Stage Analysis

The system tracks which run in a preparation (campaign) each race was:

| Prep | Meaning |
|------|---------|
| 1 | First up (resuming from spell) |
| 2 | Second up |
| 3+ | Deeper into prep |

**What Claude sees for each runner:**

```
**SECOND UP** (career 2nd-up record: 5: 2-1-1)

| Date | Track | Dist | Cond | Pos | Margin | Rating | Prep |
|------|-------|------|------|-----|--------|--------|------|
| 01-Jan | Randwick | 1200m | G4 | 2/10 | 1.5L | 1.020 | 2 |
| 15-Dec | Rosehill | 1200m | G4 | 5/9 | 4.0L | 0.965 | 1 |
| 01-Sep | Warwick | 1400m | H8 | 1/12 | 0L | 1.045 | 4 |

Avg Rating: 1.010 | Best: 1.045
Prep Ratings: 1st-up: 0.965 | 2nd-up: 1.020 | 3rd+: 1.045
```

**Pattern detected:** This horse improves throughout a prep. Slow first up (0.965), better second up (1.020), best deeper in (1.045).

---

## Using the Predictor

### Quick Start

```python
from core.predictor import analyze_race

# Analyze a single race
prediction = analyze_race("Randwick", 1, "09-Jan-2026")

if prediction.has_value_bet:
    print(f"VALUE BET: {prediction.selection} @ ${prediction.odds}")
    print(f"Estimated: {prediction.estimated_probability}%")
    print(f"Implied: {prediction.implied_probability}%")
    print(f"Edge: +{prediction.edge:.1f}%")
    print(f"Reasoning: {prediction.reasoning}")
else:
    print(f"No value bet found: {prediction.reasoning}")
```

### Full Pipeline

```python
from core.race_data import RaceDataPipeline
from core.predictor import Predictor

# 1. Get race data
pipeline = RaceDataPipeline()
race_data, error = pipeline.get_race_data("Randwick", 1, "09-Jan-2026")

if error:
    print(f"Error: {error}")
else:
    # 2. See what Claude will receive
    print(race_data.to_prompt_text())

    # 3. Run prediction
    predictor = Predictor()
    result = predictor.predict(race_data)
    print(result.to_dict())
```

### Custom Instructions

```python
# Tell Claude to focus on specific factors
prediction = analyze_race(
    "Randwick", 1, "09-Jan-2026",
    custom_instructions="Focus on wet track form - it's currently raining."
)
```

---

## Next Steps

1. Test predictor on historical races to validate accuracy
2. Add user customization options (factor weighting)
3. Build confidence scoring system
4. Add stake sizing based on edge/confidence
