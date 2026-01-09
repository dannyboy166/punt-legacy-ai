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

## What Claude Receives

For each runner, Claude sees **raw form data** - no pre-calculated averages. Claude uses its own reasoning to identify patterns.

**Example runner data:**

```
### 3. So You Ready
Barrier: 1 | Weight: 59kg | Age: 4G
Odds: $3.30 (ladbrokes) → 30.3% implied
Jockey: Jay Ford (A/E: 0.49)
Trainer: Ms K Buchanan (A/E: 0.89)
Career: 13: 1-1-0 (8% win)
**FIRST UP** (career 1st-up record: 3: 0-0-0)

| Date | Track | Dist | Cond | Pos | Margin | Rating | Prep |
|------|-------|------|------|-----|--------|--------|------|
| 26-Dec | Beaumont | 2100m | G4 | 6/12 | 2.1L | 1.003 | 3 |
| 10-Dec | Wyong | 1600m | G4 | 6/6 | 11.8L | 0.990 | 2 |
| 20-Nov | Newcastle | 1400m | G4 | 7/8 | 10.7L | 0.972 | 1 |
| 12-Jun | Gosford | 2100m | S6 | 1/9 | 3L | 0.996 | 5 |
```

**Claude reasons:** "Today is 2100m G4. This horse has a previous win at Gosford 2100m (0.996). Ratings improving this prep: 0.972 → 0.990 → 1.003. Third-up and peaking."

**Key principle:** No pre-calculated averages. Claude analyzes which runs are relevant based on today's race conditions.

---

## Using the Predictor

### Quick Start

```python
from core.predictor import analyze_race

# Analyze a single race
prediction = analyze_race("Randwick", 1, "09-Jan-2026")

# Returns 1-3 contenders with natural language analysis
for c in prediction.contenders:
    print(f"{c.horse} @ ${c.odds} - {c.chance.upper()}")
    print(f"   {c.analysis}")

print(f"\nSummary: {prediction.summary}")
```

### Example Output

```
BALLINA R3
============================================================

1. Call To Courage (#7) @ $2.80 - BEST
   Impressive last-start winner with a strong rating of 1.011 at Lismore.
   At $2.80 she looks like solid value as the form pick in a winnable Class 1.

2. Aquatier (#3) @ $2.50 - SOLID
   Dominant 5.5-length winner last start, but poor first-up record (0 from 3)
   is a major concern. Short price for a horse that historically struggles fresh.

3. Darling Take Care (#4) @ $9.50 - EACH-WAY
   Won at this exact track and distance in November. Worth an each-way play
   at decent odds despite the first-up query.

SUMMARY: Call To Courage looks the pick based on recent winning form and
first-up ability, while Aquatier's brilliant last start is offset by poor fresh form.
```

### Contender Levels

- **BEST** = Most likely winner
- **SOLID** = Genuine winning chance
- **EACH-WAY** = Could win if things go right

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

    for c in result.contenders:
        print(f"{c.horse}: {c.analysis}")
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

## Design Philosophy

**Why contenders instead of "BET/NO BET"?**

The predictor identifies 1-3 horses that could realistically win and gives natural language analysis on each, including thoughts on the price. This approach:

1. Lets users make their own betting decisions
2. Avoids rigid "bet/no bet" that can frustrate users if a "no bet" wins
3. Provides more nuanced analysis (e.g., "best horse but short price")
4. Claude can express uncertainty naturally (e.g., "worth a small each-way")

**Example analysis styles:**
- "Looks good value at this price"
- "Short price for what you're getting"
- "Worth a small each-way"
- "The one to beat but tight in the market"

---

## Next Steps

1. Build frontend to display predictions
2. Add user accounts and usage tracking
3. Implement subscription tiers
4. Historical backtesting
