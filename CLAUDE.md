# CLAUDE.md

## Project: Punt Legacy AI

AI-powered horse racing predictor product for Punt Legacy subscribers.

**Status:** Phase 1 Complete - Starting Phase 2

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

### Phase 2: Core Calculations (CURRENT)
- [ ] Speed rating calculation (with tests)
- [ ] Time/margin calculations (with tests)
- [ ] Normalization by distance/condition (with tests)

### Phase 3: Data Pipeline
- [ ] Fetch race data
- [ ] Calculate derived metrics
- [ ] Format for AI prompt

### Phase 4: AI Integration
- [ ] Build default predictor prompt
- [ ] Test on historical races
- [ ] Add user customization options

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
├── core/                  # Core utilities
│   ├── normalize.py       # Horse/track name normalization ✅
│   ├── track_mapping.py   # Track name mapping between APIs ✅
│   ├── results.py         # Prediction results & error handling ✅
│   └── logging.py         # Structured logging ✅
├── tests/                 # Unit tests (89 tests)
│   ├── test_normalize.py      # Normalization tests ✅
│   ├── test_track_mapping.py  # Track mapping tests ✅
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

## Next Steps

1. Build speed rating calculation in `core/speed.py`
2. Use form data from `api.puntingform.get_form()` for calculations
3. Implement time/margin calculations with tests
4. Port normalization logic from pfai-tracker with improvements
