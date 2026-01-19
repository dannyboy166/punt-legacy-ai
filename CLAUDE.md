# CLAUDE.md

## Project: Punt Legacy AI

AI-powered horse racing predictor product for Punt Legacy subscribers.

**Status:** Phase 4 Complete - Predictor Working

---

## Project Vision

A subscription product where users can:
1. Get AI-powered race predictions (1-3 contenders per race)
2. Natural language analysis with price commentary
3. Free tier (2-3 races/day) → Paid tiers for more

**Key principle:** Let Claude be the expert - provide good data, minimal rules.

---

## Development Phases

### Phase 1: API Documentation & Foundation ✅ COMPLETE
- [x] Document ALL PuntingForm API endpoints (10 endpoints)
- [x] Document ALL Ladbrokes API endpoints
- [x] Document PuntingForm odds reliability issue
- [x] Build clean API wrappers with tests
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
- [x] Include place odds from Ladbrokes

### Phase 4: AI Integration ✅ COMPLETE
- [x] Build predictor prompt (simplified, lets Claude decide)
- [x] Claude API integration
- [x] Natural language tags (not forced categories)
- [x] Place odds consideration for each-way ($1.80+ threshold)

### Phase 5: Product
- [ ] Build frontend to display predictions
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
├── server.py              # FastAPI server ✅
├── requirements.txt       # Python dependencies ✅
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
├── experiments/           # Experimental features
│   ├── bet_type_predictor.py   # v2 predictor (0-3 picks, skips bad races) ✅
│   └── backtest.py             # Backtesting script ✅
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

## Scratching Detection

Scratched horses are automatically filtered out using multiple checks:

1. **PuntingForm `scratched` field** - Primary check
2. **Ladbrokes `is_scratched` field** - Secondary check (note: uses `is_scratched` not `scratched`)
3. **Blank jockey** - Late scratching indicator (if jockey field is empty/whitespace, horse is likely scratched)
4. **SP = $0** - PuntingForm sets Starting Price to 0 for scratched horses

This ensures late scratchings are caught even if APIs haven't updated their `scratched` flags yet.

**Note:** For backtesting, PuntingForm often removes scratched horses entirely from the runners list rather than marking them as scratched.

---

## Barrier Trials & Form Confidence

### Barrier Trial Detection

Form runs from barrier trials are flagged with `isBarrierTrial: true` from PuntingForm. In the output:
- Form table has a "Trial" column showing `TRIAL` for barrier trials
- `race_runs_count` excludes trials (actual race runs only)
- `trial_runs_count` shows number of trial runs

### Form Confidence Warnings

**Race-level warnings** (shown in `race_data.warnings`):
- `"LOW CONFIDENCE: 5/8 runners (63%) are first-up with limited form"` - if 50%+ are first-up
- `"3/8 runners are first-up"` - if 3+ are first-up but <50%
- `"5/8 runners have < 3 race runs (limited form data)"` - if majority have limited data

**Runner-level indicators**:
```
Form: 5 race runs, 2 trials
Form: 2 race runs ⚠️ LIMITED FORM DATA
⚠️ NO FORM AVAILABLE - first starter or no data
```

This helps Claude (and users) understand when predictions are less confident due to missing data.

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
Barrier: 1 | Weight: 59kg
Odds: $3.30 win / $1.45 place → 30.3% implied
Jockey: Jay Ford (A/E: 0.49)
Trainer: Ms K Buchanan (A/E: 0.89)
Career: 13: 1-1-0 (8% win)
**FIRST UP** (career 1st-up record: 3: 0-0-0)
Form: 4 race runs, 1 trials

| Date | Track | Dist | Cond | Pos | Margin | Rating | Prep | Trial |
|------|-------|------|------|-----|--------|--------|------|-------|
| 26-Dec | Beaumont | 2100m | G4 | 6/12 | 2.1L | 1.003 | 3 | - |
| 10-Dec | Wyong | 1600m | G4 | 6/6 | 11.8L | 0.990 | 2 | - |
| 20-Nov | Newcastle | 1400m | G4 | 7/8 | 10.7L | 0.972 | 1 | - |
| 05-Nov | Rosehill | 1000m | G3 | 2/8 | 1.5L | N/A | - | TRIAL |
| 12-Jun | Gosford | 2100m | S6 | 1/9 | 3L | 0.996 | 5 | - |
```

**Data includes:**
- Barrier, weight (no age/sex - removed as not critical)
- Win AND place odds from Ladbrokes
- Jockey/trainer A/E ratios
- Career record + first-up/second-up record
- Form summary with race run count and trial count
- Last 10 runs with speed ratings, prep run number, and barrier trial flag

**Key principle:** No pre-calculated averages. Claude analyzes which runs are relevant based on today's race conditions.

---

## Using the Predictor

### Quick Start

```python
from core.race_data import RaceDataPipeline
from core.predictor import Predictor

# 1. Get race data
pipeline = RaceDataPipeline()
race_data, error = pipeline.get_race_data("Gosford", 4, "09-Jan-2026")

if error:
    print(f"Error: {error}")
else:
    # 2. Run prediction
    predictor = Predictor()
    result = predictor.predict(race_data)

    # 3. Display results
    for c in result.contenders:
        print(f"{c.horse} (#{c.tab_no}) @ ${c.odds}")
        print(f'   "{c.tag}"')
        print(f"   {c.analysis}")

    print(f"\nSummary: {result.summary}")
```

### Example Output

```
============================================================

  GOSFORD RACE 4
  1600m • G4 • Class 1

============================================================

  1. FEDERAL RESERVE (#2)
     $2.35 win / $1.50 place

     "The one to beat"

     Strong recent form with a win at Beaumont and close
     second at Newcastle. Price looks fair given the form
     edge, but he's clearly the horse to beat.

  --------------------------------------------------

  2. MURPHILLY (#5)
     $5.50 win / $2.60 place

     "Value pick"

     Won his last start at Gosford over this exact distance.
     The jockey/trainer combo has excellent A/E figures,
     making this price attractive value.

  --------------------------------------------------

  3. OCEAN TSUNAMI (#7)
     $3.30 win / $1.45 place

     "Lightly-raced improver"

     Won first-up on debut then ran a strong second. The
     price reflects the upside potential but carries the
     risk of inexperience.

============================================================

  SUMMARY
  Federal Reserve looks the most reliable pick with strong
  recent form, while Murphilly offers good value returning
  to his winning track and distance.

============================================================
```

### Natural Tags

Claude uses natural language tags - not forced categories. Examples:
- "The one to beat"
- "Value pick"
- "Main danger"
- "First-up specialist"
- "Course specialist"
- "Each-way chance" (only if place odds $1.80+)

---

## Design Philosophy

**Let Claude be the expert.**

The prompt is intentionally simple - it explains what the data means, but doesn't force Claude to follow rigid rules. Claude decides:
- How many contenders (1-3)
- What tags to use
- How to weight different factors (speed ratings, A/E, prep patterns, etc.)
- Whether to mention each-way (only if place odds $1.80+)

**Why contenders instead of "BET/NO BET"?**

1. Lets users make their own betting decisions
2. Avoids rigid "bet/no bet" that frustrates users if a "no bet" wins
3. Claude can express nuance naturally ("best horse but short price")
4. Quality over quantity - if only 1 horse stands out, just pick 1

**Speed ratings are RELATIVE:**

Compare within the field only. If everyone is 0.98 and one horse is 0.99, that horse is best. No absolute thresholds like "must be 1.015+".

---

## Cost

~$0.025 per race (~6,500 tokens)
- 50 races/day = ~$1.25/day
- Uses Claude Sonnet 4 by default

---

## Integration with racing-tips-platform

The predictor is exposed via FastAPI server and called from the racing-tips-platform Next.js app.

### Architecture

```
racing-tips-platform (Next.js)     punt-legacy-ai (Python FastAPI)
        │                                    │
        │  POST /predict                     │
        │  {track, race_number, date}        │
        ├───────────────────────────────────►│
        │                                    │ → PuntingForm API
        │                                    │ → Ladbrokes API
        │                                    │ → Claude API
        │◄───────────────────────────────────┤
        │  {contenders[], summary}           │
```

### FastAPI Server

```bash
# Start the server
cd punt-legacy-ai
uvicorn server:app --host 0.0.0.0 --port 8000

# Endpoints
GET  /meetings?date=09-Jan-2026     # List tracks
GET  /races?track=Gosford&date=X    # List races at track
POST /predict                        # Generate prediction
```

### Modifying Claude Prompts

The prompts are in `core/predictor.py`:
- `SYSTEM_PROMPT` - The role/instructions for Claude
- `USER_PROMPT_TEMPLATE` - The race data template

Edit these directly - the FastAPI server uses them automatically.

### Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
PUNTINGFORM_API_KEY=...
```

### Running Locally

```bash
# Terminal 1: Start predictor API
cd punt-legacy-ai
pip install fastapi uvicorn
uvicorn server:app --reload --port 8000

# Terminal 2: Start frontend
cd racing-tips-platform
npm run dev
```

### Deployment

Deploy FastAPI server to Railway/Render/Fly.io with:
- Python 3.11+
- Environment variables set
- Port 8000 exposed

Set `PREDICTOR_API_URL` in racing-tips-platform to the deployed URL.

---

## Experimental Predictor (v2)

A simplified predictor in `experiments/bet_type_predictor.py` with cleaner logic:

### Key Differences from Live Predictor

| Feature | Live (`core/predictor.py`) | Experimental (`experiments/`) |
|---------|---------------------------|-------------------------------|
| Picks | 1-3 contenders always | 0-3 contenders |
| Skip races | Never | Yes - if 50%+ have no race form |
| Tags | Free-form | 3 fixed: "The one to beat", "Each-way chance", "Value bet" |
| Trial handling | Shown but counted | Explicitly excluded from analysis |
| Prompt style | Detailed instructions | Minimal - lets AI decide |

### Usage

```bash
# Single race
python experiments/bet_type_predictor.py --track "Randwick" --race 4 --date "19-Jan-2026"

# Backtesting (uses PuntingForm SP odds for finished races)
python experiments/backtest.py "Rosehill" 2 "17-Jan-2026"
```

### When It Skips a Race (0 Contenders)

The predictor will return 0 contenders when:
- **50%+ of field has no race form** - only barrier trials, can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

Example: Rosehill R1 on 17-Jan-2026 had only 3/9 runners with actual race form (rest were first starters with only trial form). The predictor correctly returned 0 contenders.

### Prompt Philosophy

```
Focus on **normalized speed ratings** from RACE runs (not trials) at similar
distance and conditions. More recent runs are more relevant.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form - horses don't always try
- If a horse has 0 race runs, they are UNKNOWN - could be brilliant or useless
- If 50%+ of field has no race form, pick 0 contenders - too many unknowns to assess
```

The prompt is intentionally minimal - tells Claude what data it has and what to focus on, but doesn't dictate specific thresholds or rules.

### Scratching Detection

The experimental predictor checks 3 things:
1. PuntingForm `scratched` field
2. Blank jockey name (with whitespace strip)
3. SP = $0

This catches late scratchings that APIs haven't formally flagged yet.

### Backtest Script

`experiments/backtest.py` allows backtesting on historical races:
- Uses PuntingForm Starting Price (SP) instead of Ladbrokes (finished races have no live odds)
- Verifies form data is from BEFORE race day (not cheating)
- Filters scratched horses properly

```bash
# Example backtest output
python experiments/backtest.py "Flemington" 2 "17-Jan-2026"

# BACKTEST: Flemington R2 - 17-Jan-2026
# Distance: 2520m | Condition: G4
# Form: 10/10 have race runs
#
# PREDICTION:
#   Tarvue (#6) $4.20 - "The one to beat"
#   Navy Heart (#9) $10.00 - "Each-way chance"
#   Scintillante (#5) $11.00 - "Value bet"
```

---

## Next Steps

1. ~~Historical backtesting~~ ✅ Done (experiments/backtest.py)
2. Prediction accuracy tracking
3. Switch live predictor to v2 approach
4. User customization options
