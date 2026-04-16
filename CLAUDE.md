# CLAUDE.md

## Project: Punt Legacy AI

AI-powered horse racing predictor product for Punt Legacy subscribers.

**Status:** Phase 5 Complete - V6 Predictor Optimized (46% TTOB win rate, +22% ROI)

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
- [x] Place odds consideration for each-way (win $4+, place $1.80+)

### Phase 5: Product ✅ COMPLETE
- [x] Build frontend to display predictions
- [x] User accounts (NextAuth)
- [x] Usage tracking (per-user daily limits)
- [x] Billing integration (Stripe)
- [x] Tipsheet generator (admin feature)
- [x] Auto-skip races with insufficient form data

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
│   ├── backtest.py             # Backtesting script ✅
│   └── ab_test.py              # A/B testing framework ✅
└── data/                  # Example responses, test data
    └── ab_results/        # A/B test results (JSON)
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

**Example runner data (V6 configuration):**

```
### 3. So You Ready
Barrier: 1 | Weight: 59kg
Odds: $3.30 win / $1.45 place → 30.3% implied
Jockey: Jay Ford (A/E: 0.49)
Career: 13: 1-1-0 (8% win)
**FIRST UP** (career 1st-up record: 3: 0-0-0)
Form: 4 race runs, 1 trials

| Date | Track | Dist | Cond | Adj | Prep | Trial |
|------|-------|------|------|-----|------|-------|
| 26-Dec | Beaumont | 2100m | G4 | 100.3 | 3 | - |
| 10-Dec | Wyong | 1600m | G4 | 99.0 | 2 | - |
| 20-Nov | Newcastle | 1400m | G4 | 97.2 | 1 | - |
| 05-Nov | Rosehill | 1000m | G3 | N/A | - | TRIAL |
| 12-Jun | Gosford | 2100m | S6 | 99.6 | 5 | - |
```

**V6 configuration (recommended):**
- **Adj column only** - Venue-adjusted rating (100 = benchmark). No Rating column.
- **No Pos/Margin columns** - Already captured in Adj rating
- **No Trainer A/E** - Not predictive enough (removed)
- Jockey A/E kept - More predictive than trainer
- Barrier, weight, career stats, prep pattern retained

**Data includes:**
- Barrier, weight (no age/sex - removed as not critical)
- Win AND place odds from Ladbrokes
- Jockey A/E ratio (trainer removed in V6)
- Career record + first-up/second-up record
- Form summary with race run count and trial count
- Last 10 runs with Adj rating, prep run number, and barrier trial flag

**Key principle:** Less noise = better predictions. V6 testing showed +30% ROI improvement over V0.

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

### Tags

The predictor uses these specific tags:
- **"The one to beat"** - Clear standout on ratings at similar distance and conditions vs the field
- **"Each-way chance"** - Good ratings at similar distance/conditions, place odds $1.80+
- **"Value bet"** - Odds $5.00+ AND ratings competitive with top of field (not just any longshot)
- **"Main danger"** - Second-best pick when there's a clear standout

---

## Design Philosophy

**Let Claude be the expert.**

The prompt is intentionally simple - it explains what the data means, but doesn't force Claude to follow rigid rules. Claude decides:
- How many contenders (0-3)
- What tags to use
- How to weight different factors (speed ratings, A/E, prep patterns, etc.)
- Whether to skip a race entirely (0 picks)

**Key data (V6 configuration - see A/B Testing Results):**
- **Adj column ONLY** (venue-adjusted ratings) - Primary data for comparing horses. Normalizes track quality so Randwick vs country tracks are comparable. Rating column is removed to reduce noise.
- **Jockey A/E** - A/E > 1.0 is positive, A/E < 0.85 is a red flag.
- **Trainer A/E** - Removed from V6 (not predictive enough).
- **Pos/Margin columns** - Removed from V6 (already baked into Adj rating).

**Why V6 works better:**
- Less noise = better focus on what matters (Adj ratings)
- Margin is already captured in the Adj rating
- Trainer A/E wasn't predictive enough to justify the noise
- Simpler data helps Claude identify clear standouts

**Why contenders instead of "BET/NO BET"?**

1. Lets users make their own betting decisions
2. Avoids rigid "bet/no bet" that frustrates users if a "no bet" wins
3. Claude can express nuance naturally ("best horse but short price")
4. Quality over quantity - if only 1 horse stands out, just pick 1

**Speed ratings are RELATIVE:**

Compare within the field only. If everyone is 98 and one horse is 99, that horse is best. No absolute thresholds like "must be 101.5+".

---

## Cost

~$0.025 per race (~6,500 tokens)
- 50 races/day = ~$1.25/day
- Uses Claude Sonnet 4 by default

### Cost-Saving Features

**Auto-skip for insufficient form data:**
- If >50% of runners have 0 race runs (only trials or first starters), the race is skipped
- Returns a friendly message without calling Claude API
- Skipped races don't count against user's daily limit
- Saves ~$0.025 per skipped race

**User tier limits:**
| Tier | Daily Limit |
|------|-------------|
| Free | 2 predictions |
| Basic ($9.99/mo) | 6 predictions |
| Pro ($29.99/mo) | 50 predictions |

---

## Tipsheet Pick Flag

Each contender includes a `tipsheet_pick: bool` field indicating whether Claude would genuinely bet on this horse.

**When `tipsheet_pick = true`:**
- Speed ratings clearly support this horse vs the field
- The odds represent real value (not just "best of a bad bunch")
- Claude is confident in the pick

**Usage:**
- Frontend shows ⭐ star icon next to tipsheet picks
- Admin tipsheet generator highlights these for daily tipsheet creation
- Useful for filtering "must-back" picks from "worth considering" picks

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

# Core Endpoints
GET  /meetings?date=09-Jan-2026     # List tracks
GET  /races?track=Gosford&date=X    # List races at track
GET  /odds?track=X&race_number=N&date=X  # Live Ladbrokes odds (for manual tip page)
POST /predict                        # Generate prediction (accepts allow_finished for past races)
POST /predict-meeting                # Generate predictions for entire meeting (admin)
POST /backtest                       # Run backtest on historical races

# Stats Endpoints
GET  /stats/summary                  # Overall prediction stats
GET  /stats/by-tag                   # Performance by tag
GET  /stats/by-tag-staking           # Performance by tag with staking ROI
GET  /stats/by-meeting               # Performance by meeting (track + date)
GET  /stats/by-confidence            # Performance by confidence level
GET  /stats/by-race-confidence       # Performance by race-level confidence
GET  /stats/by-mode                  # Performance by mode (normal vs promo)

# Tracking Endpoints
GET  /predictions/pending            # Predictions awaiting outcomes
GET  /predictions/recent?limit=50    # Recent predictions
POST /outcomes                       # Record race outcomes
POST /outcomes/sync?race_date=X      # Auto-sync outcomes from PuntingForm
POST /backfill                       # Import historical predictions
```

#### Stats by Meeting Response
```json
GET /stats/by-meeting
[
  {
    "track": "Canterbury",
    "date": "16-Jan-2026",
    "total_picks": 5,
    "wins": 1,
    "places": 2,
    "win_rate": 0.2,
    "place_rate": 0.4,
    "flat_profit": -2.5,
    "by_tag": {
      "The one to beat": {"total": 2, "wins": 1, "places": 1, "profit": 1.5},
      "Each-way chance": {"total": 2, "wins": 0, "places": 1, "profit": -2.0},
      "Value bet": {"total": 1, "wins": 0, "places": 0, "profit": -1.0}
    }
  }
]
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

## Admin Replay (Marketing Videos)

Admin can re-run the predictor on **past/finished races** for marketing content creation. Instead of recording every race live, admin can find a winner, re-run the predictor on that race, and screen-record the prediction appearing.

**How it works:**
- Admin uses `/ai-predictor` and picks any past date
- The frontend sends `allow_finished: true` to the `/predict` endpoint (admin only)
- The Ladbrokes race status check ("has started"/"has finished") is bypassed
- Ladbrokes SP (Starting Price) odds are used instead of live pre-race odds
- Everything else is identical: same form data, same prompt, same Claude model

**Limitations:**
- Only works if Ladbrokes still has the meeting data (typically a few weeks)
- Odds shown will be SP (final starting prices), not pre-race prices — usually very close
- Non-admin users are unaffected — they still can't predict finished races

**Files:**
- `api/ladbrokes.py` - `allow_finished` param on `get_odds_for_pf_track()`
- `core/race_data.py` - `allow_finished` param on `get_race_data()`
- `server.py` - `allow_finished` field on `PredictionRequest`

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

# A/B testing multiple variations (see ab_test.py)
python experiments/ab_test.py
```

### When It Skips a Race (0 Contenders)

The predictor will return 0 contenders when:
- **50%+ of field has no race form** - only barrier trials, can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

Example: Rosehill R1 on 17-Jan-2026 had only 3/9 runners with actual race form (rest were first starters with only trial form). The predictor correctly returned 0 contenders.

### Prompt Philosophy

The prompt (in `core/predictor.py`) explains:
- **Rating scale**: 100 = benchmark, higher = faster
- **Form table columns**: What Dist%, CStep, WtCh, Prep mean
- **Comparison factors**: Distance, conditions, prep stage, weight
- **Critical rules**: Trials don't count, 50%+ unknowns = skip race

The prompt tells Claude what the data means and what factors to compare, but doesn't dictate specific thresholds. Claude decides how to weigh factors and which horses to pick.

See `SYSTEM_PROMPT` in `core/predictor.py` for the full prompt.

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

## Tipsheet Generator (Admin Feature)

Generate predictions for an entire meeting at once via `POST /predict-meeting`:

```bash
curl -X POST http://localhost:8000/predict-meeting \
  -H "Content-Type: application/json" \
  -d '{"track": "Randwick", "date": "22-Jan-2026", "race_start": 1, "race_end": 8}'
```

**Response includes:**
- All race predictions (same format as `/predict`)
- `tipsheet_picks[]` - Just the ⭐ picks across all races
- `total_races`, `races_with_picks`, `estimated_cost`

**Frontend features (`/admin/tipsheet`):**
- Saves generated tipsheets to localStorage (persists 7 days)
- Load/delete previous tipsheets without regenerating
- Copy button formats tipsheet for Instagram/social

---

---

## Claude Code Analysis (Free Alternative)

Export race data and paste to Claude Code for **free analysis** without Claude API costs.

### Quick Start

```bash
cd /Users/danielsamus/punt-legacy-ai

# Single race
python3 tools/export_for_claude_code.py "Randwick" 5 22-Mar-2026

# Range of races
python3 tools/export_for_claude_code.py "Randwick" 3-7 22-Mar-2026

# All races at meeting
python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026

# Skip instructions header (if pasting multiple batches)
python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026 --no-instructions
```

### What It Outputs

The script outputs:
1. **Analysis instructions** - Same methodology as the API predictor
2. **Race data** - Exactly what Claude API would see

```
## ANALYSIS INSTRUCTIONS
You are an expert horse racing analyst...

---

# Randwick Race 5: [Race Name]
Distance: 1200m | Condition: G4 | Class: BM72
Field Size: 12 | Pace: moderate (2 leaders)

## Runners

### 1. Horse Name
Odds: $4.50 win / $1.80 place → 22.2% implied
Jockey: J Smith (A/E: 1.12)
...

| Date | Track | Dist | Cond | Adj | Prep | Trial |
|------|-------|------|------|-----|------|-------|
| 15-Mar | Canterbury | 1200m | G4 | 100.5 | 3 | - |
```

**Note:** V6 configuration shows only Adj column (no Rating, Pos, or Margin columns).

### Workflow

```bash
# 1. Export data
python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026 > /tmp/races.txt

# 2. Copy to clipboard (macOS)
cat /tmp/races.txt | pbcopy

# 3. Paste to Claude Code and ask:
"Analyze these races and pick your best bets"
```

### Rating Columns Explained

| Column | What It Means |
|--------|---------------|
| **Adj** | Venue-adjusted rating. Normalized by distance, condition, AND track quality. Makes ratings comparable across all venues (e.g., Randwick vs Yass). **This is the primary column - use this.** 100 = benchmark. Higher = faster. |

**Note:** The V6 predictor configuration shows only Adj (no Rating column). Rating was removed because Adj is strictly better - it includes all the same normalization plus venue quality adjustment.

### Analysis Methodology (Simplified)

**Goal:** Scan races and find clear standouts on ratings. Not every race has a play.

#### Step 1: Look at recent Adj ratings at similar-ish distance/condition

For each horse, find their **most recent runs** at similar distance and conditions to today's race.

- **Recency matters most** - last 1-3 runs are far more relevant than older runs
- If the most recent run was at a completely different distance/condition, look at the 2nd/3rd/4th most recent run that IS relevant
- Similar-ish distance is fine

#### Step 2: Find the standout

Compare the recent relevant Adj ratings across the field:

#### Step 3: Weight relief is a bonus

**Don't overanalyze weight.** But if a horse with standout ratings ALSO has a significant weight drop, that's a bonus signal - especially for roughies.

#### Understanding the ratings

- **100 = expected speed** for that distance/condition
- **Higher Adj = better** regardless of finishing position
- **Ignore margin** - already baked into the rating
- **⚠️eased** = horse wasn't fully pushed, actual ability likely higher
- **TRIAL runs don't count** - horses don't try in trials

#### Step 4: Check the Notes column

The **Notes** column contains stewards reports - always check these for context on why a rating might be misleading:

- **"Missed the jump"** - slow start, rating underestimates ability
- **"Ran wide without cover"** - burned extra energy, true ability higher
- **"Held up"** - couldn't get clear run, unlucky
- **"Eased down"** - jockey stopped pushing, horse wasn't tested
- **"Contacted at start"** - interference affected performance

**Example:** Horse has 102, 101, 98 Adj ratings. The 98 has notes "missed jump, wide throughout" - that run was compromised. True form is closer to 101-102.

### What Makes a Value Bet

A roughie ($10+) with:
1. **Standout recent Adj** - clearly among the best in the field
2. **Weight relief** (bonus, not required)
3. **Odds that don't reflect the ratings**

### Common Mistakes to Avoid

1. ❌ Looking at best-ever rating instead of **recent** ratings
2. ❌ Being too rigid on exact distance
3. ❌ Ignoring a horse because they haven't run the exact distance
4. ❌ Overthinking weight - it's a bonus, not the primary factor
5. ❌ Trying to find a bet in every race - skip if no clear standout

### How to Use Claude Code for Predictions

**Step 1: Export the data**
```bash
cd /Users/danielsamus/punt-legacy-ai
python3 tools/export_for_claude_code.py "Rosehill" all 22-Mar-2026 > /tmp/races.txt
cat /tmp/races.txt | pbcopy
```

**Step 2: Start a new Claude Code chat and paste the data**

**Step 3: Ask Claude to analyze** - Example prompt:
```
Analyze these races and pick 0-3 contenders per race.

For each race:
1. Look at Adj ratings at SIMILAR distance and conditions to the race
2. More recent runs = more relevant
3. Ignore TRIAL runs (horses don't try)
4. Mark your strongest pick as "tipsheet_pick" = ⭐

Output format per race:
- The one to beat: [horse] @ $X.XX - [2-3 sentence analysis referencing ratings]
- Each-way chance: [horse] @ $X.XX - [reason]
- Value bet: [horse] @ $X.XX - [reason]
```

### Backtest Results (Mar 21, 2026 Rosehill - 9 races)

| Metric | Result |
|--------|--------|
| Winners | 4/9 (44%) |
| ROI (flat stake) | **+42.8%** |
| Top picks placed (1-4) | 7/9 |



### Files

- `tools/export_for_claude_code.py` - Export script
- `core/race_data.py` - Data pipeline (uses `to_prompt_text(include_venue_adjusted=True)`)
- `core/predictor.py` - Original prompts (SYSTEM_PROMPT)

---

## Daily Picks Workflow (Manual with Claude Code)

A simple morning routine for generating daily subscriber picks using Claude Code.

### The Approach

**One rule:** Pick ONLY when there's a clear standout based on recent Adj ratings at similar distance/condition, AND the odds are worth it (not too short).

**What to skip:**
- Races with limited form data (too many first starters/unknowns)
- Races where multiple horses have similar ratings (no clear standout)
- Clear standout but odds too short (e.g., $1.20 for marginal edge)

### Step-by-Step

```bash
# 1. Export data for a meeting
cd /Users/danielsamus/punt-legacy-ai
source .env && python3 tools/export_for_claude_code.py "Pakenham" all 19-Mar-2026 --no-instructions

# 2. Paste output to Claude Code and ask:
```

**Prompt to use:**
```
Look at every race. For each one, tell me:
1. Is there a CLEAR standout based on recent Adj ratings at similar distance/condition?
2. If yes - are the odds worth betting?

Only pick if BOTH are true. One pick max per race.

Skip races where:
- Not enough form data to judge
- Multiple horses have similar ratings (no clear edge)
- Standout exists but odds too short for the edge

Output format:
R1: SKIP - [reason]
R2: BET - Horse Name @ $X.XX - [1 sentence: why they're the standout on ratings]
R3: SKIP - [reason]
...
```

### What "Clear Standout" Means

Look at the Adj column for runs at similar distance and condition:

| Scenario | Decision |
|----------|----------|
| Horse A: 101, 102, 101. Horse B: 99, 98, 100 | BET Horse A - clearly better |
| Horse A: 101, 99. Horse B: 100, 101 | SKIP - too close |
| Horse A: 102 but $1.30. Horse B: 99 at $5 | Consider skipping - edge doesn't justify $1.30 |
| 4/8 horses have no form data | SKIP - too many unknowns |

### What NOT to Focus On

- ❌ Whether they've been winning/placing lately (ratings already tell you performance)
- ❌ Jockey/trainer stats (minor factor)
- ❌ Gear changes (noise)
- ❌ Barrier draws (minor except extreme cases)
- ❌ Best-ever rating (use RECENT form)

### Backtest Results

**Cranbourne 20-Mar-2026 (7 races):**
- 2 picks made: Luigi The Brave $1.75, Finance Shogun $5.00
- 5 races skipped (no clear standout or limited form)
- Result: **2/2 winners (100%)**

---

## Manual Analysis Methodology

**See `PREDICTIONS.md`** for the manual race analysis methodology used when Claude Code analyzes races.

---

## AI Predictor Performance Analysis (April 2026)

Comprehensive analysis of ~4000+ predictions from Feb 3 - Apr 1, 2026.

### Quick Stats Commands

```bash
# From racing-tips-platform directory:
npm run stats                      # Overall summary
npm run stats -- --by-tag          # Performance by tag
npm run stats -- --by-tipsheet     # ⭐ Starred vs regular picks
npm run stats -- --pending         # Predictions still awaiting outcomes
npm run sync-outcomes              # Sync results from PuntingForm

# Direct API queries (Railway production):
curl -s "https://punt-legacy-ai-production.up.railway.app/stats/summary"
curl -s "https://punt-legacy-ai-production.up.railway.app/stats/by-tag"
curl -s "https://punt-legacy-ai-production.up.railway.app/stats/by-pfai-rank-metro?tag=The%20one%20to%20beat&metro=true"
```

### Overall Performance Summary

| Metric | Value |
|--------|-------|
| Total predictions | ~4100 |
| Results recorded | ~3400 |
| Still pending | ~700 (mostly small country tracks) |

### Performance by Tag (All Tracks)

| Tag | Picks | Win % | Place % | Avg Odds | ROI |
|-----|-------|-------|---------|----------|-----|
| The one to beat | 852 | 32.6% | 61.0% | $3.51 | -6.2% |
| Each-way chance | 619 | 14.5% | 41.2% | $6.19 | -14.2% |
| Main danger | 230 | 22.2% | 53.5% | $4.22 | -4.0% |
| Value bet | 405 | 9.1% | 32.1% | $9.82 | -37.7% |
| Bonus Bet | 45 | 22.2% | 48.9% | $8.31 | +67.3% |

**Key insight:** "Bonus Bet" tag (longshots at $5+) is profitable but low volume.

### Metro vs Non-Metro Performance

**Metro tracks:** Randwick, Rosehill, Canterbury, Warwick Farm, Flemington, Caulfield, Moonee Valley, Sandown, Pakenham, Eagle Farm, Doomben, Gold Coast, Morphettville, Ascot, Belmont, Hobart, Launceston

#### The One to Beat

| Location | Picks | Win % | ROI |
|----------|-------|-------|-----|
| Metro | 350 | 34.6% | **-1.9%** |
| Non-metro | 502 | 31.3% | -9.2% |

**Metro is nearly breakeven** for main picks.

### PFAI Rank Analysis

PFAI = PuntingForm AI ranking (1 = their top pick). Used to filter non-metro tipsheet stars.

#### The One to Beat @ Metro by PFAI Rank

| PFAI Rank | Picks | Win % | Avg Odds | ROI |
|-----------|-------|-------|----------|-----|
| #1 | 134 | 34.3% | $3.17 | -8.4% |
| #2 | 85 | 43.5% | $3.33 | **+12.0%** |
| #3 | 34 | 35.3% | $3.67 | -4.1% |
| #4 | 39 | 20.5% | $4.32 | -29.7% |
| #5 | 32 | 34.4% | $3.75 | +13.0% |
| **1-3 Total** | **253** | **37.5%** | - | **-1.0%** |
| **4+ Total** | **97** | **26.8%** | - | **-4.4%** |

#### The One to Beat @ Non-Metro by PFAI Rank

| PFAI Rank | Picks | Win % | Avg Odds | ROI |
|-----------|-------|-------|----------|-----|
| #1 | 141 | 38.3% | $3.16 | +8.7% |
| #2 | 107 | 35.5% | $3.18 | -2.7% |
| #3 | 68 | 27.9% | $4.06 | -8.2% |
| #4 | 74 | 25.7% | $3.76 | **-43.4%** |
| #5+ | 100+ | 20-25% | $4-5 | **-40% to -51%** |
| **1-3 Total** | **316** | **34.8%** | - | **+0.1%** |
| **4+ Total** | **186** | **25.8%** | - | **-22.0%** |

**Key insight:** Non-metro PFAI #4+ is disaster zone (-40% to -50% ROI). This is why the tipsheet star filter exists.

### Other Tags @ Metro by PFAI Rank

#### Each-Way Chance @ Metro

| Filter | Picks | Win % | ROI |
|--------|-------|-------|-----|
| All | 331 | 16.0% | -7.9% |
| PFAI 1-3 | 196 | 19.9% | **+2.2%** |
| PFAI 4+ | 135 | 10.4% | -22.6% |

**Recommendation:** Consider adding PFAI 1-3 filter for Each-way chance on metro too.

#### Value Bet @ Metro

| Filter | Picks | Win % | ROI |
|--------|-------|-------|-----|
| All | 214 | 9.3% | -47.0% |
| PFAI 1-3 | 68 | 13.2% | -33.7% |
| PFAI 4+ | 146 | 7.5% | -53.3% |

**Note:** Value bet is terrible across the board. PFAI filter helps slightly but still deeply negative.

#### Main Danger @ Metro

| Filter | Picks | Win % | ROI |
|--------|-------|-------|-----|
| All | 115 | 30.4% | **+6.1%** |
| PFAI 1-3 | 73 | 26.0% | -16.8% |
| PFAI 4+ | 42 | 38.1% | +46.0% |

**Note:** Weird result - PFAI 4+ does better. Small sample likely explains this anomaly.

### Tipsheet Star (⭐) Logic

The `tipsheet_pick` flag indicates picks Claude would genuinely bet on.

**Metro tracks:** Claude's tipsheet_pick is used as-is (no PFAI filter)

**Non-metro tracks:** For "The one to beat" only:
```python
if tipsheet_pick and tag == "The one to beat":
    if not is_metro_track(track):
        if pfai_rank is None or pfai_rank > 3:
            tipsheet_pick = False  # Remove the star
```

So non-metro "The one to beat" needs BOTH:
1. ✅ Claude thinks it's value at the price
2. ✅ PFAI agrees (rank 1, 2, or 3)

**Why this filter?** Non-metro PFAI #4+ had -43% to -51% ROI - pure noise.

### Data Sources

- **Predictions stored:** Railway SQLite (`punt-legacy-ai-production.up.railway.app`)
- **Results from:** PuntingForm API via `/outcomes/sync`
- **PFAI ranks:** Backfilled from PuntingForm via `/backfill/pfai-rank`
- **Tracking start date:** Feb 3, 2026

### API Endpoints for Analysis

```bash
# Overall stats
GET /stats/summary

# By tag
GET /stats/by-tag

# By PFAI rank with metro filter
GET /stats/by-pfai-rank-metro?tag=The%20one%20to%20beat&metro=true
GET /stats/by-pfai-rank-metro?tag=Each-way%20chance&metro=true
GET /stats/by-pfai-rank-metro?tag=Value%20bet&metro=true

# Sync outcomes for a date
POST /outcomes/sync?race_date=01-Apr-2026

# Backfill PFAI ranks (if missing)
POST /backfill/pfai-rank
```

### Key Recommendations

1. **Metro "The one to beat"** - Nearly breakeven (-1.9%), no filter needed
2. **Non-metro "The one to beat"** - Keep PFAI 1-3 filter (turns -22% into +0.1%)
3. **Each-way chance @ Metro** - Consider adding PFAI 1-3 filter (+2.2% vs -22.6%)
4. **Value bet** - Avoid regardless of filters
5. **Bonus Bet** - Small sample but profitable (+67%), worth tracking

---

### Optimal Betting Strategy (April 2026)

Comprehensive analysis of 1,186 "The one to beat" picks (Feb 3 - Apr 1, 2026).

#### The Promo Edge

"The one to beat" has a high 2nd/3rd rate, making it ideal for Money Back 2nd/3rd promos:

| Outcome | Rate | Without Promo | With Promo |
|---------|------|---------------|------------|
| 1st (WIN) | 31% | Keep profit | Keep profit |
| 2nd | 22% | Lose stake | **Get bonus back** |
| 3rd | 15% | Lose stake | **Get bonus back** |
| 4th+ | 35% | Lose stake | Lose stake |

**Result:** ~37% of bets return a bonus bet (worth ~75% of stake)

| Scenario | ROI |
|----------|-----|
| Without promos | -14.9% |
| **With promos** | **+12.9%** |

#### By Day of Week

| Day | Picks | Win % | 2nd/3rd % | ROI | +Promo |
|-----|-------|-------|-----------|-----|--------|
| **Saturday** | 313 | 33.5% | 34.5% | -4.1% | **+21.8%** ✅ |
| Monday | 75 | 33.3% | 30.7% | -5.1% | +17.9% ✅ |
| Wednesday | 211 | 29.4% | 37.4% | -15.8% | +12.2% ✅ |
| Tuesday | 126 | 30.2% | 36.5% | -15.8% | +11.6% ✅ |
| Thursday | 127 | 29.1% | 37.0% | -22.0% | +5.7% ⚠️ |
| Sunday | 160 | 28.1% | 28.7% | -22.5% | -0.9% ❌ |
| Friday | 174 | 24.7% | 36.2% | -30.9% | -3.8% ❌ |

#### By Track Type

| Type | Picks | Win % | 2nd/3rd % | ROI | +Promo |
|------|-------|-------|-----------|-----|--------|
| **Metro** | 497 | 32.2% | 36.0% | -6.2% | **+20.8%** ✅ |
| Non-Metro | 689 | 28.3% | 33.8% | -22.8% | +2.6% ⚠️ |

#### By Odds Range

| Odds | Picks | Win % | 2nd/3rd % | ROI | +Promo |
|------|-------|-------|-----------|-----|--------|
| **$2.00-$2.99** | 395 | 38.7% | 36.7% | -3.7% | **+23.9%** ✅ |
| $3.00-$4.99 | 450 | 22.0% | 38.0% | -19.7% | +8.8% ⚠️ |
| Under $2 | 160 | 52.5% | 28.1% | -14.5% | +6.6% ⚠️ |
| $5.00+ | 181 | 10.5% | 28.2% | -34.0% | -12.8% ❌ |

#### Best Combos

| Combo | Picks | Win % | 2nd/3rd % | ROI | +Promo |
|-------|-------|-------|-----------|-----|--------|
| **Sat Metro Under $3** | 112 | **53.6%** | 29.5% | **+21.3%** | **+43.4%** 🔥 |
| Sat Metro All | 236 | 35.6% | 34.3% | +3.2% | +28.9% ✅ |
| Friday Metro | 38 | 23.7% | 50.0% | -10.0% | +27.5% ✅ |
| Wed Non-Metro | 55 | 32.7% | 47.3% | -12.9% | +22.6% ✅ |

#### Simple Rules

**ALWAYS BET (with promos):**
- ✅ Saturday + Metro + Under $3 odds → **+43.4% ROI**
- ✅ Saturday + Metro (any odds) → **+28.9% ROI**

**GOOD TO BET (with promos):**
- ✅ Any Metro track → +20.8%
- ✅ $2-$3 odds range → +23.9%
- ✅ Mon/Tue/Wed → +11-18%

**AVOID:**
- ❌ Friday (especially non-metro) → negative even with promos
- ❌ Sunday → breakeven at best
- ❌ $5+ odds → -12.8% even with promos
- ❌ Non-metro without promos → -22.8%

#### The Optimal Strategy

> **Bet "The one to beat" on Saturday metro races under $3, always with Money Back 2nd/3rd promos.**
>
> - 112 picks sample
> - 53.6% win rate
> - +21.3% ROI without promos
> - **+43.4% ROI with promos**

---

### Predictor Upgrade (April 14, 2026)

Deployed significant predictor improvements after backtesting:

**Changes:**
1. **Venue-adjusted ratings (Adj column)** - Now the primary data. Normalizes track quality so ratings are comparable across all venues.
2. **TTOB definition tightened** - "Clear standout on ratings at similar DISTANCE and CONDITIONS vs the field"
3. **Value bet definition tightened** - Must have "ratings competitive with top of field", not just any longshot
4. **Jockey A/E emphasis** - A/E < 0.85 flagged as red flag, trainer A/E de-emphasized

**Backtesting results:**
| Test | OLD | NEW |
|------|-----|-----|
| Metro Apr 11 (Randwick, Caulfield, Doomben) | 4/21 (19%) | **8/25 (32%)** |
| Non-metro Apr 9-10 (extra picks) | - | 9/14 winners, +6.70 units |
| Eagle Farm Apr 8 | VB Saint Aldwyn $8.50 | **WON** |

**Key finding:** Venue-adjusted ratings help most on metro tracks where track quality varies significantly.

---

### A/B Testing Results (April 16, 2026)

Comprehensive A/B testing of 188 races comparing 5 predictor variations.

#### Variations Tested

| ID | Name | Description |
|----|------|-------------|
| **V0** | Production | Current live predictor (from Railway database) |
| **V1** | Baseline | Rating + Adj columns, Pos/Margin, full trainer/jockey A/E |
| **V3** | No Pos/Margin | Same as V1 but removes Pos and Margin columns from form table |
| **V6** | Lean | **WINNER** - Adj only (no Rating), no Pos/Margin, no Trainer A/E, full prompt |
| **V7** | Lean + Minimal | Same as V6 but with minimal prompt |

#### V6 Configuration (Recommended)

The winning configuration simplifies the data Claude receives:

| Feature | V0/V1 (Old) | V6 (New) |
|---------|-------------|----------|
| Rating column | ✅ Shown | ❌ Removed |
| Adj column | ✅ Shown | ✅ **Primary data** |
| Pos column | ✅ Shown | ❌ Removed |
| Margin column | ✅ Shown | ❌ Removed |
| Jockey A/E | ✅ Shown | ✅ Shown |
| Trainer A/E | ✅ Shown | ❌ Removed |
| Prompt style | Detailed | Full (same as V1) |

**Why V6 works:** Less noise = better focus. The Adj column already captures performance (margin is baked in). Removing redundant columns helps Claude identify clear standouts.

#### Results: The One To Beat (188 Races)

| Variation | Picks | Win % | Place % | Avg Odds | ROI |
|-----------|-------|-------|---------|----------|-----|
| **V6: Lean** | 150 | **46.0%** | 74.0% | $3.19 | **+22.2%** ✅ |
| V7: Lean+Minimal | 185 | 41.6% | 67.6% | $3.55 | +19.6% |
| V3: No Pos/Margin | 162 | 43.2% | 72.2% | $3.26 | +19.6% |
| V1: Baseline | 174 | 37.4% | 73.6% | $3.29 | +2.0% |
| **V0: Production** | 171 | 33.3% | 62.6% | $3.56 | **-8.7%** ❌ |

**V6 vs V0 improvement:**
- Win rate: 33.3% → 46.0% (+12.7%)
- ROI: -8.7% → +22.2% (+30.9%)

#### Results by Location (TTOB)

**Metro (73 races):**
| Variation | Picks | Win % | ROI |
|-----------|-------|-------|-----|
| V3: No Pos/Margin | 66 | 43.9% | **+28.6%** |
| V7: Lean+Minimal | 73 | 39.7% | +16.8% |
| V6: Lean | 57 | 42.1% | +14.3% |
| V1: Baseline | 71 | 35.2% | -2.4% |
| V0: Production | 65 | 30.8% | -11.6% |

**Non-Metro (115 races):**
| Variation | Picks | Win % | ROI |
|-----------|-------|-------|-----|
| **V6: Lean** | 93 | **48.4%** | **+27.0%** |
| V7: Lean+Minimal | 112 | 42.9% | +21.5% |
| V3: No Pos/Margin | 96 | 42.7% | +13.5% |
| V1: Baseline | 103 | 38.8% | +5.0% |
| V0: Production | 106 | 34.9% | -6.9% |

**Key insight:** V6 dominates non-metro (+27.0% ROI). V3 wins metro (+28.6% ROI).

#### Results: Value Bet (188 Races)

| Variation | Picks | Win % | Avg Odds | ROI |
|-----------|-------|-------|----------|-----|
| **V6: Lean** | 156 | 12.8% | $12.54 | **+6.7%** ✅ |
| V1: Baseline | 130 | 12.3% | $11.80 | -12.6% |
| V3: No Pos/Margin | 140 | 10.7% | $12.17 | -18.2% |
| V7: Lean+Minimal | 187 | 11.2% | $12.50 | -27.3% |
| V0: Production | 65 | 7.7% | $9.65 | -49.8% |

**V6 is the only profitable variation for Value bets.**

#### Results: All Picks Combined (188 Races)

| Variation | Picks | Win % | ROI |
|-----------|-------|-------|-----|
| V6: Lean | 458 | 23.6% | **-2.6%** |
| V7: Lean+Minimal | 551 | 23.2% | -7.8% |
| V3: No Pos/Margin | 475 | 23.2% | -9.4% |
| V0: Production | 470 | 24.0% | -10.8% |
| V1: Baseline | 489 | 22.9% | -13.1% |

#### Test Data Files

Results stored in `data/ab_results/`:
- `ab_test_v6v7_20260415_223245.json` - 88 races (V6, V7 only)
- `ab_test_20260415_143936.json` - 36 races (V1-V5)
- `ab_test_20260415_171131.json` - 52 races (V1-V5)
- `ab_test_v1367_20260416_070633.json` - 50 races (V1, V3, V6, V7)
- `ab_test_balanced_20260416_095104.json` - 50 races (V1, V3, V6, V7) - 25 metro, 25 non-metro

#### Backtesting Methodology

The backtesting is legitimate:
1. **Form data** comes from PuntingForm API - only historical runs before race day
2. **Predictions** made by Claude using historical data only
3. **Results** fetched separately from actual race outcomes
4. **No future data** used in predictions

#### Recommendations

1. **Deploy V6 to production** - Best overall performer
2. **TTOB is the star tag** - 46% win rate, +22% ROI
3. **Value bet works with V6** - Only profitable variation (+6.7%)
4. **Non-metro is where V6 shines** - 48.4% win rate, +27% ROI
5. **Consider V3 for metro-only** - Slightly better at +28.6% ROI

---

### V6 Deployed to Production (April 16, 2026)

V6 is now live on Railway. All new predictions use the V6 configuration.

**What changed:**
- Adj column only (no Rating, Pos, Margin)
- No Trainer A/E display (keeps Jockey A/E)
- Simplified SYSTEM_PROMPT

**Backwards compatible:** Set `v6_mode=False` in `to_prompt_text()` to revert.

**Stats note:** Existing 3899 predictions are V0. New predictions are V6. Compare performance over time.

### Pending Improvements

- [x] Deploy V6 configuration to production ✅ (April 16, 2026)
- [ ] Add PFAI filter for Each-way chance at metro
- [ ] Track starred vs non-starred performance separately
- [ ] Build automated weekly performance reports
- [ ] Consider hybrid approach (V3 for metro, V6 for non-metro)

---

## Next Steps

1. ~~Historical backtesting~~ ✅ Done (experiments/backtest.py)
2. ~~Prediction accuracy tracking~~ ✅ Done (tracking endpoints)
3. ~~Switch live predictor to v2 approach~~ ✅ Done (auto-skip, tipsheet_pick)
4. ~~Tipsheet generator~~ ✅ Done (/admin/tipsheet)
5. ~~Claude Code export~~ ✅ Done (tools/export_for_claude_code.py)
6. ~~Daily Picks Workflow~~ ✅ Done (manual Claude Code routine)
7. ~~Simple Daily Picks System~~ ✅ Done (condition proximity documented)
8. ~~Performance analysis~~ ✅ Done (April 2026 - see above)
9. ~~A/B Testing~~ ✅ Done (April 16, 2026 - 188 races, V6 wins)
10. **Deploy V6 to production** - 46% win rate, +22% ROI on TTOB
11. Performance dashboard improvements
12. User customization options
13. Consider hybrid V3/V6 (V3 for metro, V6 for non-metro)
