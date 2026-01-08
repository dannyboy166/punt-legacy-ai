# CLAUDE.md

## Project: Punt Legacy AI

AI-powered horse racing predictor product for Punt Legacy subscribers.

**Status:** Phase 1 - API Documentation & Data Foundation

---

## Project Vision

A subscription product where users can:
1. Get AI-powered race predictions
2. Customize which factors the AI emphasizes
3. Free tier (2-3 races/day) → Paid tiers for more

**Key principle:** Every calculation must be 100% correct and well-documented.

---

## Development Phases

### Phase 1: API Documentation (CURRENT)
- [ ] Document ALL PuntingForm API endpoints
- [ ] Document ALL Ladbrokes API endpoints
- [ ] Test each endpoint, save example responses
- [ ] Identify all available data fields

### Phase 2: Core Calculations
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

## Project Structure

```
punt-legacy-ai/
├── CLAUDE.md           # This file - project guidance
├── api/                # API clients
│   ├── puntingform.py  # PuntingForm API wrapper
│   └── ladbrokes.py    # Ladbrokes API wrapper
├── core/               # Core calculations
│   ├── speed.py        # Speed rating calculations
│   ├── time.py         # Time/margin calculations
│   └── normalize.py    # Distance/condition normalization
├── tests/              # Unit tests for everything
├── docs/               # API documentation
│   ├── puntingform_api.md
│   └── ladbrokes_api.md
└── data/               # Example responses, test data
```

---

## Guiding Principles

1. **100% Correctness** - Every calculation must be verified with tests
2. **Documentation First** - Document what we're building before building it
3. **Ask When Unsure** - If unclear about data/logic, ask before implementing
4. **Clean Code** - This is a product, not a prototype
5. **Test Everything** - No untested code in core/

---

## Related Projects

- `/Users/danielsamus/pfai-tracker` - Original research/prototyping (reference)
- `/Users/danielsamus/racing-tips-platform` - Punt Legacy website

---

## APIs

### PuntingForm API
- **Base:** `https://api.puntingform.com.au/v2/`
- **Auth:** `?apiKey=YOUR_KEY`
- **Docs:** https://docs.puntingform.com.au/reference/meetingslist
- **Key:** Stored in `.env` as `PUNTINGFORM_API_KEY`

### Ladbrokes API
- **Base:** `https://api.ladbrokes.com.au/affiliates/v1/racing/`
- **Auth:** None required (public affiliate API)

### Claude API
- **Key:** Stored in `.env` as `ANTHROPIC_API_KEY`

---

## Speed Rating Formula (Reference)

From pfai-tracker - to be reimplemented with full testing:

```python
# 1. Calculate horse's finishing time
seconds_per_length = 2.4 / winner_speed  # Dynamic based on winner's speed
horse_time = winner_time + (margin * seconds_per_length)

# 2. Calculate actual speed
actual_speed = distance / horse_time  # meters per second

# 3. Get expected speed (normalized baseline)
expected_speed = baseline_speed(distance) * condition_multiplier(track_condition)

# 4. Calculate rating
rating = actual_speed / expected_speed
# > 1.0 = faster than average for distance/condition
# < 1.0 = slower than average
```

**Critical:** This formula must be thoroughly tested before use.

---

## Next Steps

1. Open new chat in this project directory
2. Go through each PuntingForm API endpoint
3. Document available fields
4. Save example responses
5. Identify what's useful for predictions
