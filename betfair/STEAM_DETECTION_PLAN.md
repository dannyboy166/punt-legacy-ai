# Steam Detection & Late Money System — Build Plan

## Status: Planning (May 5, 2026)

Live API key applied for (£499). Once approved, build this system.

---

## What We're Building

A system that combines AI form picks with Betfair market confirmation:
1. AI predictor picks 1-3 contenders per race (already working)
2. Stream live Betfair data for those horses from T-15 minutes
3. Detect steam moves (rapid price shortening with volume)
4. Only bet when form analysis AND market money agree

---

## Architecture

```
AI Predictor (existing)
    │ picks contenders
    ▼
Streaming Layer (betfairlightweight)
    │ real-time price + volume updates
    ▼
Signal Processing (MarketSignals)
    │ steam detection, WOM, volume spikes
    ▼
Decision Engine
    │ form pick + market confirmation = BET
    │ form pick + no confirmation = SKIP
    ▼
Bet Placement (BetPlacer - existing)
    │ safety limits, logging
    ▼
CSV/DB logging for backtesting
```

---

## Key Signals to Detect

### 1. Steam Move (LTP Shortening)
- Price drops 3+ ticks in under 2 minutes
- Must be sustained (doesn't bounce back within 30s)
- More significant in final 5 minutes before race

### 2. Volume Confirmation (totalMatched)
- New money matched on runner in last 60s > threshold ($2k+)
- Distinguishes real money from noise
- Only available with LIVE API key

### 3. Weight of Money (WOM)
- Back/lay ratio > 1.2 = bullish (more money wanting to back)
- Sustained WOM > 1.2 for 30+ seconds = confirmed support
- Combined with LTP shortening = strong signal

### 4. Opening Price Comparison
- How far has the price moved from opening?
- Net shortening > 20% from open = significant steamer

---

## Tech Stack

- **betfairlightweight** — Python library for streaming + REST API
- **MarketSignals** — copy from betfair-signal-monitor/core/signals.py (proven, tested)
- **BetPlacer** — existing in punt-legacy-ai/betfair/ (safety limits, logging)
- **Polling API** — for runner names/metadata (streaming doesn't include these)

### Why NOT flumine?
- Overkill for steam detection
- Adds unnecessary abstraction layer
- Start simple with betfairlightweight streaming directly
- Consider flumine later for complex multi-market strategies

---

## Existing Code to Reuse

### From betfair-signal-monitor/:
- `core/signals.py` — MarketSignals class (LTP tracking, WOM, volume, consistency %)
- `core/api.py` — BetfairAPI with SSL auth, market discovery, retry logic
- `certs/` — SSL certificates (already referenced by punt-legacy-ai)

### From punt-legacy-ai/betfair/:
- `client.py` — Login, find markets, get odds, place bets
- `bet_placer.py` — Safety limits, dry_run mode, CSV logging

---

## Build Phases

### Phase 1: Foundation ✅ DONE
- API client with SSL auth
- Find markets by track/date
- Place back bets
- Safety limits (max stake, daily loss)
- Bet logging to CSV

### Phase 2: Signal Integration (NEXT)
- Install betfairlightweight (`pip install betfairlightweight`)
- Copy MarketSignals class from signal monitor
- Build streaming connection for pre-race markets
- Detect steam moves (LTP shortening + volume)
- Store signals to CSV for analysis

### Phase 3: AI + Market Integration
- Connect to AI predictor picks (fetch from Prisma or predict endpoint)
- Only monitor horses the AI picked as contenders
- Decision logic: AI pick + steam confirmation = BET
- AI pick + drift/no volume = SKIP
- Automatic bet placement with confidence-based staking

### Phase 4: Frontend Integration
- Show "money backing" indicator on AI predictor picks
- Highlight confirmed picks for subscribers
- Premium feature: live money flow alerts

### Phase 5: Optimization
- Backtest signal thresholds against results
- Tune: how many ticks = steam? What volume threshold?
- Track P&L by signal type
- Add more signals (price depth, in-play momentum)

---

## GitHub Repos to Reference

- [betfairlightweight](https://github.com/betcode-org/betfair) — Python Betfair API wrapper with streaming
- [flumine](https://github.com/betcode-org/flumine) — Event-based trading framework (for later)
- [dickreuter/betfair-horse-racing](https://github.com/dickreuter/betfair-horse-racing) — Automated horse racing system
- [michaelvrxoj/betfair-python-trading-bot](https://github.com/michaelvrxoj/betfair-python-trading-bot-automation) — Pre-race bot
- [Betfair Data Scientists tutorials](https://betfair-datascientists.github.io/tutorials/How_to_Automate_1/)

---

## Key Decisions Made

1. **Build in punt-legacy-ai/betfair/** not separate project — keeps AI + betting together
2. **Use betfairlightweight not flumine** — simpler for steam detection
3. **Streaming API for prices, polling for metadata** — hybrid approach
4. **Start with monitoring only** (dry_run) — prove signals predict winners before real money
5. **Integrate with AI picks** — only watch contenders, not entire field

---

## Research Sources

- [How to spot a steamer - Betfair](https://betting.betfair.com/betfair-announcements/betting-apps/betfair-trading-how-to-spot-a-steamer-231014-696.html)
- [Horse Racing Trading Strategies - Traderline](https://traderline.com/education/betfair-horse-racing-trading-strategies)
- [Drifters & Steamers - Caan Berry](https://caanberry.com/drifters-steamers-market-movers/)
- [Betfair Streaming API docs](https://support.developer.betfair.com/hc/en-us/articles/6540502258077)
- [Horse Racing Market Movers](https://horseracingbettingodds.com/articles/horse-racing-market-movers-steam/)
