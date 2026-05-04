# Betfair Integration

Automated bet placement and market monitoring for punt-legacy-ai predictions.

## Status: Phase 1 - Foundation

---

## Plan

### Phase 1: Foundation (Current)
- [x] Betfair API client (reused from betfair-signal-monitor)
- [x] Login with SSL certificates
- [x] Find race markets by track/date
- [x] Place a bet (back order)
- [x] Basic safety limits (max stake, daily loss limit)
- [ ] Test with $1 bets on real markets

### Phase 2: Manual Picks Integration
- [ ] Script: I analyze a race in Claude Code, output pick, place bet
- [ ] Match horse name to Betfair selection ID
- [ ] Confirm odds before placing (min odds threshold)
- [ ] Log all bets to CSV (track, race, horse, odds, stake, result)

### Phase 3: Market Monitoring
- [ ] Monitor odds movement for AI predictor picks
- [ ] Detect late money (LTP shortening, WOM signals)
- [ ] Flag picks with money confirmation on frontend
- [ ] Requires: Betfair Live API key (currently using delayed/free)

### Phase 4: Frontend Integration
- [ ] Show "money backing" indicator on AI predictor picks
- [ ] Highlight which AI picks have late money support
- [ ] Premium feature for subscribers

### Phase 5: Semi-Automated Betting
- [ ] Claude analyzes races, outputs picks with confidence
- [ ] System monitors market, places bets when money confirms
- [ ] Human approval step (optional)
- [ ] Full bet tracking and P&L reporting

---

## Setup

### Prerequisites
- Betfair account with API access
- SSL certificates (betfair.crt + betfair.key) in certs/ or betfair-signal-monitor/certs/
- Environment variables in .env:
  ```
  BETFAIR_APP_KEY=xxx
  BETFAIR_USERNAME=xxx
  BETFAIR_PASSWORD=xxx
  ```

### Usage
```python
from betfair.client import BetfairClient

client = BetfairClient()
client.login()

# Find a race
markets = client.find_race("Randwick", "04-May-2026")

# Place a bet
result = client.place_bet(
    market_id="1.234567890",
    selection_id=12345678,
    odds=3.50,
    stake=10.00
)
```

---

## Safety Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Max stake per bet | $20 | Hard cap per individual bet |
| Max daily loss | $100 | Stop placing bets after this loss |
| Min odds | $1.50 | Don't bet below this price |
| Max odds | $50.00 | Don't bet above this price |

These are hardcoded as defaults and can be overridden per session.

---

## Files

```
betfair/
├── README.md          # This file
├── client.py          # Betfair API client (login, markets, bet placement)
├── bet_placer.py      # High-level bet placement with safety checks
└── bet_log.csv        # Auto-generated bet history
```
