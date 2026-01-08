# PuntingForm Odds Reliability Issue

**Status:** NOT FIXED (as of Jan 8, 2026)

This document explains why we use Ladbrokes odds instead of PuntingForm's `bestPrice_Current` field.

---

## The Problem

PuntingForm's `bestPrice_Current` field sometimes returns incorrect odds - significantly higher than actual market prices.

### Examples of Bad Data

| Date | Track | Horse | PF Price | Actual SP | Ratio |
|------|-------|-------|----------|-----------|-------|
| Jan 7, 2026 | Eagle Farm R1 | Yoyo Yeezy | $7.90 | $1.30 | 6.1x |
| Dec 24, 2025 | Mornington R1 | Shalhavmusik | $21.80 | $2.80 | 7.8x |
| Dec 14, 2025 | Bordertown R1 | French Love | $35.90 | $4.60 | 7.8x |
| Dec 2025 | Various | Shadashi | $9.40 | $1.90 | 4.9x |
| Dec 2025 | Various | Maxandus | $6.40 | $1.40 | 4.6x |
| Dec 2025 | Various | Zouking | $4.20 | $1.26 | 3.3x |

### Impact

When we calculate edge using inflated prices, the horse appears to have huge value:

```
Edge = (100 / PFAI_price) - (100 / market_price)

Example with bad data:
Edge = (100 / 2.00) - (100 / 7.90) = 50 - 12.7 = +37.3%  <-- FAKE VALUE

Actual edge:
Edge = (100 / 2.00) - (100 / 1.30) = 50 - 76.9 = -26.9%  <-- NO VALUE
```

This makes unprofitable bets look very profitable.

---

## Root Cause

Unknown. PuntingForm aggregates odds from multiple providers:
- TAB
- Sportsbet
- Ladbrokes
- Tabtouch
- Dabble
- Bluebet
- Swiftbet
- Colossalbet
- Boombet
- Topsport
- Elitebet

Theory: The system may be mixing Win and Place odds, or holding stale prices from certain providers.

---

## Timeline

| Date | Event |
|------|-------|
| Dec 24, 2025 | Issue identified (Shalhavmusik) |
| Dec 26, 2025 | Reported to David Truong (PuntingForm) |
| Dec 27, 2025 | David acknowledged bug, said fix in next sprint |
| Dec 27, 2025 | UAT API suggested as workaround - had opposite issue |
| Jan 6, 2026 | David said fix deployed to production |
| Jan 7, 2026 | **Issue still occurring** (Yoyo Yeezy 6.1x) |
| Jan 7, 2026 | Re-reported to David |

**Contact:** David Truong <david.truong@puntingform.com.au>

---

## Our Solution

Use **Ladbrokes odds** instead of PuntingForm's `bestPrice_Current`.

### Implementation

```python
# Try Ladbrokes first (accurate)
lb_odds = get_ladbrokes_odds(track, race_number)
horse_odds = lb_odds.get(normalize_horse_name(horse_name))

if horse_odds and horse_odds.get('fixed_win'):
    price = horse_odds['fixed_win']
    source = 'Ladbrokes'
else:
    # Fall back to PuntingForm (may be inaccurate)
    price = pf_data.get('bestPrice_Current')
    source = 'PuntingForm'
```

### What We Still Use From PuntingForm

Everything except `bestPrice_Current`:

| Data | Source |
|------|--------|
| Live odds | **Ladbrokes** |
| Opening odds | PuntingForm `bestPrice_SinceOpen` |
| Form history | PuntingForm `/form/form` |
| Career stats | PuntingForm `/form/fields` |
| Jockey/Trainer A/E | PuntingForm `/form/fields` |
| Speed maps | PuntingForm `/User/Speedmaps` |
| Scratchings | PuntingForm `/Updates/Scratchings` |
| Track conditions | PuntingForm `/Updates/Conditions` |

---

## Identifying Bad Data

Signs that PuntingForm price may be wrong:

1. **Price > 2x SP** - Early price much higher than starting price
2. **Price vs Ladbrokes mismatch** - Significant difference between sources
3. **is_reliable = False** - API flagging unreliable data
4. **Frozen prices** - Price hasn't changed across multiple snapshots

### Validation Check

```python
def validate_price(pf_price, ladbrokes_price, threshold=2.0):
    """Return True if prices are within acceptable range."""
    if not pf_price or not ladbrokes_price:
        return False

    ratio = max(pf_price, ladbrokes_price) / min(pf_price, ladbrokes_price)
    return ratio <= threshold
```

---

## Horse Name Matching Between APIs

**Critical issue:** Horse names must match exactly when linking PuntingForm and Ladbrokes data.

### Known Differences

| Issue | PuntingForm | Ladbrokes | Solution |
|-------|-------------|-----------|----------|
| Case | "Fast Horse" | "FAST HORSE" | `.lower()` |
| Whitespace | "Fast  Horse" | "Fast Horse" | `.strip()`, normalize spaces |
| Apostrophes | "O'Brien" | "O'Brien" or "OBrien" | Remove or normalize |
| Special chars | "D'artagnan" | "Dartagnan" | Remove non-alphanumeric |

### Normalization Function

```python
import re

def normalize_horse_name(name: str) -> str:
    """
    Normalize horse name for matching across APIs.

    - Lowercase
    - Remove apostrophes and special characters
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower()

    # Remove apostrophes and special characters (keep letters, numbers, spaces)
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Collapse multiple spaces to single space
    name = re.sub(r"\s+", " ", name)

    # Strip whitespace
    return name.strip()
```

### Example

```python
>>> normalize_horse_name("O'Brien's Pride")
"obriens pride"

>>> normalize_horse_name("FAST  HORSE")
"fast horse"

>>> normalize_horse_name("D'Artagnan (NZ)")
"dartagnan nz"
```

---

## Blocklist

When bad data is identified, add to blocklist to prevent re-adding:

**File:** `data/blocklist.json`

```json
{
  "blocked": [
    {
      "date": "07-Jan-2026",
      "track": "Eagle Farm",
      "race_number": 1,
      "horse_name": "Yoyo Yeezy",
      "reason": "Bad PF price data (6.1x SP)"
    }
  ]
}
```

---

## Impact on Backtests

| Scenario | ROI |
|----------|-----|
| Raw backtest (with bad data) | ~+35% |
| Clean backtest (bad data removed) | ~+16% |

Both are profitable, but real-world results will be closer to the clean numbers.

**Recommendation:** Always verify tips against actual TAB/Ladbrokes odds before publishing.
