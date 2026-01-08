# PuntingForm API Documentation

Complete documentation of all available PuntingForm API endpoints and their data fields.

**Base URL:** `https://api.puntingform.com.au/v2/`
**Auth:** All endpoints require `apiKey` query parameter
**Docs:** https://docs.puntingform.com.au/reference/meetingslist

---

## Endpoints Overview

| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/form/meetingslist` | GET | Get all meetings for a date | ✅ Use |
| `/form/fields` | GET | Runner details, career stats, records | ✅ Use |
| `/form/meeting` | GET | Same as fields (duplicate) | ⚠️ Skip |
| `/form/form` | GET | **CRITICAL** Past runs with times/margins | ✅ Use |
| `/form/results` | GET | Race results after finish | ✅ Use |
| `/form/strikerate` | GET | Jockey/Trainer career stats | ✅ Use |
| `/form/comment` | GET | Best Bets/Race.Net comments | ❌ Needs subscription |
| `/Updates/Conditions` | GET | Track conditions | ✅ Use |
| `/Updates/Scratchings` | GET | Scratched horses | ✅ Use |
| `/User/Worksheets` | GET | Market prices and edge | ⚠️ Prices only |
| `/User/Speedmaps` | GET | Pace/settling predictions | ✅ Use |
| `/Ratings/MeetingRatings` | GET | PFAI rankings | ⚠️ Document only |

---

## 1. Meetings List

**Endpoint:** `GET /v2/form/meetingslist`

**Purpose:** Get all race meetings for a specific date.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingDate` | string | ✅ | Date in format `dd-MMM-yyyy` (e.g., "08-Jan-2026") |
| `apiKey` | string | ✅ | Your API key |

### Response

Returns array of meetings:

```json
{
  "meetingId": 236584,
  "meetingDate": "2026-01-08T00:00:00",
  "track": {
    "name": "Rockhampton",
    "trackId": "238",
    "location": "C",
    "state": "QLD",
    "country": "AUS",
    "abbrev": "ROCK",
    "surface": "Turf"
  }
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `meetingId` | Unique ID - use for all other API calls |
| `track.name` | Track name |
| `track.state` | State (NSW, VIC, QLD, etc.) |
| `track.surface` | Turf or Synthetic |

---

## 2. Fields (Runner Details)

**Endpoint:** `GET /v2/form/fields`

**Purpose:** Get detailed runner information for a meeting.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingId` | int | ✅ | Meeting ID from meetingslist |
| `raceNumber` | int | ❌ | Race number (0 = all races) |
| `apiKey` | string | ✅ | Your API key |

### Response Structure

```
payLoad
├── track: {name, trackId, state, surface}
├── meetingId
├── meetingDate
├── railPosition
├── expectedCondition
├── isBarrierTrial
├── hasSectionals
├── races[]
│   ├── raceId
│   ├── number
│   ├── name
│   ├── distance
│   ├── raceClass
│   ├── prizeMoney
│   ├── startTime
│   ├── weightType
│   ├── ageRestrictions
│   ├── sexRestrictions
│   └── runners[]
```

### Runner Fields (64 total)

**Basic Info:**
| Field | Type | Description |
|-------|------|-------------|
| `runnerId` | int | Unique runner ID |
| `name` | string | Horse name |
| `tabNo` | int | Tab/saddle cloth number |
| `barrier` | int | Barrier position |
| `age` | int | Horse age |
| `sex` | string | Gelding, Mare, Colt, etc. |
| `colour` | string | Horse colour |
| `weight` | float | Carried weight |
| `weightAllocated` | float | Allocated weight |
| `weightAdjustment` | float | Weight adjustment (apprentice claim) |

**Career Stats:**
| Field | Type | Description |
|-------|------|-------------|
| `careerStarts` | int | Total career starts |
| `careerWins` | int | Career wins |
| `careerSeconds` | int | Career 2nds |
| `careerThirds` | int | Career 3rds |
| `winPct` | float | Win percentage |
| `placePct` | float | Place percentage |
| `prizeMoney` | float | Career prizemoney |
| `last10` | string | Last 10 starts (e.g., "3214x56789") |

**Connections:**
| Field | Type | Description |
|-------|------|-------------|
| `trainer` | object | `{fullName, trainerId, location}` |
| `jockey` | object | `{fullName, jockeyId, ridingWeight}` |
| `jockeyClaim` | float | Apprentice claim (kg) |
| `owners` | string | Owner names |

**Breeding:**
| Field | Type | Description |
|-------|------|-------------|
| `sire` | string | Father |
| `dam` | string | Mother |
| `sireofDam` | string | Maternal grandfather |
| `foalDate` | string | Birth date |
| `country` | string | Country of origin |

**Track/Distance Records:**
| Field | Type | Description |
|-------|------|-------------|
| `trackRecord` | object | `{starts, firsts, seconds, thirds}` at this track |
| `distanceRecord` | object | `{starts, firsts, seconds, thirds}` at this distance |
| `trackDistRecord` | object | Track + distance combined record |

**Condition Records:**
| Field | Type | Description |
|-------|------|-------------|
| `firmRecord` | object | Record on Firm tracks |
| `goodRecord` | object | Record on Good tracks |
| `softRecord` | object | Record on Soft tracks |
| `heavyRecord` | object | Record on Heavy tracks |
| `syntheticRecord` | object | Record on Synthetic tracks |

**Spell Records:**
| Field | Type | Description |
|-------|------|-------------|
| `firstUpRecord` | object | First-up record (after spell) |
| `secondUpRecord` | object | Second-up record |

**Group Records:**
| Field | Type | Description |
|-------|------|-------------|
| `group1Record` | object | Group 1 race record |
| `group2Record` | object | Group 2 race record |
| `group3Record` | object | Group 3 race record |

**A/E (Actual vs Expected) Stats:**
| Field | Type | Description |
|-------|------|-------------|
| `jockeyA2E_Career` | object | Jockey career A/E |
| `jockeyA2E_Last100` | object | Jockey last 100 rides A/E |
| `trainerA2E_Career` | object | Trainer career A/E |
| `trainerA2E_Last100` | object | Trainer last 100 runners A/E |
| `trainerJockeyA2E_Career` | object | Combo career A/E |
| `trainerJockeyA2E_Last100` | object | Combo last 100 A/E |

**A/E Object Structure:**
```json
{
  "jockey": "Dylan Gibbons",
  "trainer": "Matthew Smith",
  "a2E": 1.15,
  "wins": 12,
  "starts": 100,
  "expectedWins": 10.43
}
```
- `a2E > 1.0` = Outperforming (hot)
- `a2E < 1.0` = Underperforming (cold)

**Other:**
| Field | Type | Description |
|-------|------|-------------|
| `gearChanges` | string | "Blinkers first time", etc. |
| `emergencyIndicator` | bool | Is emergency runner |
| `handicap` | float | Handicap rating |
| `prepRuns` | int | Runs this prep |
| `formId` | int | Form record ID |

---

## 3. Form (Past Runs) - CRITICAL

**Endpoint:** `GET /v2/form/form`

**Purpose:** Get past race history for each runner. **This is the most important endpoint for speed calculations.**

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingId` | int | ✅ | Meeting ID |
| `raceNumber` | int | ❌ | Race number (0 = all races) |
| `runs` | int | ❌ | Number of past runs (max 10) |
| `index` | int | ❌ | Pagination index |
| `count` | int | ❌ | Total count to return |
| `apiKey` | string | ✅ | Your API key |

### Response Structure

Returns array of runners, each with `forms` array containing past runs.

### Past Run Fields (CRITICAL FOR SPEED)

**Timing Data:**
| Field | Type | Description | USE FOR SPEED |
|-------|------|-------------|---------------|
| `officialRaceTime` | string | Winner's time `"00:01:11.3400000"` | ✅ YES |
| `pfRaceTime` | string | PF adjusted time | Optional |
| `distance` | int | Race distance in meters | ✅ YES |
| `position` | int | Finishing position | ✅ YES |
| `margin` | float | Lengths behind winner | ✅ YES |
| `starters` | int | Field size | ✅ YES |

**Track/Condition:**
| Field | Type | Description |
|-------|------|-------------|
| `track` | object | `{name, trackId, state}` |
| `trackCondition` | string | Condition label (e.g., "S5") |
| `trackConditionNumber` | int | Condition number (1-10) |
| `rail` | int | Rail position |

**Weight:**
| Field | Type | Description |
|-------|------|-------------|
| `weight` | float | Carried weight |
| `weightAllocated` | float | Allocated weight |
| `weightAdjustment` | float | Claim adjustment |
| `weightType` | string | Handicap, Set Weight, etc. |

**Race Details:**
| Field | Type | Description |
|-------|------|-------------|
| `meetingDate` | string | Date of race |
| `raceName` | string | Race name |
| `raceClass` | string | Class of race |
| `prizeMoney` | int | Prize money |
| `ageRestrictions` | string | Age restrictions |
| `sexRestrictions` | string | Sex restrictions |

**Prices:**
| Field | Type | Description |
|-------|------|-------------|
| `priceSP` | float | Starting price |
| `priceTAB` | float | TAB price |
| `priceBF` | float | Betfair price |
| `flucs` | string | Price fluctuations |

**Running Style:**
| Field | Type | Description |
|-------|------|-------------|
| `inRun` | string | Running positions `"settling_down,6;m800,6;m400,6;finish,2;"` |

**Other:**
| Field | Type | Description |
|-------|------|-------------|
| `barrier` | int | Barrier drawn |
| `jockey` | object | Jockey details |
| `jockeyClaim` | float | Apprentice claim |
| `stewardsReport` | string | Any stewards comments |
| `gearChanges` | string | Gear changes for that run |
| `isBarrierTrial` | bool | Was it a barrier trial |
| `hasSectionalData` | bool | Has sectional times |
| `top4Finishers` | array | Top 4 finishers in that race |
| `kri` | int | KRI rating |
| `prepRuns` | int | Runs this prep at that point |

---

## 4. Results

**Endpoint:** `GET /v2/form/results`

**Purpose:** Get race results after races have finished.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingId` | int | ✅ | Meeting ID |
| `raceNumber` | int | ❌ | Race number (0 = all races) |
| `apiKey` | string | ✅ | Your API key |

### Response - Race Level

| Field | Type | Description |
|-------|------|-------------|
| `raceId` | int | Race ID |
| `raceNumber` | int | Race number |
| `distance` | int | Distance |
| `raceClass` | string | Class |
| `officialRaceTime` | string | Winner's time |
| `officialRaceTimeString` | string | Formatted time |
| `trackCondition` | int | Condition number |
| `trackConditionLabel` | string | Condition label |
| `weightType` | string | Weight type |
| `windDirection` | int | Wind direction |
| `windSpeed` | int | Wind speed |

### Response - Runner Level

| Field | Type | Description |
|-------|------|-------------|
| `position` | int | Finishing position (0 = scratched) |
| `margin` | float | Lengths behind winner |
| `runner` | string | Horse name |
| `runnerId` | int | Runner ID |
| `tabNo` | int | Tab number |
| `barrier` | int | Barrier |
| `weight` | float | Weight carried |
| `price` | float | Starting price |
| `flucs` | string | Price fluctuations |
| `inRun` | string | Running positions |
| `jockey` | string | Jockey name |
| `trainer` | string | Trainer name |
| `gearChanges` | string | Gear changes |
| `stewardsReports` | string | Stewards comments |

---

## 5. Track Conditions

**Endpoint:** `GET /v2/Updates/Conditions`

**Purpose:** Get current track conditions for all tracks.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `jurisdiction` | int | ❌ | 0 = all, 1/2 = specific regions |
| `apiKey` | string | ✅ | Your API key |

### Response

| Field | Type | Description |
|-------|------|-------------|
| `track` | string | Track name |
| `condition` | string | Condition label (Good, Soft, Heavy) |
| `conditionNumber` | int | Condition number (1-10) |
| `rail` | string | Rail position |
| `weather` | string | Weather conditions |

### Condition Numbers

| Number | Label |
|--------|-------|
| 1-2 | Firm |
| 3-4 | Good |
| 5-6 | Soft |
| 7-8 | Heavy |
| 9-10 | Very Heavy |

---

## 6. Scratchings

**Endpoint:** `GET /v2/Updates/Scratchings`

**Purpose:** Get all scratched horses.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `jurisdiction` | int | ❌ | 0 = all |
| `apiKey` | string | ✅ | Your API key |

### Response

| Field | Type | Description |
|-------|------|-------------|
| `meetingId` | int | Meeting ID |
| `raceNo` | int | Race number |
| `runnerId` | int | Runner ID |
| `tabNo` | int | Tab number |
| `reason` | string | Scratching reason |

---

## 7. Worksheets (Market Prices)

**Endpoint:** `GET /v2/User/Worksheets`

**Purpose:** Get current market prices and PFAI edge data.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingId` | int | ✅ | Meeting ID |
| `raceNo` | int | ❌ | Race number (0 = all races) |
| `apiKey` | string | ✅ | Your API key |

### Response

| Field | Type | Description | Use? |
|-------|------|-------------|------|
| `tabNo` | int | Tab number | ✅ |
| `runner` | string | Horse name | ✅ |
| `runnerId` | int | Runner ID | ✅ |
| `barrier` | int | Barrier | ✅ |
| `weight` | float | Weight | ✅ |
| `bestPrice_SinceOpen` | float | Opening/best price | ✅ |
| `bestPrice_Current` | float | Current price | ✅ |
| `edge` | float | PFAI calculated edge | ⚠️ (uses PFAI) |
| `pfaiScore` | int | PFAI confidence score | ❌ Skip |
| `pfaiPrice` | float | PFAI model price | ❌ Skip |
| `mapA2E` | float | Map A/E | ✅ |

**Note:** We use this for market prices only, ignoring PFAI-specific fields.

---

## 8. Speedmaps

**Endpoint:** `GET /v2/User/Speedmaps`

**Purpose:** Get predicted pace and settling positions.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingId` | int | ✅ | Meeting ID |
| `raceNo` | int | ❌ | Race number (0 = all races) |
| `apiKey` | string | ✅ | Your API key |

### Response

| Field | Type | Description |
|-------|------|-------------|
| `runnerId` | int | Runner ID |
| `runnerName` | string | Horse name |
| `tabNo` | int | Tab number |
| `barrier` | int | Barrier |
| `speed` | int | Early speed rank (1 = fastest) |
| `settle` | int | Predicted settling position |
| `ratedRunStyle` | int | Run style rating |
| `ratedSettle` | int | Settle rating |
| `mapA2E` | float | Map A/E |
| `jockeyA2E` | float | Jockey A/E |

### Use Cases

- Count runners with `speed` = 1-2 to determine pace scenario
- Hot pace = 3+ leaders → favors closers
- Soft pace = 0-1 leaders → favors leaders

---

## 9. Strikerate (Jockey/Trainer Stats)

**Endpoint:** `GET /v2/form/strikerate`

**Purpose:** Get career and recent stats for jockeys/trainers.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `entityType` | int | ✅ | 1 = Trainers, 2 = Jockeys |
| `jurisdiction` | int | ❌ | 0 = all |
| `apiKey` | string | ✅ | Your API key |

### Response

| Field | Type | Description |
|-------|------|-------------|
| `entityId` | int | Jockey/Trainer ID |
| `entityName` | string | Name |
| `careerWins` | int | Career wins |
| `careerStarts` | int | Career starts |
| `careerSeconds` | int | Career 2nds |
| `careerThirds` | int | Career 3rds |
| `careerExpectedWins` | float | Expected wins based on odds |
| `careerPL` | float | Career profit/loss |
| `careerTurnover` | float | Career turnover |
| `last100Wins` | int | Wins in last 100 |
| `last100Starts` | int | Starts in last 100 |
| `last100Seconds` | int | 2nds in last 100 |
| `last100Thirds` | int | 3rds in last 100 |
| `last100ExpectedWins` | float | Expected wins last 100 |
| `last100PL` | float | P/L last 100 |
| `last100Turnover` | float | Turnover last 100 |

### Calculating A/E

```python
a_e = last100Wins / last100ExpectedWins
# > 1.0 = outperforming (hot)
# < 1.0 = underperforming (cold)
```

---

## 10. Meeting Ratings (PFAI) - DOCUMENT ONLY

**Endpoint:** `GET /v2/Ratings/MeetingRatings`

**Purpose:** Get PFAI rankings and related data. **We document this but don't use PFAI rankings for predictions.**

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `meetingId` | int | ✅ | Meeting ID |
| `apiKey` | string | ✅ | Your API key |

### Response

| Field | Type | Description | Use? |
|-------|------|-------------|------|
| `runnerId` | int | Runner ID | ✅ |
| `runnerName` | string | Horse name | ✅ |
| `tabNo` | int | Tab number | ✅ |
| `barrier` | int | Barrier | ✅ |
| `isReliable` | bool | Is data reliable | ✅ Critical! |
| `pfaiScore` | int | PFAI confidence | ❌ |
| `pfaiPrice` | float | PFAI model price | ❌ |
| `pfaiRank` | int | PFAI rank (1=best) | ❌ |
| `runStyle` | string | Running style code | ✅ |
| `predictedSettlePosition` | int | Predicted settling | ✅ |
| `timeRank` | int | Time-based rank | ⚠️ |
| `timePrice` | float | Time-based price | ⚠️ |
| `earlyTimeRank` | int | Early speed rank | ⚠️ |
| `last600TimeRank` | int | Last 600m rank | ⚠️ |
| `last400TimeRank` | int | Last 400m rank | ⚠️ |
| `last200TimeRank` | int | Last 200m rank | ⚠️ |
| `weightClassRank` | int | Weight/class rank | ⚠️ |
| `classChange` | float | Class change | ⚠️ |

**Note:** The `isReliable` field is critical - if `false`, the price data may be incorrect.

---

## API Tips & Tricks

### Get All Races in Meeting
Use `raceNumber=0` or `raceNo=0` to get all races at once.

### Pagination for Form
Use `index` and `count` params for large form requests:
```python
params = {'meetingId': id, 'index': 0, 'count': 200, 'apiKey': key}
```

### Jurisdiction Codes
| Code | Region |
|------|--------|
| 0 | All Australia |
| 1 | NSW/ACT |
| 2 | VIC/TAS |

### Rate Limiting
No rate limiting observed - parallel calls are safe.

---

## Example: Get All Data for a Meeting

```python
from concurrent.futures import ThreadPoolExecutor

meeting_id = 236584

with ThreadPoolExecutor(max_workers=5) as ex:
    f_fields = ex.submit(get_fields, meeting_id, 0)
    f_form = ex.submit(get_form, meeting_id, 0, 10)
    f_speedmaps = ex.submit(get_speedmaps, meeting_id, 0)
    f_conditions = ex.submit(get_conditions, 0)
    f_scratchings = ex.submit(get_scratchings, 0)

    fields = f_fields.result()
    form = f_form.result()
    speedmaps = f_speedmaps.result()
    conditions = f_conditions.result()
    scratchings = f_scratchings.result()
```
