# Ladbrokes API Documentation

Public affiliate API for live racing odds.

**Base URL:** `https://api.ladbrokes.com.au/affiliates/v1`
**Auth:** Headers required (see below)
**Docs:** https://nedscode.github.io/affiliate-feeds/#racing-apis

---

## Authentication

All requests require identifying headers:

```python
headers = {
    "From": "your_email@example.com",
    "X-Partner": "YourOrganization"
}
```

Without these headers, requests may be rate-limited or blocked (429 error).

---

## Endpoints Overview

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/racing/meetings` | GET | List meetings for date range |
| `/racing/meetings/{id}` | GET | Specific meeting details |
| `/racing/events/{id}` | GET | Race details with runners and odds |

---

## 1. Racing Meetings

**Endpoint:** `GET /racing/meetings`

**Purpose:** Get all race meetings for a date.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `date_from` | string | No | Start date (YYYY-MM-DD, or: now, today, week, month) |
| `date_to` | string | No | End date |
| `category` | string | No | T=Thoroughbred, H=Harness, G=Greyhound |
| `country` | string | No | Country code (AUS, NZ, HK, etc.) |
| `limit` | int | No | Max results (default 100, max 200) |
| `offset` | int | No | Pagination offset |
| `enc` | string | No | Response format (json, xml, html) |

### Response

```json
{
  "header": {
    "title": "Meetings",
    "generated_time": 1736300400
  },
  "data": {
    "meetings": [
      {
        "meeting": "uuid-string",
        "name": "Randwick",
        "date": "2026-01-08",
        "category": "T",
        "category_name": "Thoroughbred",
        "country": "AUS",
        "state": "NSW",
        "races": [
          {
            "id": "race-uuid-string",
            "race_number": 1,
            "description": "Maiden Plate",
            "advertised_start": 1736300400,
            "status": "open",
            "distance": 1200
          }
        ]
      }
    ]
  }
}
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `meeting` | string | Meeting UUID |
| `name` | string | Track name |
| `date` | string | Meeting date |
| `category` | string | T/H/G |
| `state` | string | State (NSW, VIC, QLD, etc.) |
| `races[]` | array | List of races |
| `races[].id` | string | Race UUID - use for events endpoint |
| `races[].race_number` | int | Race number |
| `races[].status` | string | open, closed, final, abandoned |

---

## 2. Race/Event Details

**Endpoint:** `GET /racing/events/{race_id}`

**Purpose:** Get full race details including runners and odds.

### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Race UUID from meetings response |
| `enc` | string | No | Response format |

### Response

```json
{
  "data": {
    "race": {
      "event_id": "race-uuid",
      "meeting_id": "meeting-uuid",
      "meeting_name": "Randwick",
      "race_number": 1,
      "description": "Maiden Plate",
      "status": "open",
      "advertised_start": 1736300400,
      "actual_start": null,
      "distance": 1200,
      "weather": "Fine",
      "track_condition": "Good 4",
      "form_guide": "...",
      "comment": "..."
    },
    "runners": [
      {
        "name": "Fast Horse",
        "barrier": 3,
        "runner_number": 5,
        "jockey": "J. McDonald",
        "trainer_name": "C. Waller",
        "weight": 57.0,
        "age": 3,
        "sex": "Gelding",
        "colour": "Bay",
        "silk_colours": "Purple, gold stars",
        "scratched": false,
        "scratch_time": null,
        "favourite": true,
        "mover": false,
        "odds": {
          "fixed_win": 3.50,
          "fixed_place": 1.40
        },
        "flucs": "3.50, 3.80, 3.50"
      }
    ],
    "results": [],
    "dividends": []
  }
}
```

### Runner Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Horse name |
| `barrier` | int | Barrier position |
| `runner_number` | int | Saddle cloth number |
| `jockey` | string | Jockey name |
| `trainer_name` | string | Trainer name |
| `weight` | float | Weight carried |
| `age` | int | Horse age |
| `sex` | string | Gelding, Mare, Colt, etc. |
| `scratched` | bool | Is scratched |
| `scratch_time` | int | Unix timestamp if scratched |
| `favourite` | bool | Is market favourite |
| `mover` | bool | Has price moved significantly |

### Odds Fields

| Field | Type | Description |
|-------|------|-------------|
| `odds.fixed_win` | float | Fixed win price |
| `odds.fixed_place` | float | Fixed place price |
| `flucs` | string | Price fluctuation history |

### Results Fields (after race)

| Field | Type | Description |
|-------|------|-------------|
| `position` | int | Finishing position |
| `name` | string | Horse name |
| `margin_length` | float | Margin behind winner |
| `runner_number` | int | Saddle cloth number |

---

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Invalid race/meeting ID |
| 429 | Rate Limited - Add identification headers |
| 500 | Server Error |
| 503 | Service Unavailable |

---

## Example Usage

```python
import requests

BASE_URL = "https://api.ladbrokes.com.au/affiliates/v1"
HEADERS = {
    "From": "your_email@example.com",
    "X-Partner": "YourOrg"
}

# Get today's thoroughbred meetings
response = requests.get(
    f"{BASE_URL}/racing/meetings",
    params={"date_from": "today", "category": "T", "country": "AUS"},
    headers=HEADERS
)
meetings = response.json()["data"]["meetings"]

# Get odds for a specific race
race_id = meetings[0]["races"][0]["id"]
response = requests.get(
    f"{BASE_URL}/racing/events/{race_id}",
    headers=HEADERS
)
race_data = response.json()["data"]
```

---

## Comparison with PuntingForm

| Feature | Ladbrokes | PuntingForm |
|---------|-----------|-------------|
| Auth | Headers (email + partner) | API key |
| Odds accuracy | **Accurate** | Unreliable (see below) |
| Form data | Limited | **Comprehensive** |
| Speed maps | No | Yes |
| Jockey/Trainer stats | Limited | **Full A/E data** |
| Scratchings | In runner data | Separate endpoint |

**Use Ladbrokes for:** Live odds
**Use PuntingForm for:** Everything else (form, stats, conditions)

---

## Known Issues with PuntingForm Odds

See [PuntingForm Odds Issue](./puntingform_odds_issue.md) for full details.

**Summary:** PuntingForm's `bestPrice_Current` field sometimes returns incorrect odds - significantly higher than actual market prices. This affects our predictions because we calculate edge based on these prices.

**Solution:** Use Ladbrokes odds instead of PuntingForm's `bestPrice_Current` for live market prices.
