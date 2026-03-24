# PREDICTIONS.md

Manual race analysis methodology for Claude Code.

---

## How to Analyze Races

When asked to analyze a race, apply common sense using these principles:

### Before You Start

**Slow down. Look for high ratings, but check they're relevant to THIS race before flagging.**

For each rating, ask: "Is this form relevant to the race we're analyzing?"
- Is the distance similar? (within ~20% of today's distance)
- Is it recent enough?
- Is the condition relevant? (adjacent conditions or 2 count)

**Distance relevance gradient:**
- Within 10% = most relevant
- Within 20% = still relevant
- Within 30% = less relevant
- Beyond 30% = limited relevance

High rating + relevant = flag it. High rating + not relevant = ignore it.

---

## Core Principle

**Best ratings at relevant runs.** Think logically - don't overweight any single factor too aggressively.

---

## What to Look For

**First:** Scan for any standout ratings clearly above the field at relevant distance/condition. These should always be considered.

**Then:** Look at who is consistently rating at the top of the field.

Both matter - don't ignore a standout rating, but also value consistency.

### Key Factors

1. **Adj ratings** - The primary metric
2. **Recency** - Recent ratings matter more than old runs
3. **Distance match** - Form at similar distance to today's race
4. **Condition match** - Form at similar/adjacent conditions (see scale below)

---

## Selection Rules

- **Top pick:** Horse whose recent Adj ratings at similar distance/condition are clearly higher than the rest of the field
- **Value angle:** Consider if they have a relevant run or runs clearly above the field - especially with a weight drop
- **First/second up:** Check their historical 1st-up or 2nd-up ratings if applicable

---

## Secondary Factors (Note but don't overweight)

- **Weight changes** - Scale: 1-2kg minor | 3-4kg decent | 5-6kg solid | 7+kg big change
- **Barrier draws** - Inside vs outside at certain tracks/distances
- **Class rise/drop** - Stepping up significantly or down
- **Gear changes** - Blinkers/winkers first time
- **Days since last run** - Backing up quickly vs fresh

---

## Skip Races When

- Limited form (too many first starters/unknowns)
- Multiple similar contenders (no clear edge)
- Clear standout but odds don't justify the bet

---

## Condition Proximity Scale

Track conditions in order (adjacent conditions are highly relevant):

```
G3 → G4 → S5 → S6 → S7 → H8 → H9 → H10
Good ────────────────> Soft ──────────> Heavy
```

**Relevance gradient:**
- Exact condition match = most relevant
- 1 step away = still relevant
- 2 steps away = slightly less relevant
- 3+ steps = diminishing relevance

| Today's Condition | Most Relevant Form | Also Relevant (1 step) |
|-------------------|-------------------|------------------------|
| H8 | H8 | S7, H9 |
| S7 | S7 | S6, H8 |
| S5 | S5 | G4, S6 |
| G4 | G4 | G3, S5 |

---

## Understanding the Ratings

| Column | What It Means |
|--------|---------------|
| **Rating** | Normalized by distance + condition. 100 = expected speed. |
| **Adj** | Further normalized by track quality. Makes ratings comparable across venues. |

- **Ignore finishing position** - a higher rating is better regardless of where they finished
- **Ignore margin** - already baked into the rating
- **Eased runs (⚠️eased)** - horse wasn't fully pushed, actual ability likely higher than rating shown

---

## Quick Export Command

```bash
source venv/bin/activate && source .env && python3 tools/export_for_claude_code.py "Track" R# DD-MMM-YYYY --no-instructions
```

Examples:
```bash
# Single race
python3 tools/export_for_claude_code.py "Rosehill" 2 24-Mar-2026 --no-instructions

# All races at meeting
python3 tools/export_for_claude_code.py "Rosehill" all 24-Mar-2026 --no-instructions
```
