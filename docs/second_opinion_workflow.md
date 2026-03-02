# Second Opinion Workflow

A process for getting a human-like second opinion on AI predictions, using track speed ratings that the AI predictor doesn't have access to.

**Key:** The live AI predictor costs API tokens. The second opinion is FREE - it's Claude (in conversation) analyzing the data rationally.

## What the AI Misses

The AI predictor sees raw speed ratings and IS told to focus on similar distance/conditions. However, it **doesn't know which tracks are fast or slow**:

- **Fast tracks** (Randwick, Rosehill, Caulfield) inflate ratings - a 101 there might only be a 99 at an average track
- **Slow tracks** (Stony Creek, Moe, Cranbourne) deflate ratings - a 97 there might really be a 100

The second opinion catches horses being **overrated** (fast track form) or **underrated** (slow track form) by adjusting ratings based on track speed.

## Quick Start

```bash
# 1. Run the second opinion tool:
python3 tools/second_opinion.py "Rosehill" 4 "25-Feb-2026"

# 2. Get AI picks from /predict or tipsheet

# 3. Paste both to Claude for second opinion
```

## What the Tool Shows

1. **Instructions** - What to focus on (distance + condition matching)
2. **Track ratings reference** - Fast/slow tracks with sample sizes
3. **Form table per runner** with:
   - Raw rating and track-adjusted rating
   - Relevance indicator: ✅ YES (distance + condition match), ~dist only, ❌ no
4. **Best relevant adjusted rating** per horse (only from matching form)
5. **Final ranking** - Horses ranked by best relevant adjusted rating

## Key Insight: Relevant Form Only

The tool only considers form that matches:
- **Distance**: Within ±20% of today's race
- **Condition**: Within ±2 levels on the 1-10 scale

**Condition Scale**: F1 → F2 → G3 → G4 → S5 → S6 → S7 → H8 → H9 → H10

**Example**: For an S6 race, relevant conditions are G4, S5, S6, S7, H8 (±2 levels).
A horse with a 103 rating at 1200m on G3 is **irrelevant** if today is 1400m on S6 (3 levels apart)!

## Track Rating Scale

| Rating | Label | Meaning |
|--------|-------|---------|
| ≥1.015 | ⚠️FAST | Ratings inflated by ~1.5+ points |
| 1.005-1.015 | fast | Ratings slightly inflated |
| 0.990-1.005 | avg | Ratings genuine |
| 0.975-0.990 | slow | Ratings understated |
| ≤0.975 | ⚠️SLOW | Ratings understated by ~2.5+ points |
| <50 samples | ⚠️LOW DATA | Don't trust adjustment |

## Sample Size Check

Only trust track adjustments from tracks with **50+ samples**. The tool flags low-sample tracks.

## Key Sydney Tracks
- **Fast** (inflate ratings): Randwick (1.010), Hawkesbury (1.010), Rosehill (1.007), Canterbury (1.006), Warwick Farm (1.005)
- **Avg/Slow** (genuine/understated): Kembla (0.997), Gosford (0.996), Newcastle (0.988), Beaumont (0.985)

## Daily Process

1. **Run the tool** for each race:
   ```bash
   python3 tools/second_opinion.py "Track" N "DD-Mon-YYYY"
   ```
2. **Check the ranking** at the bottom - who has the best RELEVANT adjusted rating?
3. **Get AI picks** from /predict endpoint
4. **Compare**:
   - Does AI agree with ranking? Good.
   - Does AI like a horse with no relevant form? Question it!
   - Does AI miss a horse with slow track form? Value opportunity!
5. **Claude gives second opinion**:
   - ✅ BET: Best relevant rating + fair price
   - 🔸 LEAN: Good rating but short price or concerns
   - ❌ NO BET: No relevant form, or form is from fast tracks

## Example: Rosehill R4 (25-Feb-2026)

**Race**: 2400m G4

**AI picked**: Centenario $2.60 ⭐, Lunar Lover $3.70 ⭐, Liberty Park $5.50 ⭐

**Relevant conditions for G4**: F2, G3, G4, S5, S6 (±2 levels)

**Second opinion ranking** (best relevant adjusted rating):
| Rank | Horse | Odds | Best Adj | Notes |
|------|-------|------|----------|-------|
| 1 | Subarctic | $11 | 102.4 | slow track form (understated) |
| 2 | Oakfield Hawk | $14 | 102.0 | slow track + Berry A/E 1.18 |
| 3 | Lunar Lover | $3.70 | 101.7 | all fast track form |
| 4 | Liberty Park | $5.50 | 100.1 | his 103.1 was on S7 (3 levels away) |

**Problems found**:
- Centenario $2.60 - **NOT IN RANKING** - has no form at 2400m!
- Liberty Park's best rating (103.1) was on S7 - outside ±2 levels of G4

**Second opinion**:
- ✅ BET Oakfield Hawk $14 - 2nd best rating, Tommy Berry, trainer A/E 1.34
- 🔸 LEAN Subarctic $11 - best rating but poor jockey A/E

## Files

- `tools/second_opinion.py` - Main tool (shows relevant form + rankings)
- `core/normalization/track_ratings.csv` - Track speed ratings (with sample sizes)
- `docs/second_opinion_workflow.md` - This doc
