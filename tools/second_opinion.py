#!/usr/bin/env python3
"""
Second Opinion Tool

Runs the AI predictor then formats the data with track speed ratings
for manual review and second opinion analysis.

Usage:
    python tools/second_opinion.py "Warwick" 6 "23-Feb-2026"
"""

import sys
import os
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.race_data import RaceDataPipeline
# NOTE: We do NOT call the predictor here - that costs API tokens
# This tool just formats data for Claude (in conversation) to review

# Load track ratings
TRACK_RATINGS = {}
TRACK_RATINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core", "normalization", "track_ratings.csv"
)

def load_track_ratings():
    """Load track ratings from CSV with sample counts."""
    global TRACK_RATINGS
    if TRACK_RATINGS:
        return TRACK_RATINGS

    with open(TRACK_RATINGS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            venue = row['venue']
            try:
                rating = float(row['track_rating'])
                samples = int(row.get('samples', 0))
                TRACK_RATINGS[venue.lower()] = {'rating': rating, 'samples': samples}
            except (ValueError, KeyError):
                pass
    return TRACK_RATINGS


def get_track_rating(track_name: str) -> tuple[float | None, str, int]:
    """
    Get track rating for a venue.
    Returns (rating, label, samples) where label is like "FAST", "SLOW", "avg", etc.
    """
    ratings = load_track_ratings()

    # Normalize track name
    track_lower = track_name.lower().strip()

    # Try exact match first
    data = None
    if track_lower in ratings:
        data = ratings[track_lower]
    else:
        # Try partial matches
        for key, val in ratings.items():
            if track_lower in key or key in track_lower:
                data = val
                break

    if data is None:
        return None, "?", 0

    r = data['rating']
    samples = data['samples']

    # Classify - but flag if low samples
    if samples < 50:
        label = "⚠️LOW DATA"
    elif r >= 1.015:
        label = "⚠️FAST"
    elif r >= 1.005:
        label = "fast"
    elif r <= 0.975:
        label = "⚠️SLOW"
    elif r <= 0.990:
        label = "slow"
    else:
        label = "avg"

    return r, label, samples


def adjust_rating(rating: float, track_rating: float) -> float:
    """
    Adjust a speed rating based on track speed.
    Fast tracks inflate ratings, slow tracks deflate them.
    """
    # Simple adjustment: divide by track rating and multiply by 1.0 (baseline)
    # e.g., 102.0 at a 1.02 track = 102.0 / 1.02 * 1.0 = 100.0 adjusted
    # But we want to keep it in the same scale, so just show relative adjustment
    adjustment = (track_rating - 1.0) * 100  # e.g., 1.02 -> +2.0 points inflation
    return rating - adjustment


def format_runner_form(runner, race_distance: int, race_condition_num: int) -> str:
    """Format a runner's form with track ratings."""
    lines = []

    # Header
    lines.append(f"\n### {runner.tab_no}. {runner.name}")
    lines.append(f"**${runner.odds:.2f}** win / ${runner.place_odds:.2f} place")

    if runner.jockey:
        jockey_ae = f"{runner.jockey_a2e:.2f}" if runner.jockey_a2e else "?"
        lines.append(f"Jockey: {runner.jockey} (A/E: {jockey_ae})")
    if runner.trainer:
        trainer_ae = f"{runner.trainer_a2e:.2f}" if runner.trainer_a2e else "?"
        lines.append(f"Trainer: {runner.trainer} (A/E: {trainer_ae})")

    lines.append(f"Barrier: {runner.barrier} | Weight: {runner.weight}kg")

    if runner.first_up:
        lines.append("**FIRST UP**")
    elif runner.second_up:
        lines.append("**SECOND UP**")

    # Form table with track ratings
    if runner.form:
        lines.append("")
        lines.append("| Date | Track | Dist | Cond | Pos | Rating | Adj | Relevant? |")
        lines.append("|------|-------|------|------|-----|--------|-----|-----------|")

        best_relevant_adj = None

        for f in runner.form[:8]:  # Last 8 runs
            # Get track rating
            track_rating, track_label, samples = get_track_rating(f.track)

            # Check relevance to today's race
            dist_relevant = False
            cond_relevant = False

            # Distance relevance (±20%)
            if f.distance:
                dist_diff = abs(f.distance - race_distance) / race_distance
                dist_relevant = dist_diff <= 0.20

            # Condition relevance (±2 levels on 1-10 scale)
            cond_diff = None
            if f.condition_num and race_condition_num:
                cond_diff = abs(f.condition_num - race_condition_num)
                cond_relevant = cond_diff <= 2

            # Format rating with adjustment
            if f.rating and not f.is_barrier_trial:
                raw_rating = f.rating * 100
                rating_str = f"{raw_rating:.1f}"

                if track_rating and samples >= 50:
                    adj = adjust_rating(raw_rating, track_rating)
                    adj_str = f"{adj:.1f}"
                    # Track best relevant adjusted rating
                    if dist_relevant and cond_relevant:
                        if best_relevant_adj is None or adj > best_relevant_adj:
                            best_relevant_adj = adj
                elif track_rating and samples < 50:
                    adj_str = f"{raw_rating:.1f}?"  # Show raw with ? for low samples
                else:
                    adj_str = "-"
            else:
                rating_str = "N/A"
                adj_str = "-"

            # Build relevance indicator
            if f.is_barrier_trial:
                relevance = "TRIAL"
            elif dist_relevant and cond_relevant:
                if cond_diff == 0:
                    relevance = "✅ YES"
                else:
                    relevance = f"✅ YES (±{cond_diff})"
            elif dist_relevant:
                if cond_diff is not None:
                    relevance = f"~dist only (±{cond_diff} cond)"
                else:
                    relevance = "~dist only (?)"
            elif cond_relevant:
                relevance = "~cond only"
            else:
                if cond_diff is not None:
                    relevance = f"❌ no (±{cond_diff} cond)"
                else:
                    relevance = "❌ no"

            # Add eased flag
            if f.margin and f.margin > 8:
                relevance += " (eased)"

            # Position
            pos_str = f"{f.position}/{f.starters}" if f.position and f.starters else "?"

            # Distance string
            dist_str = f"{f.distance}m" if f.distance else "?"

            lines.append(f"| {f.date} | {f.track[:10]} | {dist_str} | {f.condition or '?'} | {pos_str} | {rating_str} | {adj_str} | {relevance} |")

        # Show best relevant rating summary
        if best_relevant_adj:
            lines.append(f"\n**Best relevant adj rating: {best_relevant_adj:.1f}**")
        else:
            lines.append(f"\n**⚠️ No relevant form (matching distance + condition)**")

    return "\n".join(lines)


def analyze_runner_adjusted_form(runner, race_distance: int, race_condition_num: int) -> dict:
    """
    Analyze a runner's form with track adjustments.
    Returns dict with best adjusted rating, wet track form, etc.
    """
    analysis = {
        "name": runner.name,
        "tab_no": runner.tab_no,
        "odds": runner.odds,
        "place_odds": runner.place_odds,
        "jockey_ae": runner.jockey_a2e,
        "trainer_ae": runner.trainer_a2e,
        "first_up": runner.first_up,
        "adjusted_ratings": [],
        "raw_ratings": [],  # For when adjusted not available
        "relevant_form_count": 0,
        "best_adj_rating": None,
        "best_raw_rating": None,  # Fallback
        "has_slow_track_form": False,  # Form from slow tracks (understated)
        "has_fast_track_form": False,  # Form from fast tracks (inflated)
        "has_course_form": False,  # Form at today's track
    }
    race_track = runner.form[0].track if runner.form else None  # Assume first form entry has today's track if any

    for f in runner.form[:8]:
        if f.is_barrier_trial:
            continue

        track_rating, track_label, samples = get_track_rating(f.track)

        # Check distance relevance (±20%)
        dist_relevant = False
        if f.distance:
            dist_diff = abs(f.distance - race_distance) / race_distance
            dist_relevant = dist_diff <= 0.20

        if f.rating:
            raw_rating = f.rating * 100

            # Check condition relevance (±2 levels on 1-10 scale)
            cond_relevant = False
            if f.condition_num and race_condition_num:
                cond_diff = abs(f.condition_num - race_condition_num)
                cond_relevant = cond_diff <= 2

            if dist_relevant and cond_relevant:
                analysis["raw_ratings"].append(raw_rating)
                analysis["relevant_form_count"] += 1

                if track_rating and samples >= 50:
                    adj = adjust_rating(raw_rating, track_rating)
                    analysis["adjusted_ratings"].append(adj)

                    # Track if form is from slow/fast tracks
                    if track_rating <= 0.990:
                        analysis["has_slow_track_form"] = True
                    if track_rating >= 1.005:
                        analysis["has_fast_track_form"] = True

    # Calculate bests
    if analysis["adjusted_ratings"]:
        analysis["best_adj_rating"] = max(analysis["adjusted_ratings"])
    if analysis["raw_ratings"]:
        analysis["best_raw_rating"] = max(analysis["raw_ratings"])

    return analysis


def generate_betting_recommendation(race_data, ai_result) -> str:
    """
    Generate a betting recommendation based on track-adjusted analysis.
    Returns a formatted recommendation string.
    """
    lines = []

    # Analyze all runners
    analyses = []
    for runner in race_data.runners:
        analysis = analyze_runner_adjusted_form(runner, race_data.distance, race_data.condition_num)
        analyses.append(analysis)

    # Sort by best rating (use adjusted if available, else raw)
    def get_best_rating(a):
        return a["best_adj_rating"] or a["best_raw_rating"] or 0

    rated_analyses = [a for a in analyses if get_best_rating(a) > 0]
    rated_analyses.sort(key=get_best_rating, reverse=True)

    # Get AI picks for comparison
    ai_picks = {c.tab_no: c for c in ai_result.contenders}

    lines.append(f"\n{'='*70}")
    lines.append(f"  💰 BETTING RECOMMENDATION")
    lines.append(f"{'='*70}")

    # Find potential bets - check ALL runners, not just top rated
    bets = []

    for analysis in analyses:
        if not (analysis["best_adj_rating"] or analysis["best_raw_rating"]):
            continue  # Skip horses with no form

        reasons = []
        confidence = 0
        best_rating = get_best_rating(analysis)

        # Check if they're in AI picks
        in_ai = analysis["tab_no"] in ai_picks
        ai_pick = ai_picks.get(analysis["tab_no"])

        # Best rating in field?
        if rated_analyses and analysis == rated_analyses[0]:
            reasons.append("Best adjusted rating in field")
            confidence += 2
        elif rated_analyses and len(rated_analyses) > 1 and analysis == rated_analyses[1]:
            reasons.append("2nd best adjusted rating")
            confidence += 1

        # Slow track form (understated ratings)?
        if analysis["has_slow_track_form"]:
            reasons.append("Has form from slow tracks (ratings understated)")
            confidence += 1

        # Fast track form warning
        if analysis["has_fast_track_form"] and not analysis["has_slow_track_form"]:
            reasons.append("⚠️ Form mostly from fast tracks (may be inflated)")
            confidence -= 1

        # Jockey/Trainer A/E - STRONGER weighting for exceptional A/E
        if analysis["jockey_ae"]:
            if analysis["jockey_ae"] >= 1.5:
                reasons.append(f"⭐ Exceptional jockey A/E: {analysis['jockey_ae']:.2f}")
                confidence += 2
            elif analysis["jockey_ae"] >= 1.3:
                reasons.append(f"Strong jockey A/E: {analysis['jockey_ae']:.2f}")
                confidence += 1

        if analysis["trainer_ae"]:
            if analysis["trainer_ae"] >= 1.3:
                reasons.append(f"Strong trainer A/E: {analysis['trainer_ae']:.2f}")
                confidence += 1
            elif analysis["trainer_ae"] >= 1.15:
                reasons.append(f"Good trainer A/E: {analysis['trainer_ae']:.2f}")

        # Price value check - more aggressive at longer odds
        odds = analysis["odds"]
        if odds:
            # Higher odds = need less confidence, shorter odds = need more
            if odds >= 10.0 and confidence >= 2:
                reasons.append(f"💰 Big value at ${odds:.2f}")
                confidence += 1
            elif odds >= 6.0 and confidence >= 2:
                reasons.append(f"Value at ${odds:.2f}")
            elif odds >= 4.0 and confidence >= 3:
                reasons.append(f"Fair price at ${odds:.2f}")
            elif odds < 3.0 and confidence < 4:
                reasons.append(f"⚠️ Short price ${odds:.2f} - need strong conviction")
                confidence -= 1

        # AI agreement bonus
        if in_ai and ai_pick.tipsheet_pick:
            reasons.append("AI tipsheet pick ⭐")
            confidence += 1
        elif in_ai:
            reasons.append("AI picked")

        if confidence >= 2 and reasons:
            bets.append({
                "analysis": analysis,
                "reasons": reasons,
                "confidence": confidence,
                "ai_pick": ai_pick,
                "best_rating": best_rating
            })

    # Sort by confidence, then by odds (prefer value)
    bets.sort(key=lambda x: (x["confidence"], x["analysis"]["odds"] or 0), reverse=True)

    if not bets:
        lines.append("\n  ❌ NO BET - No clear edge found after track adjustment")
        lines.append("     Either form is too similar or too much from fast tracks")
    else:
        for i, bet in enumerate(bets[:3]):
            analysis = bet["analysis"]
            rating_str = f"{analysis['best_adj_rating']:.1f}" if analysis["best_adj_rating"] else f"{analysis['best_raw_rating']:.1f}*" if analysis["best_raw_rating"] else "?"

            if bet["confidence"] >= 5:
                bet_type = "✅ STRONG BET"
            elif bet["confidence"] >= 3:
                bet_type = "✅ BET"
            else:
                bet_type = "🔸 LEAN"

            # Check for each-way
            place_odds = analysis["place_odds"]
            each_way = ""
            if place_odds and place_odds >= 1.80:
                each_way = f" (Each-way @ ${place_odds:.2f})"

            lines.append(f"\n  {bet_type}: {analysis['name']} (#{analysis['tab_no']}) @ ${analysis['odds']:.2f}{each_way}")
            lines.append(f"     Best Rating: {rating_str}")
            for reason in bet["reasons"]:
                lines.append(f"     • {reason}")

    lines.append(f"\n{'='*70}\n")
    return "\n".join(lines)


def run_second_opinion(track: str, race_number: int, date: str, ai_picks: str = None):
    """
    Format race data with track ratings for Claude (in conversation) to review.

    This does NOT call the AI predictor API - that's already done separately.
    Paste the AI picks below, then paste this output to Claude for second opinion.
    """

    print(f"\n{'='*70}")
    print(f"  SECOND OPINION DATA: {track} R{race_number} - {date}")
    print(f"{'='*70}")

    # Get race data
    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=True)

    if error:
        print(f"\nERROR: {error}")
        return

    # Format condition abbreviation
    cond_abbrev = f"{race_data.condition[0].upper()}{race_data.condition_num}" if race_data.condition_num else race_data.condition

    # Instructions for Claude
    print(f"\n{'='*70}")
    print(f"  📋 SECOND OPINION INSTRUCTIONS")
    print(f"{'='*70}")
    print(f"""
Today's race is {cond_abbrev}. Form is relevant if within ±2 condition levels.
(e.g., for S6: G4, S5, S6, S7, H8 are all relevant)

Form runs marked ✅ YES = matching distance (±20%) AND condition (±2 levels).

For each runner, compare their **Adj** (track-adjusted) ratings:
- Fast tracks (1.005+) inflate ratings → adjust DOWN
- Slow tracks (0.990-) deflate ratings → adjust UP
- Only trust adjustments from tracks with 50+ samples

Key questions:
1. Who has the best adjusted rating on RELEVANT form?
2. Is the price fair for that edge?
3. Are any horses being overlooked (slow track form = understated)?
4. Are any horses overrated (fast track form = inflated)?
""")

    # Placeholder for AI picks (user pastes these from tipsheet)
    print(f"{'='*70}")
    print(f"  AI PREDICTOR PICKS")
    print(f"{'='*70}")
    print("  [Paste AI picks here when sending to Claude]\n")

    # Race info
    print(f"\n{'='*70}")
    print(f"  RACE INFO")
    print(f"{'='*70}")
    # Format condition as abbreviation (e.g., "S6", "G4", "H8")
    cond_abbrev = f"{race_data.condition[0].upper()}{race_data.condition_num}" if race_data.condition_num else race_data.condition
    print(f"\n**{race_data.distance}m | {cond_abbrev} | {race_data.class_}**")
    print(f"Field: {len(race_data.runners)} runners")

    # Track rating for TODAY's track
    today_rating, today_label, today_samples = get_track_rating(track)
    if today_rating:
        print(f"Today's track rating: {today_rating:.4f} ({today_label}, {today_samples} samples)")
    else:
        print(f"Today's track rating: Not in database")

    # Key tracks reference
    print(f"\n{'='*70}")
    print(f"  TRACK RATINGS REFERENCE (from form)")
    print(f"{'='*70}")
    print("\n| Track | Rating | Type | Samples |")
    print("|-------|--------|------|---------|")

    # Collect tracks from form
    form_tracks = set()
    for runner in race_data.runners:
        for f in runner.form[:8]:
            form_tracks.add(f.track)

    for t in sorted(form_tracks):
        r, label, samples = get_track_rating(t)
        if r:
            print(f"| {t[:15]} | {r:.4f} | {label} | {samples} |")
        else:
            print(f"| {t[:15]} | ? | unknown | - |")

    # Format each runner
    print(f"\n{'='*70}")
    print(f"  RUNNER FORM WITH TRACK ADJUSTMENTS")
    print(f"{'='*70}")

    # Sort by odds
    sorted_runners = sorted(race_data.runners, key=lambda r: r.odds if r.odds else 999)

    for runner in sorted_runners:
        print(format_runner_form(runner, race_data.distance, race_data.condition_num))

    # Summary ranking by best relevant adjusted rating
    print(f"\n{'='*70}")
    print(f"  📊 RANKING BY BEST RELEVANT ADJUSTED RATING")
    print(f"{'='*70}")
    print(f"\nOnly includes form at ±20% distance AND ±2 condition levels of {cond_abbrev}:\n")

    rankings = []
    for runner in race_data.runners:
        analysis = analyze_runner_adjusted_form(runner, race_data.distance, race_data.condition_num)
        best = analysis.get('best_adj_rating') or analysis.get('best_raw_rating')
        if best:
            rankings.append({
                'name': runner.name,
                'tab_no': runner.tab_no,
                'odds': runner.odds,
                'best_adj': analysis.get('best_adj_rating'),
                'best_raw': analysis.get('best_raw_rating'),
                'jockey_ae': runner.jockey_a2e,
                'trainer_ae': runner.trainer_a2e,
                'has_slow': analysis.get('has_slow_track_form'),
                'has_fast': analysis.get('has_fast_track_form'),
            })

    # Sort by best adjusted (or raw if no adjusted)
    rankings.sort(key=lambda x: x['best_adj'] or x['best_raw'] or 0, reverse=True)

    print("| Rank | Horse | Odds | Best Adj | J A/E | T A/E | Track Type |")
    print("|------|-------|------|----------|-------|-------|------------|")
    for i, r in enumerate(rankings[:10], 1):
        adj_str = f"{r['best_adj']:.1f}" if r['best_adj'] else f"{r['best_raw']:.1f}*" if r['best_raw'] else "-"
        jae = f"{r['jockey_ae']:.2f}" if r['jockey_ae'] else "-"
        tae = f"{r['trainer_ae']:.2f}" if r['trainer_ae'] else "-"
        track_type = ""
        if r['has_slow']:
            track_type += "slow✓ "
        if r['has_fast']:
            track_type += "fast⚠️"
        if not track_type:
            track_type = "-"
        print(f"| {i} | {r['name'][:15]} | ${r['odds']:.2f} | {adj_str} | {jae} | {tae} | {track_type} |")

    if not rankings:
        print("  ⚠️ No runners with relevant form data!")

    print(f"\n* = raw rating (no track adjustment available)")
    print(f"slow✓ = has form from slow tracks (ratings may be understated)")
    print(f"fast⚠️ = has form from fast tracks (ratings may be inflated)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python tools/second_opinion.py <track> <race_number> <date>")
        print("Example: python tools/second_opinion.py 'Warwick' 6 '23-Feb-2026'")
        sys.exit(1)

    track = sys.argv[1]
    race_number = int(sys.argv[2])
    date = sys.argv[3]

    run_second_opinion(track, race_number, date)
