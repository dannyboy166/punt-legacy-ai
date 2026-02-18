"""
Automated betting experiment based purely on speed ratings.

The edge is speed ratings. AI analyzes raw form runs and estimates win probabilities.
Uses value-adjusted Dutch betting - equal profit if odds are fair, more profit on value picks.

Usage:
    # Single race
    python experiments/ratings_auto_bet.py --track "Randwick" --race 4 --date "12-Feb-2026"

    # Backtest mode (checks actual results)
    python experiments/ratings_auto_bet.py --track "Randwick" --race 4 --date "10-Feb-2026" --backtest
"""

import argparse
import os
import json
import re
from dotenv import load_dotenv
import anthropic

from core.race_data import RaceDataPipeline
from api.puntingform import PuntingFormAPI

load_dotenv()

SYSTEM_PROMPT = """You are a speed ratings analyst. Your job is to identify horses that could win based on their ratings.

**Your edge is speed ratings.** Prioritize runs at similar distance (±20%) and condition (±2 levels), but consider all form. More recent runs matter more.

Analyze each horse's form runs individually. Look at:
- Ratings at similar distance/condition to today's race
- Recency of runs
- Consistency vs one-off performances
- Trends (improving or declining)

**Output JSON:**
```json
{
  "bet": true | false,
  "selections": [
    {
      "horse": "Name",
      "tab_no": 1,
      "odds": 3.50,
      "win_probability": 35,
      "reasoning": "Why you think this probability based on ratings"
    }
  ],
  "skip_reason": null | "reason if no bet",
  "summary": "Brief analysis"
}
```

**Rules:**
- List 0-5 horses you think could win based on ratings
- win_probability = your estimate of their ACTUAL chance to win (must sum to ≤100%)
- Compare your probability to implied odds probability:
  - If your prob > implied prob = VALUE (e.g., you say 25%, odds imply 20%)
  - If your prob < implied prob = NO VALUE
- Only select horses where you see value OR they're genuine winning chances

**NO BET (bet: false, selections: []) when:**
- Field is too even on ratings - no clear standouts
- Too many unknowns (horses with 0 race form, only trials)
- No value exists in the market
"""


def calculate_dutch_stakes(selections: list, total_stake: float = 100) -> list:
    """
    Calculate standard Dutch stakes for equal profit.

    Returns list of selections with 'dutch_stake' added.
    """
    if not selections:
        return []

    # Calculate implied probabilities from odds
    total_implied = sum(1 / s['odds'] for s in selections)

    for s in selections:
        # Standard Dutch: stake proportional to 1/odds
        s['dutch_stake'] = total_stake * (1 / s['odds']) / total_implied
        s['implied_prob'] = (1 / s['odds']) * 100

    return selections


def calculate_value_adjusted_stakes(selections: list, total_stake: float = 100) -> list:
    """
    Calculate value-adjusted Dutch stakes (fixed stake method).

    If AI thinks probability > implied probability, increase stake proportionally.
    Scale others down to maintain total stake.
    """
    if not selections:
        return []

    # First get standard Dutch stakes
    selections = calculate_dutch_stakes(selections, total_stake)

    # Calculate value multipliers
    for s in selections:
        ai_prob = s.get('win_probability', s['implied_prob'])
        implied_prob = s['implied_prob']

        # Value multiplier: AI_prob / implied_prob
        # If AI thinks 25% and implied is 20%, multiplier = 1.25
        s['value_multiplier'] = ai_prob / implied_prob if implied_prob > 0 else 1.0
        s['is_value'] = ai_prob > implied_prob

    # Apply multipliers to Dutch stakes
    for s in selections:
        s['raw_adjusted_stake'] = s['dutch_stake'] * s['value_multiplier']

    # Scale to hit total stake
    raw_total = sum(s['raw_adjusted_stake'] for s in selections)
    scale_factor = total_stake / raw_total if raw_total > 0 else 1

    for s in selections:
        s['final_stake'] = round(s['raw_adjusted_stake'] * scale_factor, 2)
        s['potential_return'] = round(s['final_stake'] * s['odds'], 2)
        s['potential_profit'] = round(s['potential_return'] - total_stake, 2)

    return selections


def calculate_fixed_return_stakes(selections: list, target_return: float = 100) -> tuple[list, str]:
    """
    Calculate value-adjusted fixed return stakes.

    Base: stake = target / odds (so any winner returns target)
    Adjustment: multiply stake by value_multiplier (AI_prob / implied_prob)
    Result: value picks return MORE than target, non-value return LESS

    Safety rule: sum(1/odds) must be < 1, otherwise guaranteed loss.

    Returns: (selections with stakes, error_message or None)
    """
    if not selections:
        return [], "No selections"

    # Calculate implied probabilities and value
    for s in selections:
        s['implied_prob'] = (1 / s['odds']) * 100
        ai_prob = s.get('win_probability', s['implied_prob'])
        s['value_multiplier'] = ai_prob / s['implied_prob'] if s['implied_prob'] > 0 else 1.0
        s['is_value'] = ai_prob > s['implied_prob']

    # SAFETY CHECK: sum(1/odds) must be < 1
    inverse_odds_sum = sum(1 / s['odds'] for s in selections)

    if inverse_odds_sum >= 1:
        return selections, f"INVALID BET: Combined odds too short (sum 1/odds = {inverse_odds_sum:.2f} >= 1). Need longer odds or fewer selections."

    # Calculate value-adjusted stakes
    # Base stake = target / odds
    # Adjusted stake = base × value_multiplier
    for s in selections:
        s['base_stake'] = target_return / s['odds']
        s['final_stake'] = round(s['base_stake'] * s['value_multiplier'], 2)
        s['potential_return'] = round(s['final_stake'] * s['odds'], 2)

    # Calculate totals
    total_stake = sum(s['final_stake'] for s in selections)

    for s in selections:
        s['potential_profit'] = round(s['potential_return'] - total_stake, 2)

    # Safety check - ensure we profit if ANY selection wins
    min_return = min(s['potential_return'] for s in selections)
    if min_return < total_stake:
        return selections, f"WARNING: Lowest return (${min_return:.2f}) < total stake (${total_stake:.2f}) - some wins would be losses"

    return selections, None


def run_prediction(track: str, race_number: int, date: str, backtest: bool = False, show_form: bool = True, fixed_return: float = None):
    """Run ratings-based prediction.

    Args:
        fixed_return: If set, use fixed return method (any winner returns this amount).
                     If None, use fixed stake method ($100 stake).
    """

    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=backtest)

    if error:
        print(f"Error: {error}")
        return None

    race_text = race_data.to_prompt_text()

    print(f"\n{'='*60}")
    print(f"  {track.upper()} RACE {race_number} - {date}")
    print(f"  {race_data.distance}m | {race_data.condition}")
    print(f"{'='*60}\n")

    # Show all runners with their raw form (no averages - let AI analyze)
    if show_form:
        print("  FIELD FORM & RATINGS:")
        print("  " + "-"*56)

        today_dist = race_data.distance
        today_cond = race_data.condition_num

        for runner in sorted(race_data.runners, key=lambda x: x.tab_no):
            odds_str = f"${runner.odds:.2f}" if runner.odds else "N/A"
            implied = f"({100/runner.odds:.0f}%)" if runner.odds else ""
            print(f"\n  #{runner.tab_no} {runner.name} - {odds_str} {implied}")

            if not runner.form:
                print("    ⚠️ NO FORM")
                continue

            race_runs = [f for f in runner.form if not f.is_barrier_trial]
            trial_runs = [f for f in runner.form if f.is_barrier_trial]

            if not race_runs:
                print(f"    ⚠️ NO RACE FORM ({len(trial_runs)} trials only)")
                continue

            # Show all runs - let AI determine relevance
            print(f"    Form ({len(race_runs)} runs):")
            for f in race_runs[:8]:
                # Mark if relevant (similar distance/condition)
                dist_diff = abs(f.distance - today_dist) / today_dist if today_dist else 1
                cond_diff = abs(f.condition_num - today_cond)
                is_relevant = dist_diff <= 0.20 and cond_diff <= 2
                marker = "→" if is_relevant else " "

                rating_str = f"{f.rating*100:.1f}" if f.rating else "N/A"
                print(f"    {marker} {f.date} | {f.distance}m {f.condition} | {f.position}/{f.starters} | {f.margin}L | {rating_str}")

        print("\n  " + "-"*56)
        print("  (→ = similar distance/condition to today)")
        print()

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Analyze this race:\n\n{race_text}\n\nRespond with JSON only."}
        ],
    )

    raw = response.content[0].text

    # Parse JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
        else:
            print("No JSON found")
            print(raw)
            return None
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(raw)
        return None

    # Display results
    bet = data.get("bet", False)
    selections = data.get("selections", [])

    if not bet or not selections:
        print("  NO BET")
        print(f"  Reason: {data.get('skip_reason', 'N/A')}")
        return {"bet": False, "selections": [], "track": track, "race_number": race_number, "date": date}

    # Calculate stakes based on method
    if fixed_return:
        selections, stake_error = calculate_fixed_return_stakes(selections, fixed_return)
        method_name = f"Fixed Return (target ${fixed_return:.0f})"

        if stake_error:
            print(f"  ⚠️ {stake_error}")
            if "INVALID BET" in stake_error:
                print("  Skipping this race - odds too short for fixed return method.")
                return {"bet": False, "selections": [], "track": track, "race_number": race_number, "date": date, "skip_reason": stake_error}
    else:
        selections = calculate_value_adjusted_stakes(selections)
        method_name = "Value-Adjusted Dutch ($100 stake)"

    print(f"  SELECTIONS ({method_name}):")
    print("  " + "-"*56)

    total_stake = sum(s['final_stake'] for s in selections)

    for sel in selections:
        value_marker = "✓ VALUE" if sel.get('is_value') else ""
        print(f"\n    {sel['horse']} (#{sel.get('tab_no', '?')})")
        print(f"      Odds: ${sel.get('odds', '?'):.2f} (implied {sel.get('implied_prob', 0):.0f}%)")
        print(f"      AI Probability: {sel.get('win_probability', '?')}% {value_marker}")
        print(f"      Stake: ${sel.get('final_stake', 0):.2f} → Win ${sel.get('potential_return', 0):.2f} (profit ${sel.get('potential_profit', 0):+.2f})")
        print(f"      {sel.get('reasoning', '')}")

    print(f"\n  " + "-"*56)
    print(f"  Total stake: ${total_stake:.2f}")

    if fixed_return:
        guaranteed_profit = min(s['potential_profit'] for s in selections)
        print(f"  Guaranteed profit if any wins: ${guaranteed_profit:+.2f}")

    print(f"\n  Summary: {data.get('summary', '')}")

    result = {
        "bet": True,
        "selections": selections,
        "track": track,
        "race_number": race_number,
        "date": date,
        "distance": race_data.distance,
        "condition": race_data.condition,
    }

    # If backtest mode, check actual results
    if backtest:
        result["pnl"] = check_results(track, race_number, date, selections)

    print(f"\n{'='*60}\n")
    return result


def check_results(track: str, race_number: int, date: str, selections: list) -> dict:
    """Check actual race results and calculate P&L."""

    pf_api = PuntingFormAPI()

    try:
        # Get meetings to find meeting ID
        meetings = pf_api.get_meetings(date)
        meeting_id = None
        for m in meetings:
            if track.lower() in m.get('track', {}).get('name', '').lower():
                meeting_id = m.get('meetingId')
                break

        if not meeting_id:
            print("  Could not find meeting")
            return {"error": "meeting not found"}

        # Get form data which includes finishing positions
        form_data = pf_api.get_form(meeting_id, race_number, runs=1)
        if not form_data:
            print("  Could not fetch results")
            return {"error": "no results"}

        # Get finishing positions
        results = {}
        for runner in form_data:
            pos = runner.get('finishingPosition')
            name = runner.get('runnerName', '').strip()
            sp = runner.get('startingPrice', 0)
            if pos and name:
                results[name.upper()] = {"position": pos, "sp": sp}

        # Calculate P&L
        total_stake = sum(s.get('final_stake', 0) for s in selections)
        total_return = 0

        print("\n  RESULTS:")
        for sel in selections:
            horse = sel['horse'].upper()
            stake = sel.get('final_stake', 0)

            if horse in results:
                pos = results[horse]["position"]
                sp = results[horse]["sp"] or sel.get('odds', 0)

                if pos == 1:
                    win_return = stake * sp
                    total_return += win_return
                    print(f"    ✅ {sel['horse']} - 1st @ ${sp} | Stake ${stake:.2f} → ${win_return:.2f}")
                else:
                    print(f"    ❌ {sel['horse']} - {pos}th | Stake ${stake:.2f} → $0")
            else:
                print(f"    ⚠️ {sel['horse']} - not found in results")

        profit = total_return - total_stake
        roi = (profit / total_stake) * 100 if total_stake > 0 else 0

        print(f"\n  P&L: ${profit:+.2f} ({roi:+.1f}% ROI)")

        return {
            "stake": total_stake,
            "return": total_return,
            "profit": profit,
            "roi": roi,
        }

    except Exception as e:
        print(f"  Error checking results: {e}")
        return {"error": str(e)}


def run_meeting_backtest(track: str, date: str, race_start: int = 1, race_end: int = 10, fixed_return: float = None):
    """Run backtest across multiple races at a meeting."""

    method = f"Fixed Return ${fixed_return:.0f}" if fixed_return else "Fixed Stake $100"

    print(f"\n{'#'*60}")
    print(f"  BACKTEST: {track.upper()} - {date}")
    print(f"  Races {race_start} to {race_end}")
    print(f"  Method: {method}")
    print(f"{'#'*60}")

    results = []
    total_stake = 0
    total_return = 0
    races_bet = 0
    races_won = 0

    for race_num in range(race_start, race_end + 1):
        try:
            result = run_prediction(track, race_num, date, backtest=True, show_form=False, fixed_return=fixed_return)
            if result:
                results.append(result)

                if result.get("bet") and result.get("pnl"):
                    pnl = result["pnl"]
                    if "stake" in pnl:
                        total_stake += pnl["stake"]
                        total_return += pnl["return"]
                        races_bet += 1
                        if pnl["profit"] > 0:
                            races_won += 1
        except Exception as e:
            print(f"Error on race {race_num}: {e}")
            continue

    # Summary
    print(f"\n{'#'*60}")
    print(f"  MEETING SUMMARY: {track.upper()} - {date}")
    print(f"{'#'*60}")
    print(f"  Races analysed: {len(results)}")
    print(f"  Races bet: {races_bet}")
    print(f"  Races won: {races_won}")

    if total_stake > 0:
        total_profit = total_return - total_stake
        total_roi = (total_profit / total_stake) * 100
        print(f"  Total stake: ${total_stake:.2f}")
        print(f"  Total return: ${total_return:.2f}")
        print(f"  Total P&L: ${total_profit:+.2f}")
        print(f"  ROI: {total_roi:+.1f}%")

    print(f"{'#'*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ratings-based auto bet experiment")
    parser.add_argument("--track", required=True, help="Track name")
    parser.add_argument("--race", type=int, help="Race number (omit for full meeting)")
    parser.add_argument("--date", required=True, help="Date (dd-MMM-yyyy)")
    parser.add_argument("--backtest", action="store_true", help="Check actual results and calculate P&L")
    parser.add_argument("--race-start", type=int, default=1, help="Start race for meeting backtest")
    parser.add_argument("--race-end", type=int, default=10, help="End race for meeting backtest")
    parser.add_argument("--no-form", action="store_true", help="Hide form display")
    parser.add_argument("--fixed-return", type=float, default=None,
                       help="Use fixed return method (e.g., --fixed-return 100 means any winner returns $100)")

    args = parser.parse_args()

    if args.race:
        run_prediction(args.track, args.race, args.date, backtest=args.backtest,
                      show_form=not args.no_form, fixed_return=args.fixed_return)
    else:
        run_meeting_backtest(args.track, args.date, args.race_start, args.race_end,
                            fixed_return=args.fixed_return)
