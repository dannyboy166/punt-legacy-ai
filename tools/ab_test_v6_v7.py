#!/usr/bin/env python3
"""
A/B Test: V6 (current live) vs V7 (recency + trajectory)

Uses the EXACT SAME data pipeline as the live predictor:
- core/race_data.py for data fetching and formatting
- core/predictor.py prompts (SYSTEM_PROMPT, V7_SYSTEM_PROMPT)
- Only difference from live: SP odds instead of live odds (allow_finished=True)

Usage:
    python tools/ab_test_v6_v7.py              # Run all races
    python tools/ab_test_v6_v7.py --limit 5    # Test with 5 races
    python tools/ab_test_v6_v7.py --resume <file>  # Resume from crash

Output: data/ab_results/ab_test_v6_v7_TIMESTAMP.json
"""

import sys
sys.path.insert(0, '/Users/danielsamus/punt-legacy-ai')

import os
import json
import re
import argparse
from datetime import datetime

import anthropic
from dotenv import load_dotenv

# Use the ACTUAL live predictor code
from core.race_data import RaceDataPipeline
from core.predictor import SYSTEM_PROMPT, V7_SYSTEM_PROMPT

load_dotenv()


def load_unique_races_from_ab_results() -> list[dict]:
    """
    Load unique races from all ab_results JSON files.
    Returns list of {track, race_number, date, full_results}
    """
    results_dir = "data/ab_results"
    unique_races = {}

    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} not found")
        return []

    for fname in os.listdir(results_dir):
        if not fname.endswith('.json'):
            continue
        if fname.startswith('ab_test_v6_v7'):
            continue  # Skip our own output files

        filepath = os.path.join(results_dir, fname)
        try:
            with open(filepath) as f:
                data = json.load(f)

            for race in data.get("races", []):
                key = (race["track"], race["race_number"], race["date"])

                # Only add if has full_results (actual race outcomes)
                full_results = race.get("full_results", {})
                if full_results:
                    unique_races[key] = {
                        "track": race["track"],
                        "race_number": race["race_number"],
                        "date": race["date"],
                        "full_results": full_results
                    }
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

    # Sort by date, then track, then race number
    def sort_key(r):
        try:
            dt = datetime.strptime(r["date"], "%d-%b-%Y")
        except:
            dt = datetime.min
        return (dt, r["track"], r["race_number"])

    races = sorted(unique_races.values(), key=sort_key)

    # Filter to March 21, 2026 onwards (API doesn't have older data)
    cutoff = datetime(2026, 3, 21)
    races = [r for r in races if datetime.strptime(r["date"], "%d-%b-%Y") >= cutoff]

    return races


def run_prediction(prompt_text: str, system_prompt: str) -> str:
    """Run Claude prediction - same as live predictor."""
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # Same model as live
        max_tokens=2000,
        temperature=0.2,  # Same temp as live
        system=system_prompt,
        messages=[{"role": "user", "content": f"Analyze this race and pick your contenders (0-3).\n\n{prompt_text}\n\nRespond with JSON only."}]
    )

    return response.content[0].text


def parse_prediction(raw_response: str) -> dict:
    """Parse Claude's JSON response."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {"contenders": [], "summary": f"Parse error: {raw_response[:100]}"}


def run_v6_v7_comparison(
    races: list[dict],
    output_file: str,
    existing_results: dict = None,
    verbose: bool = True
) -> dict:
    """
    Run V6 and V7 predictions using the EXACT same pipeline as live predictor.
    """
    pipeline = RaceDataPipeline()

    # Initialize or resume results
    if existing_results:
        results = existing_results
        completed_keys = {
            (r["track"], r["race_number"], r["date"])
            for r in results.get("races", [])
        }
    else:
        results = {
            "test_time": datetime.now().isoformat(),
            "variations_tested": ["v6_live", "v7_recency_trajectory"],
            "data_source": "core/race_data.py (same as live predictor)",
            "prompts": "core/predictor.py SYSTEM_PROMPT and V7_SYSTEM_PROMPT",
            "races_tested": len(races),
            "races": []
        }
        completed_keys = set()

    # Filter out already completed races
    remaining_races = [
        r for r in races
        if (r["track"], r["race_number"], r["date"]) not in completed_keys
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  V6 vs V7 A/B TEST (USING LIVE PREDICTOR PIPELINE)")
        print(f"  {len(remaining_races)} races to test ({len(completed_keys)} already done)")
        print(f"{'='*60}")

    for i, race in enumerate(remaining_races):
        track = race["track"]
        race_number = race["race_number"]
        date = race["date"]
        full_results = race["full_results"]

        if verbose:
            print(f"\n  [{i+1}/{len(remaining_races)}] {track} R{race_number} ({date})")
            winner = [h for h, r in full_results.items() if r.get("won")]
            if winner:
                print(f"    Winner: {winner[0].title()}")

        # Get race data using the EXACT same pipeline as live predictor
        # allow_finished=True bypasses "race finished" check
        # For historical races, Ladbrokes won't have data - pipeline falls back to PF SP odds
        race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=True)

        if error:
            if verbose:
                print(f"    Error: {error}")
            results["races"].append({
                "track": track,
                "race_number": race_number,
                "date": date,
                "full_results": full_results,
                "error": error,
                "variations": {}
            })
            # Save incrementally
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            continue

        # Generate prompt text using EXACT same method as live predictor
        # v6_mode=True gives us the V6 format with Notes column
        prompt_text = race_data.to_prompt_text(include_venue_adjusted=True, v6_mode=True)

        race_result = {
            "track": track,
            "race_number": race_number,
            "date": date,
            "distance": race_data.distance,
            "condition": race_data.condition,
            "full_results": full_results,
            "variations": {}
        }

        # Test both V6 and V7 prompts with SAME data
        variations = [
            ("v6_live", SYSTEM_PROMPT),
            ("v7_recency_trajectory", V7_SYSTEM_PROMPT),
        ]

        for var_key, system_prompt in variations:
            if verbose:
                print(f"    Testing {var_key}...", end=" ", flush=True)

            # Run prediction
            raw_response = run_prediction(prompt_text, system_prompt)
            parsed = parse_prediction(raw_response)

            contenders = parsed.get("contenders", [])
            summary = parsed.get("summary", "")

            # Match contenders to actual results
            for c in contenders:
                horse_name = c.get("horse", "").lower().strip()
                if horse_name in full_results:
                    actual = full_results[horse_name]
                    c["actual_position"] = actual["position"]
                    c["actual_won"] = actual["won"]
                    c["actual_placed"] = actual["placed"]
                    c["sp"] = actual.get("sp", c.get("odds"))
                else:
                    # Try partial match
                    for h_name, actual in full_results.items():
                        if horse_name in h_name or h_name in horse_name:
                            c["actual_position"] = actual["position"]
                            c["actual_won"] = actual["won"]
                            c["actual_placed"] = actual["placed"]
                            c["sp"] = actual.get("sp", c.get("odds"))
                            break

            race_result["variations"][var_key] = {
                "contenders": contenders,
                "summary": summary
            }

            if verbose:
                if not contenders:
                    print(f"0 picks")
                else:
                    wins = sum(1 for c in contenders if c.get("actual_won"))
                    print(f"{len(contenders)} picks, {wins} won")

        results["races"].append(race_result)

        # Save incrementally after EVERY race
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


def print_v6_v7_comparison(results: dict):
    """Print comparison between V6 and V7."""
    print(f"\n{'='*70}")
    print("  V6 vs V7 COMPARISON (LIVE PREDICTOR PIPELINE)")
    print(f"{'='*70}")

    # Calculate stats for each variation
    var_stats = {}

    for race in results.get("races", []):
        for var_key, var_result in race.get("variations", {}).items():
            if var_key not in var_stats:
                var_stats[var_key] = {
                    "total_picks": 0,
                    "wins": 0,
                    "places": 0,
                    "profit": 0,
                    "by_tag": {}
                }

            for c in var_result.get("contenders", []):
                tag = c.get("tag", "Unknown")
                odds = c.get("sp", c.get("odds", 0))
                won = c.get("actual_won", False)
                placed = c.get("actual_placed", False)

                var_stats[var_key]["total_picks"] += 1
                if won:
                    var_stats[var_key]["wins"] += 1
                    var_stats[var_key]["profit"] += (odds - 1)
                else:
                    var_stats[var_key]["profit"] -= 1
                if placed:
                    var_stats[var_key]["places"] += 1

                # By tag
                if tag not in var_stats[var_key]["by_tag"]:
                    var_stats[var_key]["by_tag"][tag] = {
                        "picks": 0, "wins": 0, "places": 0, "profit": 0
                    }
                var_stats[var_key]["by_tag"][tag]["picks"] += 1
                if won:
                    var_stats[var_key]["by_tag"][tag]["wins"] += 1
                    var_stats[var_key]["by_tag"][tag]["profit"] += (odds - 1)
                else:
                    var_stats[var_key]["by_tag"][tag]["profit"] -= 1
                if placed:
                    var_stats[var_key]["by_tag"][tag]["places"] += 1

    # Print overall comparison
    print(f"\n{'Variation':<28} {'Picks':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
    print("-" * 70)

    for var_key in ["v6_live", "v7_recency_trajectory"]:
        stats = var_stats.get(var_key, {})
        picks = stats.get("total_picks", 0)
        wins = stats.get("wins", 0)
        profit = stats.get("profit", 0)

        win_pct = (wins / picks * 100) if picks > 0 else 0
        roi = (profit / picks * 100) if picks > 0 else 0

        print(f"{var_key:<28} {picks:>6} {wins:>6} {win_pct:>6.1f}% {profit:>+10.2f} {roi:>+7.1f}%")

    # Print by-tag comparison
    print(f"\n{'='*70}")
    print("  BY TAG COMPARISON")
    print(f"{'='*70}")

    all_tags = set()
    for var_key in ["v6_live", "v7_recency_trajectory"]:
        stats = var_stats.get(var_key, {})
        all_tags.update(stats.get("by_tag", {}).keys())

    for tag in sorted(all_tags):
        print(f"\n  {tag}:")
        print(f"  {'Variation':<26} {'Picks':>6} {'Wins':>6} {'Win%':>7} {'ROI':>8}")
        print(f"  {'-'*54}")

        for var_key in ["v6_live", "v7_recency_trajectory"]:
            stats = var_stats.get(var_key, {})
            tag_stats = stats.get("by_tag", {}).get(tag, {})
            picks = tag_stats.get("picks", 0)
            wins = tag_stats.get("wins", 0)
            profit = tag_stats.get("profit", 0)

            win_pct = (wins / picks * 100) if picks > 0 else 0
            roi = (profit / picks * 100) if picks > 0 else 0

            print(f"  {var_key:<26} {picks:>6} {wins:>6} {win_pct:>6.1f}% {roi:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="A/B test V6 vs V7 using live predictor pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of races to test")
    parser.add_argument("--resume", type=str, help="Resume from partial results file")
    parser.add_argument("--stats-only", action="store_true", help="Just print stats from existing file")

    args = parser.parse_args()

    # Create output file path
    os.makedirs("data/ab_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/ab_results/ab_test_v6_v7_{timestamp}.json"

    # Stats only mode
    if args.stats_only and args.resume:
        with open(args.resume) as f:
            results = json.load(f)
        print_v6_v7_comparison(results)
        return

    # Load races
    print("Loading races from ab_results...")
    races = load_unique_races_from_ab_results()
    print(f"Found {len(races)} unique races with full results")

    if not races:
        print("No races found!")
        return

    # Apply limit
    if args.limit:
        races = races[:args.limit]
        print(f"Limited to {len(races)} races for testing")

    # Check for resume
    existing_results = None
    if args.resume:
        if os.path.exists(args.resume):
            with open(args.resume) as f:
                existing_results = json.load(f)
            output_file = args.resume  # Continue writing to same file
            print(f"Resuming from {args.resume}")
        else:
            print(f"Resume file not found: {args.resume}")
            return

    print(f"Output file: {output_file}")
    print(f"\nUsing EXACT same pipeline as live predictor:")
    print(f"  - Data: core/race_data.py RaceDataPipeline")
    print(f"  - Format: to_prompt_text(v6_mode=True)")
    print(f"  - V6 prompt: core/predictor.py SYSTEM_PROMPT")
    print(f"  - V7 prompt: core/predictor.py V7_SYSTEM_PROMPT")

    # Estimate time
    est_minutes = len(races) * 2 * 0.4  # 2 variations, ~24 sec each
    print(f"\nEstimated time: ~{est_minutes:.0f} minutes")
    print(f"Estimated cost: ~${len(races) * 2 * 0.025:.2f}")

    # Run test
    results = run_v6_v7_comparison(
        races,
        output_file,
        existing_results=existing_results,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS SAVED TO: {output_file}")
    print(f"{'='*60}")

    print_v6_v7_comparison(results)


if __name__ == "__main__":
    main()
