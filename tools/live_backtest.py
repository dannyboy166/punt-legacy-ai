#!/usr/bin/env python3
"""
Backtest the LIVE predictor on past races using SP odds.

This uses the exact same predictor as production but with:
- allow_finished=True to get SP odds instead of live odds
- Results calculated at SP odds

Usage:
    # Single race test
    python tools/live_backtest.py --track Randwick --race 1 --date 18-Apr-2026

    # Full track
    python tools/live_backtest.py --track Randwick --date 18-Apr-2026

    # Multiple tracks
    python tools/live_backtest.py --tracks "Randwick,Eagle Farm,Morphettville" --date 18-Apr-2026

Results saved to: data/live_backtest_DDMMMYYYY.json (appends, doesn't overwrite)
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.puntingform import PuntingFormAPI
from core.race_data import RaceDataPipeline
from core.predictor import Predictor


def get_results_file(date: str) -> str:
    """Get results file path for a date."""
    safe_date = date.replace("-", "").lower()
    return f"data/live_backtest_{safe_date}.json"


def load_existing_results(filepath: str) -> dict:
    """Load existing results or return empty dict."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {"races": [], "completed_keys": []}


def save_results(filepath: str, results: dict):
    """Save results to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def get_race_results(pf_api: PuntingFormAPI, track: str, race_number: int, date: str) -> dict:
    """Get finishing positions and SP odds from PuntingForm."""
    # First get meeting ID
    meetings = pf_api.get_meetings(date)
    meeting_id = None
    for m in meetings:
        track_name = m.get("track", {}).get("name", "").lower()
        if track_name == track.lower():
            meeting_id = m.get("meetingId")
            break

    if not meeting_id:
        return {}

    fields = pf_api.get_fields(meeting_id, race_number)
    if not fields:
        return {}

    # fields is a dict with "races" key
    races = fields.get("races", [])
    for race in races:
        if race.get("number") != race_number:
            continue

        results = {}
        for runner in race.get("runners", []):
            name = runner.get("name", "")
            # Try both possible field names
            position = runner.get("position") or runner.get("pos", 0)
            sp = runner.get("priceSP") or runner.get("SP", 0) or 0

            # Skip scratched
            if position == 0 or sp == 0:
                continue

            results[name.lower()] = {
                "position": position,
                "sp": sp
            }
        return results

    return {}


def run_backtest_race(track: str, race_number: int, date: str) -> dict:
    """Run backtest on a single race."""
    pipeline = RaceDataPipeline()
    predictor = Predictor()
    pf_api = PuntingFormAPI()

    # Get race data with allow_finished=True (uses SP odds)
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=True)

    if error:
        return {
            "track": track,
            "race_number": race_number,
            "date": date,
            "error": error,
            "contenders": []
        }

    # Run prediction
    result = predictor.predict(race_data)

    # Get actual results
    race_results = get_race_results(pf_api, track, race_number, date)

    # Calculate outcomes for each contender
    contenders = []
    field_size = len(race_results)
    place_pays = 3 if field_size >= 8 else (2 if field_size >= 5 else 1)

    for c in result.contenders:
        horse_key = c.horse.lower()
        horse_result = race_results.get(horse_key, {})
        position = horse_result.get("position")
        sp = horse_result.get("sp", c.odds)  # Use SP if available, else prediction odds

        won = position == 1
        placed = position is not None and position <= place_pays

        contenders.append({
            "horse": c.horse,
            "tab_no": c.tab_no,
            "tag": c.tag,
            "prediction_odds": c.odds,
            "sp_odds": sp,
            "position": position,
            "won": won,
            "placed": placed,
            "tipsheet_pick": c.tipsheet_pick,
            "analysis": c.analysis[:100] + "..." if len(c.analysis) > 100 else c.analysis
        })

    return {
        "track": track,
        "race_number": race_number,
        "date": date,
        "distance": race_data.distance,
        "condition": race_data.condition,
        "field_size": field_size,
        "contenders": contenders,
        "summary": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary
    }


def run_backtest_track(track: str, date: str, results_file: str):
    """Run backtest on all races at a track."""
    pf_api = PuntingFormAPI()

    # Get meetings to find race count
    meetings = pf_api.get_meetings(date)
    track_meeting = None
    for m in meetings:
        track_name = m.get("track", {}).get("name", "").lower()
        if track_name == track.lower():
            track_meeting = m
            break

    if not track_meeting:
        print(f"Track {track} not found for {date}")
        return

    race_count = track_meeting.get("numberOfRaces", 8)
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {track} - {date}")
    print(f"  Races: 1-{race_count}")
    print(f"{'='*60}")

    # Load existing results
    results = load_existing_results(results_file)

    for race_num in range(1, race_count + 1):
        race_key = f"{track}-{date}-R{race_num}"

        # Skip if already done
        if race_key in results["completed_keys"]:
            print(f"\n  R{race_num}: Already completed, skipping...")
            continue

        print(f"\n  R{race_num}: Running prediction...")

        try:
            race_result = run_backtest_race(track, race_num, date)

            if race_result.get("error"):
                print(f"    Error: {race_result['error']}")
            else:
                # Print contenders
                for c in race_result["contenders"]:
                    status = "WON!" if c["won"] else ("Placed" if c["placed"] else f"#{c['position']}")
                    star = "⭐" if c["tipsheet_pick"] else "  "
                    print(f"    {star} {c['horse']} @ ${c['sp_odds']:.2f} - {c['tag']} - {status}")

            # Save result
            results["races"].append(race_result)
            results["completed_keys"].append(race_key)
            save_results(results_file, results)
            print(f"    Saved to {results_file}")

        except Exception as e:
            print(f"    Exception: {e}")
            continue


def calculate_stats(results: dict) -> dict:
    """Calculate overall stats from results."""
    stats = {
        "total_races": 0,
        "total_picks": 0,
        "wins": 0,
        "places": 0,
        "profit": 0,
        "by_tag": {}
    }

    for race in results.get("races", []):
        if race.get("error"):
            continue

        stats["total_races"] += 1

        for c in race.get("contenders", []):
            tag = c["tag"]
            if tag == "Main danger":
                tag = "Each-way chance"

            if tag not in stats["by_tag"]:
                stats["by_tag"][tag] = {"picks": 0, "wins": 0, "places": 0, "profit": 0}

            stats["total_picks"] += 1
            stats["by_tag"][tag]["picks"] += 1

            sp = c.get("sp_odds", c.get("prediction_odds", 0))

            if c["won"]:
                stats["wins"] += 1
                stats["places"] += 1
                stats["profit"] += sp - 1
                stats["by_tag"][tag]["wins"] += 1
                stats["by_tag"][tag]["places"] += 1
                stats["by_tag"][tag]["profit"] += sp - 1
            elif c["placed"]:
                stats["places"] += 1
                stats["profit"] -= 1
                stats["by_tag"][tag]["places"] += 1
                stats["by_tag"][tag]["profit"] -= 1
            elif c["position"] is not None:
                stats["profit"] -= 1
                stats["by_tag"][tag]["profit"] -= 1

    return stats


def print_stats(stats: dict):
    """Print stats summary."""
    print(f"\n{'='*60}")
    print("  BACKTEST RESULTS")
    print(f"{'='*60}")

    print(f"\n  Races: {stats['total_races']}")
    print(f"  Total picks: {stats['total_picks']}")

    if stats['total_picks'] > 0:
        win_pct = stats['wins'] / stats['total_picks'] * 100
        roi = stats['profit'] / stats['total_picks'] * 100
        print(f"  Wins: {stats['wins']} ({win_pct:.1f}%)")
        print(f"  Places: {stats['places']}")
        print(f"  Profit: {stats['profit']:+.1f} units")
        print(f"  ROI: {roi:+.1f}%")

    print(f"\n  BY TAG:")
    print(f"  {'-'*50}")

    for tag, tag_stats in sorted(stats["by_tag"].items(), key=lambda x: -x[1]["picks"]):
        if tag_stats["picks"] == 0:
            continue
        win_pct = tag_stats["wins"] / tag_stats["picks"] * 100
        roi = tag_stats["profit"] / tag_stats["picks"] * 100
        print(f"  {tag:20} | {tag_stats['picks']:3} picks | {win_pct:5.1f}% win | {roi:+6.1f}% ROI")


def main():
    parser = argparse.ArgumentParser(description="Backtest live predictor with SP odds")
    parser.add_argument("--track", help="Single track name")
    parser.add_argument("--tracks", help="Comma-separated track names")
    parser.add_argument("--race", type=int, help="Single race number (for testing)")
    parser.add_argument("--date", required=True, help="Date (dd-MMM-yyyy)")
    parser.add_argument("--stats-only", action="store_true", help="Just print stats from existing file")

    args = parser.parse_args()

    results_file = get_results_file(args.date)

    # Stats only mode
    if args.stats_only:
        results = load_existing_results(results_file)
        stats = calculate_stats(results)
        print_stats(stats)
        return

    # Single race test
    if args.track and args.race:
        print(f"\n{'='*60}")
        print(f"  TEST: {args.track} R{args.race} - {args.date}")
        print(f"{'='*60}")

        result = run_backtest_race(args.track, args.race, args.date)

        if result.get("error"):
            print(f"\n  Error: {result['error']}")
        else:
            print(f"\n  Distance: {result['distance']}m | Condition: {result['condition']}")
            print(f"  Field size: {result['field_size']}")
            print(f"\n  CONTENDERS:")
            for c in result["contenders"]:
                status = "WON!" if c["won"] else ("Placed" if c["placed"] else f"#{c['position']}")
                star = "⭐" if c["tipsheet_pick"] else "  "
                print(f"    {star} {c['horse']} @ ${c['sp_odds']:.2f} - {c['tag']} - {status}")
            print(f"\n  Summary: {result['summary']}")
        return

    # Full track(s)
    tracks = []
    if args.tracks:
        tracks = [t.strip() for t in args.tracks.split(",")]
    elif args.track:
        tracks = [args.track]
    else:
        print("Error: Specify --track or --tracks")
        return

    for track in tracks:
        run_backtest_track(track, args.date, results_file)

    # Print final stats
    results = load_existing_results(results_file)
    stats = calculate_stats(results)
    print_stats(stats)


if __name__ == "__main__":
    main()
