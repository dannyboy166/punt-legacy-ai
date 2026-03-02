#!/usr/bin/env python3
"""
Build track ratings from historical form data.

Collects speed ratings for every track by scraping horse form histories
from the last N days of meetings. Each horse's past 10 runs go back
6-12+ months, so we get much more data than just 30 days.

Usage:
    python tools/build_track_ratings.py --days 30
    python tools/build_track_ratings.py --days 30 --output track_ratings_new.csv
"""

import argparse
import csv
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.puntingform import PuntingFormAPI
from core.speed import calculate_run_rating, parse_condition_number


def get_condition_category(condition: str) -> str | None:
    """Map condition to category: good, soft, heavy, or synthetic."""
    if not condition:
        return None

    cond_lower = condition.lower()

    # Synthetic tracks
    if "syn" in cond_lower or "poly" in cond_lower:
        return "synthetic"

    # Parse condition number
    cond_num = parse_condition_number(condition)
    if cond_num is None:
        return None

    if cond_num <= 4:
        return "good"
    elif cond_num <= 6:
        return "soft"
    else:
        return "heavy"


def get_benchmark_number(race_class: str) -> int | None:
    """Extract benchmark number from race class string.

    Examples:
        "Benchmark 58" -> 58
        "Benchmark 72" -> 72
        "Maiden" -> None
    """
    if not race_class:
        return None

    race_class = race_class.strip()

    # Check for "Benchmark XX" format
    if race_class.lower().startswith("benchmark"):
        parts = race_class.split()
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return None

    # Check for "BM58" format
    if race_class.upper().startswith("BM"):
        try:
            return int(race_class[2:])
        except ValueError:
            return None

    return None


def is_valid_benchmark(race_class: str, min_bm: int = 58, max_bm: int = 78) -> bool:
    """Check if race class is a benchmark within the specified range."""
    bm = get_benchmark_number(race_class)
    if bm is None:
        return False
    return min_bm <= bm <= max_bm


def extract_runs_from_form(
    form_data: list[dict],
    min_bm: int | None = None,
    max_bm: int | None = None,
) -> list[dict]:
    """Extract past runs from form data, optionally filtered by benchmark class.

    Args:
        form_data: List of runners with forms
        min_bm: Minimum benchmark (e.g., 58). None = no filter.
        max_bm: Maximum benchmark (e.g., 78). None = no filter.
    """
    runs = []
    filter_by_class = min_bm is not None and max_bm is not None

    for runner in form_data:
        forms = runner.get("forms") or []
        for run in forms:
            # Skip barrier trials
            if run.get("isBarrierTrial"):
                continue

            # Filter by benchmark class if specified
            if filter_by_class:
                race_class = run.get("raceClass", "")
                if not is_valid_benchmark(race_class, min_bm, max_bm):
                    continue

            # Get track name
            track_obj = run.get("track")
            if not track_obj:
                continue
            track_name = track_obj.get("name")
            if not track_name:
                continue

            # Get required fields
            distance = run.get("distance")
            condition = run.get("trackCondition")
            position = run.get("position")

            if not all([distance, condition, position]):
                continue

            # Skip invalid positions (scratched, DNF)
            if position <= 0 or position >= 90:
                continue

            runs.append({
                "track": track_name,
                "distance": distance,
                "condition": condition,
                "run": run,  # Full run data for speed calculation
            })

    return runs


def process_meeting(
    api: PuntingFormAPI,
    meeting: dict,
    min_bm: int | None = None,
    max_bm: int | None = None,
) -> list[dict]:
    """Process a single meeting and return all runs with ratings."""
    meeting_id = meeting.get("meetingId")
    track_info = meeting.get("track", {})
    meeting_track = track_info.get("name", "Unknown")

    try:
        # Get all form data (race 0 = all races)
        form_data = api.get_form(meeting_id, race_number=0, runs=10)
        runs = extract_runs_from_form(form_data, min_bm=min_bm, max_bm=max_bm)

        # Calculate speed rating for each run
        results = []
        for run_info in runs:
            rating = calculate_run_rating(run_info["run"])
            if rating is not None:
                condition_cat = get_condition_category(run_info["condition"])
                if condition_cat:
                    results.append({
                        "track": run_info["track"],
                        "condition_category": condition_cat,
                        "rating": rating,
                        "distance": run_info["distance"],
                    })

        return results

    except Exception as e:
        print(f"  Error processing {meeting_track}: {e}")
        return []


def collect_data(
    days: int,
    max_workers: int = 5,
    min_bm: int | None = None,
    max_bm: int | None = None,
) -> dict:
    """
    Collect speed rating data from the last N days of meetings.

    Args:
        days: Number of days to look back
        max_workers: Parallel workers
        min_bm: Minimum benchmark filter (e.g., 58)
        max_bm: Maximum benchmark filter (e.g., 78)

    Returns:
        Dict: {track: {condition_category: [ratings]}}
    """
    api = PuntingFormAPI()

    # Track data: {track: {condition: [ratings]}}
    track_data = defaultdict(lambda: defaultdict(list))

    # Track unique runs to avoid duplicates (same horse, same race date)
    seen_runs = set()

    total_meetings = 0
    total_runs = 0

    class_filter = ""
    if min_bm and max_bm:
        class_filter = f" (BM{min_bm}-{max_bm} only)"
    print(f"Collecting data{class_filter}...")

    # Process each day
    for day_offset in range(days):
        date = datetime.now() - timedelta(days=day_offset)
        date_str = date.strftime("%d-%b-%Y")

        print(f"\n[{day_offset + 1}/{days}] Fetching meetings for {date_str}...")

        try:
            meetings = api.get_meetings(date_str)
            if not meetings:
                print(f"  No meetings found")
                continue

            # Filter to Australian meetings only (surface not always populated on past dates)
            aus_meetings = [
                m for m in meetings
                if m.get("track", {}).get("country") == "AUS"
            ]

            print(f"  Found {len(aus_meetings)} Australian meetings")
            total_meetings += len(aus_meetings)

            # Process meetings in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_meeting, api, meeting, min_bm, max_bm): meeting
                    for meeting in aus_meetings
                }

                for future in as_completed(futures):
                    meeting = futures[future]
                    track_name = meeting.get("track", {}).get("name", "Unknown")

                    try:
                        results = future.result()
                        new_runs = 0

                        for result in results:
                            # Create unique key for this run
                            run_key = (
                                result["track"],
                                result["condition_category"],
                                round(result["rating"], 6),
                                result["distance"],
                            )

                            if run_key not in seen_runs:
                                seen_runs.add(run_key)
                                track_data[result["track"]][result["condition_category"]].append(
                                    result["rating"]
                                )
                                new_runs += 1

                        if new_runs > 0:
                            total_runs += new_runs
                            print(f"  {track_name}: +{new_runs} runs")

                    except Exception as e:
                        print(f"  {track_name}: Error - {e}")

        except Exception as e:
            print(f"  Error fetching meetings: {e}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_meetings} meetings, {total_runs} unique runs")
    print(f"TRACKS: {len(track_data)} unique tracks")
    print(f"{'='*60}")

    return dict(track_data)


def calculate_track_ratings(track_data: dict) -> list[dict]:
    """
    Calculate average ratings per track and condition.

    Returns:
        List of dicts with track rating data
    """
    results = []

    for track, conditions in sorted(track_data.items()):
        row = {
            "venue": track,
            "good_rating": None,
            "good_samples": 0,
            "soft_rating": None,
            "soft_samples": 0,
            "heavy_rating": None,
            "heavy_samples": 0,
            "synthetic_rating": None,
            "synthetic_samples": 0,
        }

        all_ratings = []
        condition_count = 0

        for condition, ratings in conditions.items():
            if ratings:
                avg = sum(ratings) / len(ratings)
                row[f"{condition}_rating"] = round(avg, 6)
                row[f"{condition}_samples"] = len(ratings)
                all_ratings.extend(ratings)
                condition_count += 1

        # Calculate overall rating
        if all_ratings:
            row["overall_track_rating"] = round(sum(all_ratings) / len(all_ratings), 6)
            row["total_sample_size"] = len(all_ratings)
            row["num_conditions"] = condition_count
            results.append(row)

    # Sort by overall rating descending
    results.sort(key=lambda x: x.get("overall_track_rating", 0), reverse=True)

    return results


def write_csv(results: list[dict], output_path: str):
    """Write results to CSV file."""
    fieldnames = [
        "venue",
        "good_rating", "good_samples",
        "soft_rating", "soft_samples",
        "heavy_rating", "heavy_samples",
        "synthetic_rating", "synthetic_samples",
        "overall_track_rating", "total_sample_size", "num_conditions",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            # Convert None to empty string for CSV
            csv_row = {k: (v if v is not None else "") for k, v in row.items()}
            writer.writerow(csv_row)

    print(f"\nWritten to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build track ratings from historical form data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="core/normalization/track_ratings.csv",
        help="Output CSV path (default: core/normalization/track_ratings.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )
    parser.add_argument(
        "--min-bm",
        type=int,
        default=None,
        help="Minimum benchmark class to include (e.g., 58)",
    )
    parser.add_argument(
        "--max-bm",
        type=int,
        default=None,
        help="Maximum benchmark class to include (e.g., 78)",
    )

    args = parser.parse_args()

    print(f"Building track ratings from last {args.days} days...")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    if args.min_bm and args.max_bm:
        print(f"Class filter: BM{args.min_bm}-{args.max_bm}")

    # Collect data
    track_data = collect_data(
        days=args.days,
        max_workers=args.workers,
        min_bm=args.min_bm,
        max_bm=args.max_bm,
    )

    if not track_data:
        print("No data collected!")
        return 1

    # Calculate ratings
    results = calculate_track_ratings(track_data)

    # Print summary
    print(f"\n{'='*60}")
    print("TOP 10 FASTEST TRACKS:")
    print(f"{'='*60}")
    for row in results[:10]:
        print(f"  {row['venue']:25} {row['overall_track_rating']:.4f} ({row['total_sample_size']} samples)")

    print(f"\n{'='*60}")
    print("TOP 10 SLOWEST TRACKS:")
    print(f"{'='*60}")
    for row in results[-10:]:
        print(f"  {row['venue']:25} {row['overall_track_rating']:.4f} ({row['total_sample_size']} samples)")

    # Write CSV
    write_csv(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
