#!/usr/bin/env python3
"""
Backfill race_class for existing predictions.

Queries PuntingForm API to get the race class for each prediction
that doesn't have one yet.

Usage:
    python scripts/backfill_race_class.py
    python scripts/backfill_race_class.py --dry-run  # Preview changes
"""

import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.puntingform import PuntingFormAPI
from core.tracking import DEFAULT_DB_PATH


def get_unique_races(db_path: Path) -> list[dict]:
    """Get all unique (track, race_number, race_date) combos without race_class."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT track, race_number, race_date
            FROM predictions
            WHERE race_class IS NULL
            ORDER BY race_date DESC, track, race_number
        """).fetchall()
        return [dict(row) for row in rows]


def update_race_class(db_path: Path, track: str, race_number: int, race_date: str, race_class: str) -> int:
    """Update race_class for all predictions matching the race."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            UPDATE predictions
            SET race_class = ?
            WHERE track = ? AND race_number = ? AND race_date = ?
        """, (race_class, track, race_number, race_date))
        conn.commit()
        return cursor.rowcount


def main():
    parser = argparse.ArgumentParser(description="Backfill race_class for existing predictions")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without updating")
    parser.add_argument("--db", type=str, help="Path to predictions database")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    print(f"Using database: {db_path}")

    # Get races needing backfill
    races = get_unique_races(db_path)
    print(f"Found {len(races)} unique races without race_class")

    if not races:
        print("Nothing to backfill!")
        return

    # Initialize API
    pf_api = PuntingFormAPI()

    # Cache for fields data (avoid repeated API calls for same meeting)
    fields_cache: dict[str, dict] = {}

    updated_count = 0
    failed_count = 0

    for race in races:
        track = race['track']
        race_number = race['race_number']
        race_date = race['race_date']

        print(f"\n{track} R{race_number} ({race_date})...", end=" ")

        # Check cache first
        cache_key = f"{track}|{race_date}"

        if cache_key not in fields_cache:
            # Fetch meeting data
            try:
                meetings = pf_api.get_meetings(race_date)
                meeting_id = None
                for m in meetings:
                    if m.get('track', {}).get('name', '').lower() == track.lower():
                        meeting_id = m.get('meetingId')
                        break

                if not meeting_id:
                    print(f"Meeting not found")
                    failed_count += 1
                    continue

                fields = pf_api.get_fields(meeting_id)
                fields_cache[cache_key] = fields
            except Exception as e:
                print(f"API error: {e}")
                failed_count += 1
                continue

        fields = fields_cache[cache_key]

        # Find the race
        race_class = None
        for r in fields.get('races', []):
            if r.get('number') == race_number:
                race_class = r.get('raceClass', '').strip().rstrip(';')
                break

        if not race_class:
            print(f"Race class not found")
            failed_count += 1
            continue

        print(f"-> {race_class}", end="")

        if args.dry_run:
            print(" (dry run)")
        else:
            count = update_race_class(db_path, track, race_number, race_date, race_class)
            print(f" (updated {count} predictions)")
            updated_count += count

    print(f"\n\nSummary:")
    print(f"  Races processed: {len(races)}")
    print(f"  Predictions updated: {updated_count}")
    print(f"  Failed lookups: {failed_count}")


if __name__ == "__main__":
    main()
