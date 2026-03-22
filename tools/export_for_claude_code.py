#!/usr/bin/env python3
"""
Export race data for Claude Code analysis.

This outputs the EXACT same data that the Claude API predictor sees,
so you can paste it to Claude Code for free analysis.

Usage:
    python3 tools/export_for_claude_code.py "Randwick" 5 22-Mar-2026        # Single race
    python3 tools/export_for_claude_code.py "Randwick" 3-7 22-Mar-2026      # Races 3-7
    python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026      # All races

Then paste the output to Claude Code and ask for analysis.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.race_data import RaceDataPipeline
from api.puntingform import PuntingFormAPI


# =============================================================================
# ANALYSIS INSTRUCTIONS (same as predictor SYSTEM_PROMPT)
# =============================================================================

ANALYSIS_INSTRUCTIONS = """
## ANALYSIS INSTRUCTIONS

You are an expert horse racing analyst. Analyze the race data below and pick 0-3 contenders.

**CRITICAL: Ratings first, odds second. Do NOT look at odds until Step 3.**

---

### Step 1: Rate each horse

For each runner, find their Adj ratings at similar distance AND condition to today's race.

**Important:**
- More recent runs are **a lot** more relevant than older runs
- TRIAL runs don't count

### Step 2: Rank the field by recent Adj

Sort runners from highest recent Adj to lowest. This is your form ranking - the horse with the best recent ratings should be near the top.

### Step 3: NOW look at odds - find value

Compare your form ranking to the market:
- Is the top-rated horse a short price? → May still be the pick if clear standout
- Is a horse with similar ratings at much bigger odds? → VALUE BET
- Is the favourite's recent form actually weaker than a longer-priced runner? → Favourite is opposable

### Step 4: Assign picks

**"The one to beat" ⭐** = Highest recent Adj AND you would bet on it at those odds. Only star clear standouts.

**"Value bet"** = Similar recent Adj to top picks but at 2-3x longer odds.

**"Each-way chance"** = Strong place chance based on ratings, good each-way odds.

---

### Understanding the ratings
- **Adj column** = speed rating normalized by distance, condition, AND track quality (use this column)
- **Rating column** = normalized by distance + condition only (Adj is more accurate)
- **100 = expected speed** for that distance/condition
- **Ignore finishing position** - higher Adj is better regardless of where they finished
- **Ignore margin** - already baked into the rating
- **⚠️eased** = stewards noted horse wasn't fully pushed, actual ability likely higher

### Other notes
- **Notes column** = official stewards report. Use to explain why a rating may be lower than ability.
- 0 race runs = UNKNOWN (first starter)
- First-up/Second-up: Check Prep column (1 = first-up, 2 = second-up). Check their career record at that prep stage.
- If 50%+ have no race form → too many unknowns, consider skipping

### Common Mistakes to Avoid
1. ❌ Trusting the market - short odds doesn't mean best recent form
2. ❌ Cherry-picking old ratings to justify a favourite
3. ❌ Looking at best-ever rating instead of RECENT ratings
4. ❌ Including runs at irrelevant distances or conditions
5. ❌ Weighting old runs too heavily
6. ❌ Caring about finishing position - only Adj matters

---
"""


def log(msg: str) -> None:
    """Print status message to stderr (keeps stdout clean for output)."""
    print(msg, file=sys.stderr)


def export_race(
    track: str,
    race_number: int,
    date: str,
    pipeline: RaceDataPipeline,
) -> tuple[str, bool]:
    """
    Export a single race's data as formatted text.

    Returns:
        Tuple of (output_text, success)
    """
    race_data, error = pipeline.get_race_data(
        track,
        race_number,
        date,
        allow_finished=True,  # Allow past races for analysis
    )

    if error:
        return f"# {track} R{race_number} - ERROR\n{error}\n", False

    if not race_data:
        return f"# {track} R{race_number} - No data available\n", False

    # Get the EXACT same text that Claude API would see
    # include_venue_adjusted=True to show both Rating and Adj columns
    prompt_text = race_data.to_prompt_text(include_venue_adjusted=True)

    return prompt_text, True


def get_race_numbers(
    track: str,
    date: str,
    race_spec: str,
    pf_api: PuntingFormAPI,
) -> list[int]:
    """
    Parse race specification and return list of race numbers.

    Args:
        track: Track name
        date: Date in PuntingForm format (dd-MMM-yyyy)
        race_spec: One of:
            - Single number: "5"
            - Range: "3-7"
            - All: "all"

    Returns:
        List of race numbers to export
    """
    if race_spec.lower() == "all":
        # Get all races for this meeting
        try:
            meetings = pf_api.get_meetings(date)
        except Exception as e:
            log(f"Error fetching meetings: {e}")
            return []

        for m in meetings:
            m_track = m.get("track", {}).get("name", "")
            if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                meeting_id = m.get("meetingId")
                try:
                    fields = pf_api.get_fields(meeting_id, 0)
                except Exception as e:
                    log(f"Error fetching fields: {e}")
                    return []
                races = fields.get("races", [])
                return sorted([r.get("number") for r in races if r.get("number")])

        log(f"Track '{track}' not found in meetings for {date}")
        return []

    elif "-" in race_spec:
        # Range like "3-7"
        try:
            start, end = race_spec.split("-")
            return list(range(int(start), int(end) + 1))
        except ValueError:
            log(f"Invalid range format: {race_spec}")
            return []

    else:
        # Single race number
        try:
            return [int(race_spec)]
        except ValueError:
            log(f"Invalid race number: {race_spec}")
            return []


def main():
    if len(sys.argv) < 4 or "--help" in sys.argv or "-h" in sys.argv:
        print("""
Export Race Data for Claude Code Analysis
==========================================

Outputs the EXACT same data that the Claude API predictor sees.
Paste the output to Claude Code for free analysis!

Usage:
    python3 tools/export_for_claude_code.py <track> <race(s)> <date> [options]

Arguments:
    track     Track name (e.g., "Randwick", "Morphettville Parks")
    race(s)   Race number(s):
              - Single: 5
              - Range: 3-7
              - All: all
    date      Date in format dd-MMM-yyyy (e.g., 22-Mar-2026)

Options:
    --no-instructions    Skip the analysis instructions header
                         (useful if pasting multiple batches)

Examples:
    python3 tools/export_for_claude_code.py "Randwick" 5 22-Mar-2026
    python3 tools/export_for_claude_code.py "Randwick" 3-7 22-Mar-2026
    python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026
    python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026 --no-instructions

Output:
    The script outputs clean prompt text to stdout (for copying).
    Progress messages go to stderr.

    To save to file and copy:
        python3 tools/export_for_claude_code.py "Randwick" all 22-Mar-2026 > races.txt
        cat races.txt | pbcopy  # Copy to clipboard (macOS)
""")
        sys.exit(1)

    track = sys.argv[1]
    race_spec = sys.argv[2]
    date = sys.argv[3]
    include_instructions = "--no-instructions" not in sys.argv

    log(f"Fetching data for {track} on {date}...")
    log("")

    # Initialize APIs
    try:
        pipeline = RaceDataPipeline()
        pf_api = PuntingFormAPI()
    except Exception as e:
        log(f"Error initializing APIs: {e}")
        log("Make sure PUNTINGFORM_API_KEY is set in your environment or .env file")
        sys.exit(1)

    # Get race numbers to export
    race_numbers = get_race_numbers(track, date, race_spec, pf_api)

    if not race_numbers:
        log(f"No races found for '{track}' on {date}")
        sys.exit(1)

    log(f"Exporting {len(race_numbers)} race(s): R{', R'.join(map(str, race_numbers))}")
    log("")

    # Export each race
    all_output = []
    success_count = 0
    error_count = 0

    for race_num in race_numbers:
        log(f"  R{race_num}...")
        output, success = export_race(track, race_num, date, pipeline)
        all_output.append(output)

        if success:
            success_count += 1
        else:
            error_count += 1
            log(f"    ^ Error (see output)")

    # Print final output (clean, to stdout)
    # Start with analysis instructions so Claude Code knows how to analyze
    if include_instructions:
        print(ANALYSIS_INSTRUCTIONS)
    print("\n---\n".join(all_output))

    # Summary
    log("")
    log("=" * 60)
    log(f"DONE! Exported {success_count} race(s)" + (f", {error_count} error(s)" if error_count else ""))
    log("")
    log("Copy the output above and paste to Claude Code.")
    log("Ask: 'Analyze these races and pick your best bets'")
    log("=" * 60)


if __name__ == "__main__":
    main()
