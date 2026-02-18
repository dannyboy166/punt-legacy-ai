#!/usr/bin/env python3
"""
Compare predictions using standard Rating vs venue-Adjusted Rating.

Usage:
    python experiments/compare_ratings.py "Dubbo" 3 "14-Feb-2026"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.race_data import RaceDataPipeline
from core.predictor import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import anthropic

# Use exact live prompt - only difference is which rating column to use
RATING_SYSTEM_PROMPT = SYSTEM_PROMPT  # Unchanged - uses "Rating" column

ADJ_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Focus on **normalized speed ratings**",
    "Focus on the **Adj** column (venue-adjusted speed ratings) instead of raw Rating. The Adj column adjusts for track quality - runs at weaker tracks get lower adjusted ratings. Use **Adj** as your primary speed metric. Focus on **these venue-adjusted speed ratings**"
)


def run_prediction(race_data_text: str, system_prompt: str, label: str):
    """Run a prediction with the given prompt."""
    client = anthropic.Anthropic()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")

    user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_data_text)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    print(response.content[0].text)
    return response.content[0].text


def main():
    if len(sys.argv) < 4:
        print("Usage: python experiments/compare_ratings.py <track> <race_number> <date>")
        print("Example: python experiments/compare_ratings.py 'Dubbo' 3 '14-Feb-2026'")
        sys.exit(1)

    track = sys.argv[1]
    race_number = int(sys.argv[2])
    date = sys.argv[3]

    print(f"\nFetching data for {track} R{race_number} on {date}...")

    pipeline = RaceDataPipeline()
    race_data, error = pipeline.get_race_data(track, race_number, date, allow_finished=True)

    if error:
        print(f"Error: {error}")
        sys.exit(1)

    race_text = race_data.to_prompt_text()

    # Show the form table for one runner so we can see Rating vs Adj
    print("\n" + "="*60)
    print("  SAMPLE FORM DATA (first runner with form)")
    print("="*60)
    for r in race_data.runners:
        if r.form:
            print(f"\n{r.name}:")
            print("| Date | Track | Rating | Adj |")
            print("|------|-------|--------|-----|")
            for f in r.form[:5]:
                rating_str = f"{f.rating * 100:.1f}" if f.rating else "N/A"
                adj_str = f"{f.rating_venue_adjusted * 100:.1f}" if f.rating_venue_adjusted else "-"
                print(f"| {f.date} | {f.track[:12]} | {rating_str} | {adj_str} |")
            break

    # Run both predictions
    run_prediction(race_text, RATING_SYSTEM_PROMPT, "USING RAW RATING (live prompt)")
    run_prediction(race_text, ADJ_SYSTEM_PROMPT, "USING VENUE-ADJUSTED RATING")

    print("\n" + "="*60)
    print("  Compare the picks above to see if Adj changes anything")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
