#!/usr/bin/env python3
"""
Compare live predictor vs venue-adjusted ratings for Randwick 14-Feb-2026.

Usage:
    python experiments/compare_randwick.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.race_data import RaceDataPipeline
from core.predictor import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import anthropic
import json

# Live predictor prompt (unchanged)
LIVE_SYSTEM_PROMPT = SYSTEM_PROMPT

# Venue-adjusted prompt - tells Claude to use Adj column
ADJ_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Focus on **normalized speed ratings**",
    "Focus on the **Adj** column (venue-adjusted speed ratings) instead of raw Rating. "
    "The Adj column adjusts for track quality - runs at weaker tracks get lower adjusted ratings. "
    "Use **Adj** as your primary speed metric"
)


def extract_picks(response_text: str) -> list[str]:
    """Extract horse names from prediction response (JSON format)."""
    import re

    # Find JSON in response
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
        contenders = data.get('contenders', [])
        return [c['horse'] for c in contenders if 'horse' in c]
    except (json.JSONDecodeError, KeyError):
        return []


def run_prediction(race_data, system_prompt: str, include_adj: bool = False) -> tuple[str, list[str]]:
    """Run a prediction and return response + picks."""
    client = anthropic.Anthropic()

    race_text = race_data.to_prompt_text(include_venue_adjusted=include_adj)
    user_prompt = USER_PROMPT_TEMPLATE.format(race_data=race_text)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    text = response.content[0].text
    picks = extract_picks(text)
    return text, picks


def main():
    track = "Randwick"
    date = "14-Feb-2026"

    pipeline = RaceDataPipeline()

    results = []

    print(f"\n{'='*70}")
    print(f"  COMPARING: Live Predictor vs Venue-Adjusted Ratings")
    print(f"  {track} - {date} - Races 1-10")
    print(f"{'='*70}\n")

    for race_num in range(1, 11):
        print(f"\n--- Race {race_num} ---")

        race_data, error = pipeline.get_race_data(track, race_num, date, allow_finished=True)

        if error:
            print(f"  Error: {error}")
            continue

        print(f"  {race_data.race_name} | {race_data.distance}m | {race_data.class_}")

        # Run live predictor (no Adj column)
        print("  Running LIVE predictor...", end=" ", flush=True)
        live_text, live_picks = run_prediction(race_data, LIVE_SYSTEM_PROMPT, include_adj=False)
        print(f"Picks: {', '.join(live_picks) if live_picks else 'None'}")

        # Run venue-adjusted predictor (with Adj column)
        print("  Running ADJUSTED predictor...", end=" ", flush=True)
        adj_text, adj_picks = run_prediction(race_data, ADJ_SYSTEM_PROMPT, include_adj=True)
        print(f"Picks: {', '.join(adj_picks) if adj_picks else 'None'}")

        # Compare
        same = live_picks == adj_picks
        print(f"  SAME: {'✓ Yes' if same else '✗ No - DIFFERENT'}")

        results.append({
            "race": race_num,
            "name": race_data.race_name,
            "class": race_data.class_,
            "live_picks": live_picks,
            "adj_picks": adj_picks,
            "same": same
        })

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    same_count = sum(1 for r in results if r["same"])
    diff_count = len(results) - same_count

    print(f"\n  Total races: {len(results)}")
    print(f"  Same picks: {same_count}")
    print(f"  Different picks: {diff_count}")

    if diff_count > 0:
        print(f"\n  Races with different picks:")
        for r in results:
            if not r["same"]:
                print(f"    R{r['race']} {r['class']}")
                print(f"      Live: {', '.join(r['live_picks'])}")
                print(f"      Adj:  {', '.join(r['adj_picks'])}")

    # Save results
    with open("experiments/randwick_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to experiments/randwick_comparison.json")


if __name__ == "__main__":
    main()
