"""
A/B Test Framework for AI Predictor Variations.

Tests different prompt/data variations on historical races to find
what performs best.

Usage:
    # Test on 5 sample races (quick test)
    python experiments/ab_test.py --limit 5

    # Test on all races from date range
    python experiments/ab_test.py --start 16-Mar-2026 --end 13-Apr-2026

    # Test specific variation only
    python experiments/ab_test.py --variation v2 --limit 5

Output saved to: data/ab_results/ab_test_TIMESTAMP.json
"""

import sys
sys.path.insert(0, '/Users/danielsamus/punt-legacy-ai')

import os
import json
import re
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import anthropic
from dotenv import load_dotenv

from api.puntingform import PuntingFormAPI
from core.speed import calculate_run_rating
from core.race_data import get_track_rating

load_dotenv()

# =============================================================================
# VARIATIONS - Each variation modifies what data/prompt Claude sees
# =============================================================================

VARIATIONS = {
    "v1_baseline": {
        "name": "V1: Baseline (Current Live)",
        "description": "Current live predictor with both Rating and Adj columns",
        "show_rating": True,
        "show_adj": True,
        "show_pos_margin": True,
        "show_trainer_ae": True,
        "prompt_style": "full",
    },
    "v2_adj_only": {
        "name": "V2: Adj Only",
        "description": "Remove Rating column, keep only venue-adjusted Adj",
        "show_rating": False,
        "show_adj": True,
        "show_pos_margin": True,
        "show_trainer_ae": True,
        "prompt_style": "full",
    },
    "v3_no_pos_margin": {
        "name": "V3: No Pos/Margin",
        "description": "Remove Pos and Margin columns (ratings capture this)",
        "show_rating": True,
        "show_adj": True,
        "show_pos_margin": False,
        "show_trainer_ae": True,
        "prompt_style": "full",
    },
    "v4_no_trainer_ae": {
        "name": "V4: No Trainer A/E",
        "description": "Remove trainer A/E (keep jockey A/E)",
        "show_rating": True,
        "show_adj": True,
        "show_pos_margin": True,
        "show_trainer_ae": False,
        "prompt_style": "full",
    },
    "v5_minimal_prompt": {
        "name": "V5: Minimal Prompt",
        "description": "Shorter prompt, let Claude figure it out",
        "show_rating": False,
        "show_adj": True,
        "show_pos_margin": True,
        "show_trainer_ae": False,
        "prompt_style": "minimal",
    },
    "v6_lean": {
        "name": "V6: Lean",
        "description": "Only Adj + Jockey A/E, no pos/margin, no trainer, full prompt",
        "show_rating": False,
        "show_adj": True,
        "show_pos_margin": False,
        "show_trainer_ae": False,
        "prompt_style": "full",
    },
    "v7_lean_minimal": {
        "name": "V7: Lean + Minimal",
        "description": "Only Adj + Jockey A/E, no pos/margin, minimal prompt",
        "show_rating": False,
        "show_adj": True,
        "show_pos_margin": False,
        "show_trainer_ae": False,
        "prompt_style": "minimal",
    },
}

# Full prompt (current live)
FULL_SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout on ratings at similar DISTANCE and CONDITIONS vs the field
- **"Each-way chance"** - Good ratings at similar DISTANCE and CONDITIONS vs the field, place odds $1.80+
- **"Value bet"** - Odds $5.00+ AND ratings that are competitive with the top of the field (not just any longshot)

**Pick 0 contenders (no bet) when:**
- A lot of field has no race form (only trials) - you can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

## Understanding the Data

### Speed Ratings
Ratings are normalized to 100 = benchmark performance. Higher = faster.
- Use the CStep column to find runs at similar conditions - these are most predictive

### Form Table Columns
| Column | Meaning |
|--------|---------|
| Dist | Race distance in metres |
| Cond | Track condition (G4=Good4, S5=Soft5, H8=Heavy8, etc.) |
| Pos | Finish position / field size |
| Margin | Lengths behind winner (0L for winner) |
| Dist% | Distance difference from TODAY's race (+8% = 8% longer, = means same) |
| CStep | Condition steps from TODAY's track (0=same, -2=drier, +2=wetter) |
| WtCh | Weight change vs that run. Negative = less weight (easier). |
| Rating | Speed rating normalized by distance + condition (100=par) |
| Adj | Rating further normalized by track quality. **USE THIS** |
| Prep | Run number in current prep (1=first-up, 2=second-up, etc.) |
| Trial | "TRIAL" if barrier trial (not a real race) |

### Key Analysis
- **Jockey A/E > 1.0 is positive. A/E < 0.85 is a red flag**
- **Trials don't count** - horses don't try
- **50%+ unknowns = skip race**

## Output

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "tag": "The one to beat" | "Each-way chance" | "Value bet",
      "analysis": "2-3 sentences referencing form",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "Brief overview or reason for 0 picks"
}
```

**tipsheet_pick = true** when you would genuinely bet on this horse."""


# Minimal prompt (V5)
MINIMAL_SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders. Tags: "The one to beat", "Each-way chance", "Value bet" ($5+).

Use the **Adj** column (venue-adjusted speed ratings). Higher = faster. 100 = par.
Compare ratings at similar distance (Dist%) and conditions (CStep).
Trials don't count. Skip if 50%+ have no form.

```json
{
  "contenders": [{"horse": "Name", "tab_no": 1, "odds": 3.50, "tag": "...", "analysis": "...", "tipsheet_pick": true}],
  "summary": "..."
}
```"""


def get_system_prompt(variation: dict) -> str:
    """Get system prompt for variation."""
    if variation["prompt_style"] == "minimal":
        return MINIMAL_SYSTEM_PROMPT
    return FULL_SYSTEM_PROMPT


# =============================================================================
# DATA GENERATION - Creates prompt text based on variation settings
# =============================================================================

def generate_prompt_for_variation(
    track: str,
    race_number: int,
    date: str,
    variation: dict
) -> tuple[Optional[str], Optional[list], Optional[dict]]:
    """
    Generate race prompt text customized for a specific variation.

    Returns: (prompt_text, runners_data, race_info) or (None, error_msg, None)
    """
    api = PuntingFormAPI()

    # Get meeting
    meetings = api.get_meetings(date)
    meeting = next((m for m in meetings if m.get('track', {}).get('name', '').lower() == track.lower()), None)

    if not meeting:
        return None, f"Track '{track}' not found", None

    meeting_id = meeting.get('meetingId')
    fields_data = api.get_fields(meeting_id, race_number)
    form_data = api.get_form(meeting_id, race_number, runs=10)
    speedmap_data = api.get_speedmaps(meeting_id, race_number)

    races = fields_data.get('races', [])
    if not races:
        return None, "No race data", None

    race = races[0]
    distance = race.get('distance', 0)
    condition = race.get('trackCondition')

    # For historical races, fields endpoint may not have condition
    # Try results endpoint instead
    if not condition:
        try:
            results_data = api.get_results(meeting_id, race_number)
            if isinstance(results_data, list) and results_data:
                race_results = results_data[0].get('raceResults', [])
                for rr in race_results:
                    if rr.get('raceNumber') == race_number:
                        cond_label = rr.get('trackConditionLabel', '')
                        cond_num = rr.get('trackCondition')
                        if cond_label and cond_num:
                            condition = f"{cond_label[0].upper()}{cond_num}"
                        elif cond_num:
                            condition = f"G{cond_num}"  # Default to Good if no label
                        distance = rr.get('distance', distance)
                        break
        except Exception:
            pass

    if not condition:
        return None, "Track condition not available", None

    # Parse condition number
    cond_num = None
    if condition and len(condition) >= 2:
        try:
            cond_num = int(condition[1:])
        except ValueError:
            pass

    race_name = race.get('raceName', '')
    race_class = race.get('raceClass', '')

    form_by_id = {r.get('runnerId'): r.get('forms', []) for r in form_data}

    # Index speedmap by tab number
    speedmap_by_tab = {}
    if isinstance(speedmap_data, list):
        for sm_entry in speedmap_data:
            for item in sm_entry.get('items', []):
                tab_no = item.get('tabNo')
                if tab_no is not None:
                    speedmap_by_tab[tab_no] = item

    race_run_counts = []
    runners_data = []

    # Build prompt lines
    lines = [
        f"# {track} Race {race_number}: {race_name}",
        f"Distance: {distance}m | Condition: {condition} | Class: {race_class}",
        "",
        "## Runners",
        ""
    ]

    for r in sorted(race.get('runners', []), key=lambda x: x.get('tabNo', 0)):
        # Skip scratched
        if r.get('scratched'):
            continue

        jockey_info = r.get('jockey', {})
        jockey_name = jockey_info.get('fullName', '') if isinstance(jockey_info, dict) else ''
        if not jockey_name.strip():
            continue

        sp = r.get('priceSP', 0) or 0
        if sp == 0:
            continue

        tab = r.get('tabNo', 0)
        name = r.get('name', 'Unknown')
        barrier = r.get('barrier', 0)
        weight = r.get('weight', 0)
        place_odds = round(1 + (sp - 1) * 0.35, 2) if sp > 1 else 1.10

        jockey = jockey_name
        trainer = r.get('trainer', {}).get('fullName', '')
        jockey_ae = r.get('jockeyA2E_Last100', {}).get('a2E')
        trainer_ae = r.get('trainerA2E_Last100', {}).get('a2E')

        career = f"{r.get('careerStarts', 0)}: {r.get('careerWins', 0)}-{r.get('careerSeconds', 0)}-{r.get('careerThirds', 0)}"
        win_pct = r.get('winPct', 0)

        runners_data.append({'tab': tab, 'name': name, 'sp': sp})

        lines.append(f"### {tab}. {name}")
        implied_prob = (100 / sp) if sp > 0 else 0
        lines.append(f"Odds: ${sp:.2f} win / ${place_odds:.2f} place → {implied_prob:.1f}% implied")

        # Jockey line
        jockey_ae_str = f"{jockey_ae:.2f}" if jockey_ae else "N/A"
        lines.append(f"Jockey: {jockey} (A/E: {jockey_ae_str})")

        # Trainer line - conditionally include A/E based on variation
        if variation["show_trainer_ae"]:
            trainer_ae_str = f"{trainer_ae:.2f}" if trainer_ae else "N/A"
            lines.append(f"Trainer: {trainer} (A/E: {trainer_ae_str})")
        else:
            lines.append(f"Trainer: {trainer}")

        lines.append(f"Career: {career} ({win_pct:.0f}% win)")
        lines.append(f"Barrier: {barrier} | Weight: {weight}kg")

        # Form
        runner_id = r.get('runnerId')
        forms = form_by_id.get(runner_id, [])
        race_runs = [f for f in forms if not f.get('isBarrierTrial')]
        trial_runs = [f for f in forms if f.get('isBarrierTrial')]
        race_run_counts.append(len(race_runs))

        # Speedmap
        sm = speedmap_by_tab.get(tab)
        if sm:
            speed_rank = sm.get('speed')
            settle_pos = sm.get('settle')
            if speed_rank is not None:
                lines.append(f"Speed Rank: {speed_rank} | Settles: {settle_pos}")

        if forms:
            form_summary = f"Form: {len(race_runs)} race runs"
            if trial_runs:
                form_summary += f", {len(trial_runs)} trials"
            if len(race_runs) < 3:
                form_summary += " ⚠️ LIMITED"
            lines.append(form_summary)
            lines.append("")

            # Build header based on variation
            header_cols = ["Date", "Track", "Dist", "Cond"]
            if variation["show_pos_margin"]:
                header_cols.extend(["Pos", "Margin"])
            header_cols.extend(["Dist%", "CStep", "WtCh"])
            if variation["show_rating"]:
                header_cols.append("Rating")
            if variation["show_adj"]:
                header_cols.append("Adj")
            header_cols.extend(["Prep", "Trial"])

            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("|" + "|".join(["------"] * len(header_cols)) + "|")

            for f in forms[:10]:
                # Parse date
                f_date_raw = f.get('meetingDate', '')
                try:
                    dt = datetime.fromisoformat(f_date_raw.replace("Z", "+00:00"))
                    f_date = dt.strftime("%d-%b")
                except:
                    f_date = f_date_raw[:10]

                f_track = f.get('track', {}).get('name', '')[:10]
                f_dist = f.get('distance', 0)
                f_cond = f.get('trackCondition', '')
                f_pos = f.get('position', 0)
                f_starters = f.get('starters', 0)
                f_margin = f.get('margin', 0)
                is_trial = f.get('isBarrierTrial', False)
                f_prep = f.get('prepRuns')
                if f_prep is not None:
                    f_prep = f_prep + 1

                # Calculate rating
                rating = calculate_run_rating(f)
                rating_str = f"{rating * 100:.1f}" if rating else "N/A"

                # Calculate venue-adjusted rating
                adj_str = "-"
                if rating:
                    track_rating = get_track_rating(f_track)
                    if track_rating:
                        adj_rating = rating * 100 / track_rating * 100
                        adj_str = f"{adj_rating:.1f}"

                prep_str = str(f_prep) if f_prep else "-"
                trial_str = "TRIAL" if is_trial else "-"
                margin_str = f"{f_margin}L"

                # Relevance columns
                if distance and f_dist:
                    dist_diff = ((f_dist - distance) / distance) * 100
                    dist_str = f"{dist_diff:+.0f}%" if abs(dist_diff) >= 1 else "="
                else:
                    dist_str = "?"

                f_cond_num = None
                if f_cond and len(f_cond) >= 2:
                    try:
                        f_cond_num = int(f_cond[1:])
                    except ValueError:
                        pass

                if cond_num and f_cond_num:
                    cond_diff = f_cond_num - cond_num
                    cond_str_diff = f"{cond_diff:+d}" if cond_diff != 0 else "="
                else:
                    cond_str_diff = "?"

                f_weight = f.get('weight', 0)
                if weight and f_weight:
                    wt_diff = weight - f_weight
                    wt_str = f"{wt_diff:+.1f}" if abs(wt_diff) >= 0.1 else "="
                else:
                    wt_str = "?"

                # Build row based on variation
                row_cols = [f_date, f_track, f"{f_dist}m", f_cond]
                if variation["show_pos_margin"]:
                    row_cols.extend([f"{f_pos}/{f_starters}", margin_str])
                row_cols.extend([dist_str, cond_str_diff, wt_str])
                if variation["show_rating"]:
                    row_cols.append(rating_str)
                if variation["show_adj"]:
                    row_cols.append(adj_str)
                row_cols.extend([prep_str, trial_str])

                lines.append("| " + " | ".join(row_cols) + " |")
        else:
            lines.append("⚠️ NO FORM AVAILABLE - first starter")

        lines.append("")

    # Field summary
    horses_with_form = sum(1 for c in race_run_counts if c > 0)
    total_horses = len(race_run_counts)

    # Calculate pace
    leaders = sum(1 for tab_no, sm in speedmap_by_tab.items() if sm.get('speed') and sm['speed'] <= 2)
    if leaders >= 3:
        pace = "hot"
    elif leaders <= 1:
        pace = "soft"
    else:
        pace = "moderate"

    lines[1] = f"Distance: {distance}m | Condition: {condition} | Class: {race_class} | Field: {total_horses} | Pace: {pace}"

    warning = f"Form status: {horses_with_form}/{total_horses} have race form"
    if horses_with_form < total_horses / 2:
        warning += " ⚠️ MAJORITY UNPROVEN"
    lines.insert(3, warning)

    race_info = {
        'distance': distance,
        'condition': condition,
        'name': race_name,
        'class': race_class,
        'with_form': horses_with_form,
        'total': total_horses
    }

    return "\n".join(lines), runners_data, race_info


def run_prediction(prompt_text: str, system_prompt: str) -> str:
    """Run Claude prediction."""
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.2,
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


def get_full_race_results(track: str, race_number: int, date: str) -> dict:
    """
    Get actual finishing positions for ALL runners in a race.

    Returns: {horse_name_lower: {position, won, placed, sp}}
    """
    api = PuntingFormAPI()

    meetings = api.get_meetings(date)
    meeting = next((m for m in meetings if m.get('track', {}).get('name', '').lower() == track.lower()), None)

    if not meeting:
        return {}

    meeting_id = meeting.get('meetingId')
    form_data = api.get_form(meeting_id, race_number, runs=1)

    results = {}
    for runner in form_data:
        name = runner.get('name', '').lower()
        position = runner.get('position', 99)
        sp = runner.get('priceSP', 0) or 0

        # Skip scratched (position=0 or SP=0)
        if position == 0 or sp == 0:
            continue

        results[name] = {
            'position': position,
            'won': position == 1,
            'placed': position <= 3,
            'sp': sp
        }

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_ab_test(
    races: list[dict],
    variations_to_test: list[str] = None,
    verbose: bool = True,
    output_file: str = None,
    existing_results: dict = None
) -> dict:
    """
    Run A/B test on a list of races.

    Args:
        races: List of {track, race_number, date, actual_results}
        variations_to_test: List of variation keys to test (default: all)
        verbose: Print progress
        output_file: If provided, save incrementally after each race
        existing_results: If resuming, pass the loaded results to continue from

    Returns:
        Full results dict
    """
    if variations_to_test is None:
        variations_to_test = list(VARIATIONS.keys())

    # If resuming, use existing results; otherwise start fresh
    if existing_results:
        results = existing_results
        results["races_tested"] = len(results["races"]) + len(races)
    else:
        results = {
            "test_time": datetime.now().isoformat(),
            "variations_tested": variations_to_test,
            "races_tested": len(races),
            "races": []
        }

    for i, race in enumerate(races):
        track = race['track']
        race_number = race['race_number']
        date = race['date']

        if verbose:
            print(f"\n{'='*60}")
            print(f"  [{i+1}/{len(races)}] {track} R{race_number} ({date})")
            print(f"{'='*60}")

        # Fetch FULL race results (positions for ALL runners)
        full_results = get_full_race_results(track, race_number, date)
        if verbose and full_results:
            winner = [h for h, r in full_results.items() if r['won']]
            print(f"  Winner: {winner[0].title() if winner else 'Unknown'}")

        race_result = {
            "track": track,
            "race_number": race_number,
            "date": date,
            "full_results": full_results,  # All runners' positions
            "variations": {}
        }

        for var_key in variations_to_test:
            variation = VARIATIONS[var_key]

            if verbose:
                print(f"\n  Testing {variation['name']}...")

            # Generate prompt for this variation
            prompt_text, runners, race_info = generate_prompt_for_variation(
                track, race_number, date, variation
            )

            if prompt_text is None:
                if verbose:
                    print(f"    ❌ Error: {runners}")
                race_result["variations"][var_key] = {
                    "error": runners,
                    "contenders": []
                }
                continue

            # Run prediction
            system_prompt = get_system_prompt(variation)
            raw_response = run_prediction(prompt_text, system_prompt)
            parsed = parse_prediction(raw_response)

            contenders = parsed.get("contenders", [])
            summary = parsed.get("summary", "")

            # Match contenders to FULL race results (all runners)
            for c in contenders:
                horse_name = c.get("horse", "").lower().strip()
                if horse_name in full_results:
                    actual = full_results[horse_name]
                    c["actual_position"] = actual["position"]
                    c["actual_won"] = actual["won"]
                    c["actual_placed"] = actual["placed"]
                else:
                    # Try partial match
                    for h_name, actual in full_results.items():
                        if horse_name in h_name or h_name in horse_name:
                            c["actual_position"] = actual["position"]
                            c["actual_won"] = actual["won"]
                            c["actual_placed"] = actual["placed"]
                            break

            race_result["variations"][var_key] = {
                "contenders": contenders,
                "summary": summary,
                "prompt_preview": prompt_text[:500] + "..."  # Save preview for debugging
            }

            if verbose:
                if not contenders:
                    print(f"    → 0 picks: {summary[:50]}...")
                else:
                    for c in contenders:
                        won = c.get("actual_won", False)
                        placed = c.get("actual_placed", False)
                        pos = c.get("actual_position", "?")
                        result = "✅ WON" if won else ("📍 PLACED" if placed else f"❌ pos {pos}")
                        print(f"    → {c['horse']} ${c['odds']:.2f} [{c['tag']}] - {result}")

        results["races"].append(race_result)

        # Incremental save after each race
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"\n  💾 Saved progress ({len(results['races'])}/{len(races)} races)")

    return results


def get_sample_races(limit: int = 5, exclude_keys: set = None) -> list[dict]:
    """Get sample races from Railway API, excluding already-tested races."""
    import urllib.request
    import random

    url = "https://punt-legacy-ai-production.up.railway.app/predictions/recent?limit=5000"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

    # Filter to Mar 16 - Apr 13 with outcomes
    start = datetime(2026, 3, 16)
    end = datetime(2026, 4, 13)

    def parse_date(d):
        return datetime.strptime(d, '%d-%b-%Y')

    filtered = [p for p in data if start <= parse_date(p['race_date']) <= end and p['outcome_recorded'] == 1]

    # Group by race
    races = {}
    for p in filtered:
        key = (p['track'], p['race_number'], p['race_date'])
        if key not in races:
            races[key] = {
                'track': p['track'],
                'race_number': p['race_number'],
                'date': p['race_date'],
                'actual_results': []
            }
        races[key]['actual_results'].append({
            'horse': p['horse'],
            'tag': p['tag'],
            'odds': p['odds'],
            'won': p['won'],
            'placed': p['placed'],
            'finishing_position': p['finishing_position']
        })

    # Exclude already-tested races
    race_list = list(races.values())
    if exclude_keys:
        race_list = [r for r in race_list if f"{r['track']}_{r['race_number']}_{r['date']}" not in exclude_keys]

    # Random sample (no fixed seed - truly random each time)
    return random.sample(race_list, min(limit, len(race_list)))


def save_results(results: dict, output_dir: str = "data/ab_results"):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ab_test_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath


def print_summary(results: dict):
    """Print summary comparison of all variations."""
    print(f"\n{'='*70}")
    print("  A/B TEST SUMMARY")
    print(f"{'='*70}")

    # Aggregate stats by variation
    var_stats = {}

    for race in results["races"]:
        for var_key, var_result in race["variations"].items():
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
                odds = c.get("odds", 0)
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
                    var_stats[var_key]["by_tag"][tag] = {"picks": 0, "wins": 0, "places": 0}
                var_stats[var_key]["by_tag"][tag]["picks"] += 1
                if won:
                    var_stats[var_key]["by_tag"][tag]["wins"] += 1
                if placed:
                    var_stats[var_key]["by_tag"][tag]["places"] += 1

    # Print comparison table
    print(f"\n{'Variation':<25} {'Picks':>6} {'Wins':>6} {'Win%':>7} {'Places':>7} {'Profit':>8} {'ROI':>7}")
    print("-" * 70)

    for var_key in results["variations_tested"]:
        stats = var_stats.get(var_key, {})
        picks = stats.get("total_picks", 0)
        wins = stats.get("wins", 0)
        places = stats.get("places", 0)
        profit = stats.get("profit", 0)

        win_pct = (wins / picks * 100) if picks > 0 else 0
        roi = (profit / picks * 100) if picks > 0 else 0

        name = VARIATIONS[var_key]["name"][:24]
        print(f"{name:<25} {picks:>6} {wins:>6} {win_pct:>6.1f}% {places:>7} {profit:>+8.2f} {roi:>+6.1f}%")

    # Print by-tag breakdown for each variation
    print(f"\n{'='*70}")
    print("  BY TAG BREAKDOWN")
    print(f"{'='*70}")

    for var_key in results["variations_tested"]:
        stats = var_stats.get(var_key, {})
        name = VARIATIONS[var_key]["name"]
        print(f"\n{name}:")

        for tag, tag_stats in stats.get("by_tag", {}).items():
            picks = tag_stats["picks"]
            wins = tag_stats["wins"]
            places = tag_stats["places"]
            win_pct = (wins / picks * 100) if picks > 0 else 0
            place_pct = (places / picks * 100) if picks > 0 else 0
            print(f"  {tag:<20} {picks:>3} picks, {wins:>2} wins ({win_pct:>5.1f}%), {places:>2} places ({place_pct:>5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test predictor variations")
    parser.add_argument("--limit", type=int, default=5, help="Number of races to test")
    parser.add_argument("--variation", type=str, help="Test single variation only")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--resume", type=str, help="Resume from partial results file")

    args = parser.parse_args()

    # Create output file path upfront for incremental saving
    os.makedirs("data/ab_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/ab_results/ab_test_{timestamp}.json"

    # Check for resume
    completed_races = set()
    existing_results = None
    if args.resume:
        if os.path.exists(args.resume):
            with open(args.resume) as f:
                existing_results = json.load(f)
            completed_races = {
                (r["track"], r["race_number"], r["date"])
                for r in existing_results.get("races", [])
            }
            output_file = args.resume  # Continue writing to same file
            print(f"Resuming from {args.resume} ({len(completed_races)} races already done)")
        else:
            print(f"Resume file not found: {args.resume}")
            sys.exit(1)

    # Load ALL previously tested races from ALL result files to avoid duplicates
    all_tested_keys = set()
    results_dir = "data/ab_results"
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(results_dir, fname)) as f:
                        existing = json.load(f)
                    for r in existing.get("races", []):
                        key = f"{r['track']}_{r['race_number']}_{r['date']}"
                        all_tested_keys.add(key)
                except:
                    pass

    print(f"Found {len(all_tested_keys)} previously tested races to exclude")

    # Get sample races (excluding already tested)
    print(f"Fetching {args.limit} NEW sample races from Railway...")
    races = get_sample_races(args.limit, exclude_keys=all_tested_keys)

    print(f"Got {len(races)} races to test")
    print(f"Output file: {output_file}")

    if not races:
        print("No new races to test!")
        sys.exit(0)

    # Determine variations to test
    if args.variation:
        if args.variation not in VARIATIONS:
            print(f"Unknown variation: {args.variation}")
            print(f"Available: {list(VARIATIONS.keys())}")
            sys.exit(1)
        variations_to_test = [args.variation]
    else:
        variations_to_test = list(VARIATIONS.keys())

    # Estimate time
    est_minutes = len(races) * len(variations_to_test) * 0.4  # ~24 seconds per API call
    print(f"Estimated time: ~{est_minutes:.0f} minutes ({len(races)} races × {len(variations_to_test)} variations)")
    print()

    # Run tests with incremental saving
    results = run_ab_test(races, variations_to_test, verbose=not args.quiet, output_file=output_file, existing_results=existing_results)

    print(f"\n✅ Results saved to: {output_file}")

    # Print summary
    print_summary(results)
