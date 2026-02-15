"""
Backtest script for historical race predictions.

This script allows backtesting the v2 predictor on finished races.

Key features:
- Uses PuntingForm Starting Price (SP) instead of Ladbrokes live odds
- Verifies form data is from BEFORE race day (not cheating)
- Filters scratched horses (scratched field, blank jockey, SP=$0)
- Same prompt and logic as bet_type_predictor.py

Usage:
    python experiments/backtest.py "Rosehill" 2 "17-Jan-2026"
    python experiments/backtest.py "Flemington" 4 "17-Jan-2026"

Output includes:
- Field with Starting Prices
- Form analysis (X/Y have race runs)
- Prediction with contenders and tags

Note: For live races, use bet_type_predictor.py instead (uses Ladbrokes odds).
"""
import sys
sys.path.insert(0, '/Users/danielsamus/punt-legacy-ai')

from datetime import datetime
from api.puntingform import PuntingFormAPI
from core.speed import calculate_speed_rating, calculate_run_rating
import anthropic
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert horse racing analyst.

Pick 0-3 contenders for this race. For each, assign a tag:
- **"The one to beat"** - Clear standout
- **"Each-way chance"** - Could win, should place, place odds worth it
- **"Value bet"** - Odds better than their form suggests

**Pick 0 contenders (no bet) when:**
- A lot of field has no race form (only trials) - you can't compare unknowns
- Field is too even with no standouts
- Insufficient data to make confident assessment

## Key Analysis

Focus on **normalized speed ratings** from RACE runs (not trials) at similar distance and conditions. More recent runs are more relevant.

**Critical:**
- Barrier trials (marked TRIAL) don't count as form - horses don't always try
- If a horse has 0 race runs, they are UNKNOWN - could be brilliant or useless
- If 50%+ of field has no race form, pick 0 contenders - too many unknowns to assess

You also have: win/place odds, jockey/trainer A/E ratios, career record, first-up/second-up records, prep run number, barrier, weight.

## Output

```json
{
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "place_odds": number,
      "tag": "The one to beat" | "Each-way chance" | "Value bet",
      "analysis": "1-2 sentences referencing RACE form",
      "tipsheet_pick": true | false
    }
  ],
  "summary": "Brief overview or reason for 0 picks"
}
```

**tipsheet_pick = true** when you would genuinely bet on this horse yourself:
- Speed ratings clearly support this horse vs the field at this distance/condition
- The odds represent real value (not just "best of a bad bunch")
- You're confident in the pick, not just filling a slot
"""

def parse_time(time_str):
    if not time_str:
        return None
    parts = time_str.split(':')
    if len(parts) < 3:
        return None
    mins = int(parts[1])
    secs = float(parts[2])
    return mins * 60 + secs

def get_backtest_prompt(track: str, race_number: int, date: str):
    api = PuntingFormAPI()
    
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
    if not condition:
        return None, "Track condition not available - cannot make accurate prediction", None
    race_name = race.get('raceName', '')

    form_by_id = {r.get('runnerId'): r.get('forms', []) for r in form_data}

    # Index speedmap by tab number (runner IDs differ between fields and speedmap APIs)
    speedmap_by_tab = {}
    if isinstance(speedmap_data, list):
        for sm_entry in speedmap_data:
            for item in sm_entry.get('items', []):
                tab_no = item.get('tabNo')
                if tab_no is not None:
                    speedmap_by_tab[tab_no] = item
    
    race_run_counts = []
    
    lines = [
        f"# {track} Race {race_number}: {race_name}",
        f"Distance: {distance}m | Condition: {condition}",
        "",
        "## Runners",
        ""
    ]
    
    runners_data = []
    
    for r in sorted(race.get('runners', []), key=lambda x: x.get('tabNo', 0)):
        # Skip scratched horses
        if r.get('scratched'):
            continue
        
        # Skip blank jockey (late scratching) - strip whitespace!
        jockey_info = r.get('jockey', {})
        jockey_name = jockey_info.get('fullName', '') if isinstance(jockey_info, dict) else ''
        if not jockey_name.strip():
            continue
        
        # Skip SP = 0 (another scratching indicator)
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
        jockey_ae_data = r.get('jockeyA2E_Last100') or {}
        trainer_ae_data = r.get('trainerA2E_Last100') or {}
        jt_ae_data = r.get('trainerJockeyA2E_Last100') or {}
        jockey_ae = round(jockey_ae_data['a2E'], 2) if jockey_ae_data.get('a2E') else None
        trainer_ae = round(trainer_ae_data['a2E'], 2) if trainer_ae_data.get('a2E') else None
        jt_ae = round(jt_ae_data['a2E'], 2) if jt_ae_data.get('a2E') else None

        career = f"{r.get('careerStarts', 0)}: {r.get('careerWins', 0)}-{r.get('careerSeconds', 0)}-{r.get('careerThirds', 0)}"
        win_pct = r.get('winPct', 0)
        place_pct = r.get('placePct', 0) or 0

        first_up_rec = r.get('firstUpRecord', {})
        first_up_str = f"{first_up_rec.get('starts', 0)}: {first_up_rec.get('firsts', 0)}-{first_up_rec.get('seconds', 0)}-{first_up_rec.get('thirds', 0)}"
        second_up_rec = r.get('secondUpRecord', {})
        second_up_str = f"{second_up_rec.get('starts', 0)}: {second_up_rec.get('firsts', 0)}-{second_up_rec.get('seconds', 0)}-{second_up_rec.get('thirds', 0)}"
        
        runners_data.append({'tab': tab, 'name': name, 'sp': sp})
        
        lines.append(f"### {tab}. {name}")
        implied_prob = (100 / sp) if sp > 0 else 0
        lines.append(f"Odds: ${sp:.2f} win / ${place_odds:.2f} place → {implied_prob:.1f}% implied")
        lines.append(f"Jockey: {jockey} (A/E: {jockey_ae or 'N/A'})")
        lines.append(f"Trainer: {trainer} (A/E: {trainer_ae or 'N/A'})")
        lines.append(f"Career: {career} ({win_pct:.0f}% win, {place_pct:.0f}% place)")

        runner_id = r.get('runnerId')
        forms = form_by_id.get(runner_id, [])

        race_runs = [f for f in forms if not f.get('isBarrierTrial')]
        trial_runs = [f for f in forms if f.get('isBarrierTrial')]
        race_run_counts.append(len(race_runs))

        # Days since last run + weight change
        days_since_last = None
        weight_change_str = ""
        if race_runs:
            last_race_run = race_runs[0]
            last_date_str = last_race_run.get('meetingDate', '')[:10]
            try:
                last_date = datetime.fromisoformat(last_date_str.replace("Z", "+00:00"))
                race_date = datetime.strptime(date, "%d-%b-%Y")
                days_since_last = (race_date - last_date).days
            except (ValueError, TypeError):
                pass
            # Weight change from last race run
            last_weight = last_race_run.get('weight', 0)
            if last_weight and weight:
                diff = weight - last_weight
                if diff > 0:
                    weight_change_str = f" (↑{diff:.1f}kg from last)"
                elif diff < 0:
                    weight_change_str = f" (↓{abs(diff):.1f}kg from last)"

        lines.append(f"Barrier: {barrier} | Weight: {weight}kg{weight_change_str}")

        # Gear changes
        gear_changes = r.get('gearChanges')
        if gear_changes and gear_changes.strip():
            lines.append(f"Gear: {gear_changes.strip()}")

        # First-up/second-up detection (matching live pipeline logic)
        SPELL_DAYS = 45
        if not forms:
            lines.append("**FIRST STARTER** (no form)")
        elif len(race_runs) == 0:
            lines.append("**FIRST STARTER** (trials only, no race form)")
        else:
            last_prep = race_runs[0].get('prepRuns', 0)

            if days_since_last and days_since_last >= SPELL_DAYS:
                # Long gap = new prep, horse is first-up
                lines.append(f"**FIRST UP** | {days_since_last} days since last run (career 1st-up record: {first_up_str})")
            elif last_prep + 1 == 1:
                # Last run was first-up (prepRuns=0), so today is second-up
                days_str = f" | {days_since_last} days since last run" if days_since_last else ""
                lines.append(f"**SECOND UP**{days_str} (career 2nd-up record: {second_up_str})")
            elif days_since_last:
                lines.append(f"Days since last run: {days_since_last}")

        # Speedmap data
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
            if len(race_runs) == 0:
                form_summary += " ⚠️ NO RACE FORM"
            elif len(race_runs) < 3:
                form_summary += " ⚠️ LIMITED"
            lines.append(form_summary)
            lines.append("")
            lines.append("| Date | Track | Dist | Cond | Pos | Margin | Rating | Prep | Trial |")
            lines.append("|------|-------|------|------|-----|--------|--------|------|-------|")
            
            for f in forms[:10]:
                # Parse date to match live pipeline format (dd-Mon)
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

                # Prep number: API is 0-indexed, convert to 1-indexed
                f_prep = f.get('prepRuns')
                if f_prep is not None:
                    f_prep = f_prep + 1

                # Use same rating calculation as live pipeline
                rating = calculate_run_rating(f)

                rating_str = f"{rating * 100:.1f}" if rating else "N/A"
                prep_str = str(f_prep) if f_prep else "-"
                trial_str = "TRIAL" if is_trial else "-"
                margin_str = f"{f_margin}L"
                if not is_trial and f_margin and f_margin >= 8:
                    margin_str += " ⚠️eased"

                lines.append(f"| {f_date} | {f_track} | {f_dist}m | {f_cond} | {f_pos}/{f_starters} | {margin_str} | {rating_str} | {prep_str} | {trial_str} |")
        else:
            lines.append("⚠️ NO FORM AVAILABLE - first starter")
        
        lines.append("")
    
    # Calculate pace scenario from speedmap data
    leaders = sum(1 for tab_no, sm in speedmap_by_tab.items() if sm.get('speed') and sm['speed'] <= 2)
    if leaders >= 3:
        pace = "hot"
    elif leaders <= 1:
        pace = "soft"
    else:
        pace = "moderate"

    # Update race header with field size, class, and pace
    race_class = race.get('raceClass', '')
    class_str = f" | Class: {race_class}" if race_class else ""
    lines[1] = f"Distance: {distance}m | Condition: {condition}{class_str} | Field Size: {len(runners_data)} | Pace: {pace} ({leaders} leaders)"

    # Add field summary
    horses_with_form = sum(1 for c in race_run_counts if c > 0)
    total_horses = len(race_run_counts)

    warning = f"⚠️ FIELD ANALYSIS: {horses_with_form}/{total_horses} horses have race form"
    if horses_with_form < total_horses / 2:
        warning += " - MAJORITY UNPROVEN (recommend 0 picks)"
    lines.insert(3, warning)
    lines.insert(4, "")

    return "\n".join(lines), runners_data, {'distance': distance, 'condition': condition, 'name': race_name, 'with_form': horses_with_form, 'total': total_horses}


def run_prediction(prompt_text):
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Analyze this race and pick your contenders (0-3).\n\n{prompt_text}\n\nRespond with JSON only."}]
    )
    
    return response.content[0].text


if __name__ == "__main__":
    track = sys.argv[1]
    race_num = int(sys.argv[2])
    date = sys.argv[3]
    
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {track} R{race_num} - {date}")
    print(f"{'='*60}")
    
    result = get_backtest_prompt(track, race_num, date)
    
    if result[0] is None:
        print(f"Error: {result[1]}")
        sys.exit(1)
    
    prompt_text, runners, race_info = result
    
    print(f"\nDistance: {race_info['distance']}m | Condition: {race_info['condition']}")
    print(f"Class: {race_info['name']}")
    print(f"Form: {race_info['with_form']}/{race_info['total']} have race runs")
    print(f"\nFIELD:")
    for r in runners:
        print(f"  {r['tab']}. {r['name']} - SP ${r['sp']:.2f}")
    
    print(f"\n{'='*60}")
    print("  PREDICTION")
    print(f"{'='*60}")
    
    raw = run_prediction(prompt_text)
    
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        data = json.loads(json_match.group()) if json_match else {}
        
        contenders = data.get('contenders', [])
        summary = data.get('summary', '')
        
        if not contenders:
            print(f"\n  ❌ NO CONTENDERS")
            print(f"\n  {summary}")
        else:
            print(f"\n  {len(contenders)} CONTENDER(S):\n")
            for c in contenders:
                tipsheet = c.get('tipsheet_pick', False)
                tipsheet_badge = " ⭐ TIPSHEET" if tipsheet else ""
                print(f"  {c['horse']} (#{c['tab_no']}){tipsheet_badge}")
                print(f"    ${c['odds']:.2f} win / ${c['place_odds']:.2f} place")
                print(f'    "{c["tag"]}"')
                print(f"    {c['analysis']}\n")
            print(f"  SUMMARY: {summary}")
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw: {raw[:500]}")
    
    print(f"\n{'='*60}")
