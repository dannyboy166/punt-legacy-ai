"""
Backtest module for historical race predictions.

Provides API-friendly backtest functionality:
- Runs predictions on finished races using Starting Prices (SP)
- Filters scratched horses (3 methods)
- Stores results in tracking database
- Auto-syncs outcomes from PuntingForm

Usage:
    from core.backtest import run_backtest, run_backtest_meeting

    # Single race
    result = run_backtest("Canterbury", 1, "16-Jan-2026")

    # All races at meeting
    results = run_backtest_meeting("Canterbury", "16-Jan-2026")
"""

import json
import re
from typing import Optional
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from api.puntingform import PuntingFormAPI
from core.speed import calculate_speed_rating
from core.predictor import SYSTEM_PROMPT
from core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


@dataclass
class BacktestRunner:
    """A runner in a backtest race."""
    tab: int
    name: str
    sp: float  # Starting Price
    place_odds: float
    scratched: bool = False


@dataclass
class BacktestContender:
    """A contender picked by Claude in backtest."""
    horse: str
    tab_no: int
    odds: float
    place_odds: float
    tag: str
    analysis: str


@dataclass
class BacktestResult:
    """Result of a single race backtest."""
    track: str
    race_number: int
    date: str
    race_name: str
    distance: int
    condition: str

    # Field info
    runners: list[BacktestRunner]
    horses_with_form: int
    total_horses: int
    scratched_count: int

    # Predictions
    contenders: list[BacktestContender]
    summary: str

    # Actual results (if synced)
    results: Optional[dict] = None  # {horse: position}

    # Errors
    error: Optional[str] = None


def parse_time(time_str: str) -> Optional[float]:
    """Parse race time string to seconds."""
    if not time_str:
        return None
    parts = time_str.split(':')
    if len(parts) < 3:
        return None
    try:
        mins = int(parts[1])
        secs = float(parts[2])
        return mins * 60 + secs
    except (ValueError, IndexError):
        return None


def get_backtest_prompt(track: str, race_number: int, date: str, api: Optional[PuntingFormAPI] = None):
    """
    Build prompt for backtest prediction.

    Returns:
        (prompt_text, runners_data, race_info, scratched_count) or (None, error_message, None, 0)
    """
    if api is None:
        api = PuntingFormAPI()

    meetings = api.get_meetings(date)
    meeting = next((m for m in meetings if m.get('track', {}).get('name', '').lower() == track.lower()), None)

    if not meeting:
        return None, f"Track '{track}' not found on {date}", None, 0

    meeting_id = meeting.get('meetingId')

    try:
        fields_data = api.get_fields(meeting_id, race_number)
    except Exception as e:
        return None, f"Race {race_number} not found", None, 0

    form_data = api.get_form(meeting_id, race_number, runs=10)

    races = fields_data.get('races', [])
    if not races:
        return None, "No race data", None, 0

    race = races[0]
    distance = race.get('distance', 0)
    condition = race.get('trackCondition') or 'G4'
    race_name = race.get('raceName', '')

    form_by_id = {r.get('runnerId'): r.get('forms', []) for r in form_data}

    race_run_counts = []
    scratched_count = 0

    lines = [
        f"# {track} Race {race_number}: {race_name}",
        f"Distance: {distance}m | Condition: {condition}",
        "",
        "## Runners",
        ""
    ]

    runners_data = []

    for r in sorted(race.get('runners', []), key=lambda x: x.get('tabNo', 0)):
        tab = r.get('tabNo', 0)
        name = r.get('name', 'Unknown')

        # Check for scratching (3 methods)
        # 1. Scratched field
        if r.get('scratched'):
            scratched_count += 1
            logger.debug(f"Scratched (field): {name}")
            continue

        # 2. Blank jockey (late scratching)
        jockey_info = r.get('jockey', {})
        jockey_name = jockey_info.get('fullName', '') if isinstance(jockey_info, dict) else ''
        if not jockey_name.strip():
            scratched_count += 1
            logger.debug(f"Scratched (no jockey): {name}")
            continue

        # 3. SP = 0 (another scratching indicator)
        sp = r.get('priceSP', 0) or 0
        if sp == 0:
            scratched_count += 1
            logger.debug(f"Scratched (SP=0): {name}")
            continue

        barrier = r.get('barrier', 0)
        weight = r.get('weight', 0)
        place_odds = round(1 + (sp - 1) * 0.35, 2) if sp > 1 else 1.10

        jockey = jockey_name
        trainer = r.get('trainer', {}).get('fullName', '')
        jockey_ae = r.get('jockeyA2E_Last100', 0)
        trainer_ae = r.get('trainerA2E_Last100', 0)

        career = f"{r.get('careerStarts', 0)}: {r.get('careerWins', 0)}-{r.get('careerSeconds', 0)}-{r.get('careerThirds', 0)}"
        win_pct = r.get('winPct', 0)

        first_up_rec = r.get('firstUpRecord', {})
        first_up_str = f"{first_up_rec.get('starts', 0)}: {first_up_rec.get('firsts', 0)}-{first_up_rec.get('seconds', 0)}-{first_up_rec.get('thirds', 0)}"

        runners_data.append({'tab': tab, 'name': name, 'sp': sp, 'place_odds': place_odds})

        lines.append(f"### {tab}. {name}")
        lines.append(f"Barrier: {barrier} | Weight: {weight}kg")
        lines.append(f"Odds: ${sp:.2f} win / ${place_odds:.2f} place")
        lines.append(f"Jockey: {jockey} (A/E: {jockey_ae or 'N/A'})")
        lines.append(f"Trainer: {trainer} (A/E: {trainer_ae or 'N/A'})")
        lines.append(f"Career: {career} ({win_pct:.0f}% win)")

        runner_id = r.get('runnerId')
        forms = form_by_id.get(runner_id, [])

        race_runs = [f for f in forms if not f.get('isBarrierTrial')]
        trial_runs = [f for f in forms if f.get('isBarrierTrial')]
        race_run_counts.append(len(race_runs))

        if not forms:
            lines.append("**FIRST STARTER** (no form)")
        elif len(race_runs) == 0:
            lines.append("**FIRST STARTER** (trials only, no race form)")
        elif race_runs[0].get('prepRuns', 0) == 0:
            lines.append(f"**SECOND UP** (career 1st-up record: {first_up_str})")

        if forms:
            form_summary = f"Form: {len(race_runs)} race runs"
            if trial_runs:
                form_summary += f", {len(trial_runs)} trials"
            if len(race_runs) == 0:
                form_summary += " - NO RACE FORM"
            elif len(race_runs) < 3:
                form_summary += " - LIMITED"
            lines.append(form_summary)
            lines.append("")
            lines.append("| Date | Track | Dist | Cond | Pos | Margin | Rating | Prep | Trial |")
            lines.append("|------|-------|------|------|-----|--------|--------|------|-------|")

            for f in forms[:10]:
                f_date = f.get('meetingDate', '')[:10]
                f_track = f.get('track', {}).get('name', '')[:10]
                f_dist = f.get('distance', 0)
                f_cond = f.get('trackCondition', '')
                f_pos = f.get('position', 0)
                f_starters = f.get('starters', 0)
                f_margin = f.get('margin', 0)
                f_prep = f.get('prepRuns', 0)
                is_trial = f.get('isBarrierTrial', False)

                rating = None
                time_secs = parse_time(f.get('officialRaceTime'))
                if time_secs and f_dist and f_pos and not is_trial:
                    try:
                        if f_pos == 1:
                            winner_time = time_secs
                        else:
                            winner_time = time_secs - (f_margin * 0.17)

                        rating = calculate_speed_rating(
                            distance=f_dist,
                            winner_time=winner_time,
                            margin=f_margin,
                            position=f_pos,
                            condition=f_cond or 'G4'
                        )
                    except:
                        pass

                rating_str = f"{rating:.3f}" if rating else "N/A"
                prep_str = str(f_prep) if f_prep else "-"
                trial_str = "TRIAL" if is_trial else "-"

                lines.append(f"| {f_date} | {f_track} | {f_dist}m | {f_cond} | {f_pos}/{f_starters} | {f_margin}L | {rating_str} | {prep_str} | {trial_str} |")
        else:
            lines.append("NO FORM AVAILABLE - first starter")

        lines.append("")

    # Add field summary
    horses_with_form = sum(1 for c in race_run_counts if c > 0)
    total_horses = len(race_run_counts)

    warning = f"FIELD ANALYSIS: {horses_with_form}/{total_horses} horses have race form"
    if horses_with_form < total_horses / 2:
        warning += " - MAJORITY UNPROVEN (recommend 0 picks)"
    lines.insert(3, warning)
    lines.insert(4, "")

    race_info = {
        'distance': distance,
        'condition': condition,
        'name': race_name,
        'with_form': horses_with_form,
        'total': total_horses
    }

    return "\n".join(lines), runners_data, race_info, scratched_count


def run_claude_prediction(prompt_text: str) -> str:
    """Call Claude to get prediction."""
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Analyze this race and pick your contenders (0-3).\n\n{prompt_text}\n\nRespond with JSON only."}]
    )

    return response.content[0].text


def parse_prediction(raw_response: str) -> tuple[list[dict], str]:
    """Parse Claude's JSON response."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('contenders', []), data.get('summary', '')
    except Exception as e:
        logger.warning(f"Parse error: {e}")
    return [], f"Parse error: {raw_response[:200]}"


def run_backtest(track: str, race_number: int, date: str, api: Optional[PuntingFormAPI] = None) -> BacktestResult:
    """
    Run backtest on a single race.

    Args:
        track: Track name (e.g., "Canterbury")
        race_number: Race number (1-12)
        date: Date in dd-MMM-yyyy format (e.g., "16-Jan-2026")
        api: Optional PuntingFormAPI instance (reused for batch operations)

    Returns:
        BacktestResult with predictions and field info
    """
    if api is None:
        api = PuntingFormAPI()

    # Get prompt and field data
    prompt_text, runners_data, race_info, scratched_count = get_backtest_prompt(track, race_number, date, api)

    if prompt_text is None:
        return BacktestResult(
            track=track,
            race_number=race_number,
            date=date,
            race_name="",
            distance=0,
            condition="",
            runners=[],
            horses_with_form=0,
            total_horses=0,
            scratched_count=scratched_count,
            contenders=[],
            summary="",
            error=runners_data  # Error message
        )

    # Build runners list
    runners = [
        BacktestRunner(
            tab=r['tab'],
            name=r['name'],
            sp=r['sp'],
            place_odds=r['place_odds']
        )
        for r in runners_data
    ]

    # Run Claude prediction
    raw_response = run_claude_prediction(prompt_text)
    contenders_data, summary = parse_prediction(raw_response)

    # Build contenders list
    contenders = [
        BacktestContender(
            horse=c.get('horse', ''),
            tab_no=c.get('tab_no', 0),
            odds=c.get('odds', 0),
            place_odds=c.get('place_odds', 0),
            tag=c.get('tag', ''),
            analysis=c.get('analysis', '')
        )
        for c in contenders_data
    ]

    return BacktestResult(
        track=track,
        race_number=race_number,
        date=date,
        race_name=race_info['name'],
        distance=race_info['distance'],
        condition=race_info['condition'],
        runners=runners,
        horses_with_form=race_info['with_form'],
        total_horses=race_info['total'],
        scratched_count=scratched_count,
        contenders=contenders,
        summary=summary
    )


def run_backtest_meeting(
    track: str,
    date: str,
    race_start: int = 1,
    race_end: int = 12
) -> list[BacktestResult]:
    """
    Run backtest on multiple races at a meeting.

    Args:
        track: Track name
        date: Date in dd-MMM-yyyy format
        race_start: First race number (default 1)
        race_end: Last race number (default 12)

    Returns:
        List of BacktestResult for each race
    """
    api = PuntingFormAPI()
    results = []

    for race_num in range(race_start, race_end + 1):
        logger.info(f"Backtesting {track} R{race_num} {date}")
        result = run_backtest(track, race_num, date, api)

        if result.error and "not found" in result.error.lower():
            # No more races at this meeting
            break

        results.append(result)

    return results


def get_race_results(track: str, race_number: int, date: str, api: Optional[PuntingFormAPI] = None) -> Optional[dict]:
    """
    Fetch actual race results from PuntingForm.

    Returns:
        Dict mapping horse names to finishing positions, or None if not available
    """
    if api is None:
        api = PuntingFormAPI()

    try:
        meetings = api.get_meetings(date)
        meeting = next((m for m in meetings if m.get('track', {}).get('name', '').lower() == track.lower()), None)

        if not meeting:
            return None

        meeting_id = meeting.get('meetingId')
        results_data = api.get_results(meeting_id, race_number)

        if not results_data:
            return None

        # PuntingForm returns [{meetingId, raceResults: [...]}]
        races = []
        if results_data and len(results_data) > 0:
            meeting_data = results_data[0]
            races = meeting_data.get('raceResults', [])

        for race in races:
            if race.get('raceNumber') == race_number:
                results = {}
                for runner in race.get('runners', []):
                    position = runner.get('position', 0)
                    horse_name = runner.get('runner', '')
                    if position > 0 and horse_name:
                        results[horse_name] = position
                return results

        return None

    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        return None
