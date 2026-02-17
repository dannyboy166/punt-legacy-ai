"""
Race Data Pipeline for Claude AI Predictor.

Fetches, processes, and formats race data from PuntingForm and Ladbrokes
into a structure suitable for the Claude AI predictor.

Usage:
    from core.race_data import RaceDataPipeline

    pipeline = RaceDataPipeline()
    race_data = pipeline.get_race_data("Randwick", 1, "09-Jan-2026")

    # race_data contains everything Claude needs:
    # - Today's race info (distance, condition, class)
    # - All runners with form history, ratings, odds
    # - Speedmap data
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import csv
import os

from api.puntingform import PuntingFormAPI
from api.ladbrokes import LadbrokeAPI
from core.normalize import normalize_horse_name, horses_match
from core.speed import calculate_run_rating, parse_condition_number
from core.results import PredictionResult, RaceStatus


# Track ratings lookup for venue-adjusted speed ratings
_TRACK_RATINGS: dict[str, float] = {}

def _load_track_ratings() -> dict[str, float]:
    """Load track ratings from CSV file."""
    global _TRACK_RATINGS
    if _TRACK_RATINGS:
        return _TRACK_RATINGS

    csv_path = os.path.join(os.path.dirname(__file__), "normalization", "track_ratings.csv")
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            venue = row.get("venue", "").strip()
            rating_str = row.get("overall_track_rating", "").strip()
            if venue and rating_str:
                try:
                    _TRACK_RATINGS[venue.lower()] = float(rating_str)
                except ValueError:
                    pass

    return _TRACK_RATINGS


def get_track_rating(track_name: str) -> Optional[float]:
    """
    Get overall track rating for venue-adjusted calculations.

    Returns None if track not found.
    Higher rating = weaker track (horses beat baseline easier).
    """
    ratings = _load_track_ratings()

    # Normalize track name for lookup
    track_lower = track_name.lower().strip()

    # Direct match
    if track_lower in ratings:
        return ratings[track_lower]

    # Try replacing hyphens with spaces (e.g., "Sandown-Lakeside" -> "Sandown Lakeside")
    track_spaces = track_lower.replace("-", " ")
    if track_spaces in ratings:
        return ratings[track_spaces]

    # Try partial match (e.g., "Sandown-Lakeside" contains "Sandown")
    for venue, rating in ratings.items():
        if venue in track_lower or track_lower in venue:
            return rating

    return None


@dataclass
class FormRun:
    """A single past run from form history."""
    date: str
    track: str
    distance: int
    condition: str  # "G4", "S5", etc. or "UNK" if missing
    condition_num: Optional[int]  # None if condition unknown
    position: int
    margin: Optional[float]  # None if missing (non-winner with no margin data)
    weight: Optional[float]  # None if missing
    barrier: Optional[int]  # None if missing
    starters: int
    class_: str
    prize_money: int
    rating: Optional[float]  # Normalized speed rating - None if can't calculate
    prep_run: Optional[int] = None  # 1 = first up, 2 = second up, etc.
    is_barrier_trial: bool = False  # True if this was a barrier trial
    rating_venue_adjusted: Optional[float] = None  # Rating adjusted by track quality

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "track": self.track,
            "distance": self.distance,
            "condition": self.condition,
            "condition_num": self.condition_num,
            "position": self.position,
            "margin": self.margin,
            "weight": self.weight,
            "barrier": self.barrier,
            "starters": self.starters,
            "class": self.class_,
            "prize_money": self.prize_money,
            "rating": round(self.rating * 100, 1) if self.rating else None,
            "prep_run": self.prep_run,  # 1=1st up, 2=2nd up, etc.
            "is_barrier_trial": self.is_barrier_trial,
            "rating_venue_adjusted": round(self.rating_venue_adjusted * 100, 1) if self.rating_venue_adjusted else None,
        }


@dataclass
class RunnerData:
    """Complete data for a single runner."""
    name: str
    tab_no: int
    barrier: int
    weight: float
    age: int
    sex: str

    # Odds
    odds: Optional[float]
    place_odds: Optional[float]
    odds_source: str
    implied_prob: Optional[float]

    # Connections
    jockey: str
    trainer: str
    jockey_a2e: Optional[float]
    trainer_a2e: Optional[float]
    jockey_trainer_a2e: Optional[float]

    # Career stats
    career_starts: int
    career_wins: int
    career_seconds: int
    career_thirds: int
    win_pct: float
    place_pct: float

    # Records at today's conditions
    track_record: Optional[dict]
    distance_record: Optional[dict]
    condition_record: Optional[dict]
    first_up: bool
    second_up: bool
    first_up_record: Optional[dict] = None  # Career record when first up
    second_up_record: Optional[dict] = None  # Career record when second up

    # Form history with ratings
    form: list[FormRun] = field(default_factory=list)

    # Speedmap
    early_speed_rank: Optional[int] = None
    settling_position: Optional[int] = None

    # PFAI Ratings (from PuntingForm)
    pfai_rank: Optional[int] = None  # 1 = best, None if not available

    # Extra fields (synced with backtest pipeline)
    days_since_last: Optional[int] = None
    gear_changes: Optional[str] = None


    @property
    def race_runs_count(self) -> int:
        """Count actual race runs (excluding barrier trials)."""
        return len([f for f in self.form if not f.is_barrier_trial])

    @property
    def trial_runs_count(self) -> int:
        """Count barrier trial runs."""
        return len([f for f in self.form if f.is_barrier_trial])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tab_no": self.tab_no,
            "barrier": self.barrier,
            "weight": self.weight,
            "age": self.age,
            "sex": self.sex,
            "odds": self.odds,
            "place_odds": self.place_odds,
            "odds_source": self.odds_source,
            "implied_prob": round(self.implied_prob, 1) if self.implied_prob else None,
            "jockey": self.jockey,
            "trainer": self.trainer,
            "jockey_a2e": round(self.jockey_a2e, 2) if self.jockey_a2e else None,
            "trainer_a2e": round(self.trainer_a2e, 2) if self.trainer_a2e else None,
            "jockey_trainer_a2e": round(self.jockey_trainer_a2e, 2) if self.jockey_trainer_a2e else None,
            "career": f"{self.career_starts}: {self.career_wins}-{self.career_seconds}-{self.career_thirds}",
            "win_pct": round(self.win_pct, 1),
            "place_pct": round(self.place_pct, 1),
            "track_record": self.track_record,
            "distance_record": self.distance_record,
            "condition_record": self.condition_record,
            "first_up": self.first_up,
            "second_up": self.second_up,
            "first_up_record": self.first_up_record,
            "second_up_record": self.second_up_record,
            "form": [f.to_dict() for f in self.form],
            "race_runs_count": self.race_runs_count,
            "trial_runs_count": self.trial_runs_count,
            "early_speed_rank": self.early_speed_rank,
            "settling_position": self.settling_position,
            "pfai_rank": self.pfai_rank,
        }


@dataclass
class RaceData:
    """Complete race data for Claude."""
    track: str
    race_number: int
    race_name: str
    distance: int
    condition: str
    condition_num: int
    class_: str
    prize_money: int
    start_time: str
    rail_position: str

    runners: list[RunnerData] = field(default_factory=list)

    # Pace scenario
    leaders_count: int = 0  # Runners with speed rank 1-2
    pace_scenario: str = "unknown"  # "hot", "moderate", "soft"

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track": self.track,
            "race_number": self.race_number,
            "race_name": self.race_name,
            "distance": self.distance,
            "condition": self.condition,
            "condition_num": self.condition_num,
            "class": self.class_,
            "prize_money": self.prize_money,
            "start_time": self.start_time,
            "rail_position": self.rail_position,
            "runners": [r.to_dict() for r in self.runners],
            "leaders_count": self.leaders_count,
            "pace_scenario": self.pace_scenario,
            "field_size": len(self.runners),
            "warnings": self.warnings,
        }

    def to_prompt_text(self, include_venue_adjusted: bool = False) -> str:
        """Format as text for Claude prompt.

        Args:
            include_venue_adjusted: If True, add Adj column with venue-adjusted ratings.
        """
        lines = [
            f"# {self.track} Race {self.race_number}: {self.race_name}",
            f"Distance: {self.distance}m | Condition: {self.condition} | Class: {self.class_}",
            f"Field Size: {len(self.runners)} | Pace: {self.pace_scenario} ({self.leaders_count} leaders)",
        ]

        # Add warnings if any exist
        if self.warnings:
            lines.append("")
            lines.append("## ⚠️ Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")

        lines.extend([
            "",
            "## Runners",
            "",
        ])

        for r in sorted(self.runners, key=lambda x: x.tab_no):
            lines.append(f"### {r.tab_no}. {r.name}")

            if r.odds:
                place_str = f" / ${r.place_odds:.2f} place" if r.place_odds else ""
                lines.append(f"Odds: ${r.odds:.2f} win{place_str} → {r.implied_prob:.1f}% implied")
            else:
                lines.append("Odds: Not available")

            lines.append(f"Jockey: {r.jockey} (A/E: {r.jockey_a2e or 'N/A'})")
            lines.append(f"Trainer: {r.trainer} (A/E: {r.trainer_a2e or 'N/A'})")
            lines.append(f"Career: {r.career_starts}: {r.career_wins}-{r.career_seconds}-{r.career_thirds} ({r.win_pct:.0f}% win)")

            # Weight change from last race run
            weight_change_str = ""
            race_form_runs = [f for f in r.form if not f.is_barrier_trial]
            if race_form_runs:
                last_weight = race_form_runs[0].weight
                if last_weight and r.weight:
                    diff = r.weight - last_weight
                    if diff > 0:
                        weight_change_str = f" (↑{diff:.1f}kg from last)"
                    elif diff < 0:
                        weight_change_str = f" (↓{abs(diff):.1f}kg from last)"
            lines.append(f"Barrier: {r.barrier} | Weight: {r.weight}kg{weight_change_str}")

            # Gear changes
            if r.gear_changes:
                lines.append(f"Gear: {r.gear_changes}")

            # Days since last run for first-up/second-up display
            if r.first_up:
                record = r.first_up_record or {}
                rec_str = f"{record.get('starts', 0)}: {record.get('firsts', 0)}-{record.get('seconds', 0)}-{record.get('thirds', 0)}"
                days_str = f" | {r.days_since_last} days since last run" if r.days_since_last else ""
                lines.append(f"**FIRST UP**{days_str} (career 1st-up record: {rec_str})")
            elif r.second_up:
                record = r.second_up_record or {}
                rec_str = f"{record.get('starts', 0)}: {record.get('firsts', 0)}-{record.get('seconds', 0)}-{record.get('thirds', 0)}"
                days_str = f" | {r.days_since_last} days since last run" if r.days_since_last else ""
                lines.append(f"**SECOND UP**{days_str} (career 2nd-up record: {rec_str})")
            elif r.days_since_last:
                lines.append(f"Days since last run: {r.days_since_last}")

            if r.early_speed_rank is not None:
                lines.append(f"Speed Rank: {r.early_speed_rank} | Settles: {r.settling_position}")

            # Form table with prep run and barrier trial indicator
            if r.form:
                # Show form summary with warning if limited
                form_summary = f"Form: {r.race_runs_count} race runs"
                if r.trial_runs_count > 0:
                    form_summary += f", {r.trial_runs_count} trials"
                if r.race_runs_count < 3:
                    form_summary += " ⚠️ LIMITED FORM DATA"
                lines.append(form_summary)
                lines.append("")
                if include_venue_adjusted:
                    lines.append("| Date | Track | Dist | Cond | Pos | Margin | Rating | Adj | Prep | Trial |")
                    lines.append("|------|-------|------|------|-----|--------|--------|-----|------|-------|")
                else:
                    lines.append("| Date | Track | Dist | Cond | Pos | Margin | Rating | Prep | Trial |")
                    lines.append("|------|-------|------|------|-----|--------|--------|------|-------|")
                for f in r.form[:10]:  # Max 10 runs
                    rating_str = f"{f.rating * 100:.1f}" if f.rating else "N/A"
                    adj_str = f"{f.rating_venue_adjusted * 100:.1f}" if f.rating_venue_adjusted else "-"
                    prep_str = f"{f.prep_run}" if f.prep_run else "-"
                    trial_str = "TRIAL" if f.is_barrier_trial else "-"
                    # Handle None margin
                    if f.margin is not None:
                        margin_str = f"{f.margin}L"
                        if not f.is_barrier_trial and f.margin >= 8:
                            margin_str += " ⚠️eased"
                    else:
                        margin_str = "?" if f.position > 1 else "0L"  # Unknown for non-winners
                    # Handle UNK condition
                    cond_str = f.condition if f.condition != "UNK" else "?"
                    if include_venue_adjusted:
                        lines.append(f"| {f.date} | {f.track[:10]} | {f.distance}m | {cond_str} | {f.position}/{f.starters} | {margin_str} | {rating_str} | {adj_str} | {prep_str} | {trial_str} |")
                    else:
                        lines.append(f"| {f.date} | {f.track[:10]} | {f.distance}m | {cond_str} | {f.position}/{f.starters} | {margin_str} | {rating_str} | {prep_str} | {trial_str} |")

            else:
                lines.append("⚠️ NO FORM AVAILABLE - first starter or no data")

            lines.append("")

        return "\n".join(lines)


class RaceDataPipeline:
    """
    Pipeline to fetch and process race data for Claude.

    Combines data from PuntingForm (form, stats) and Ladbrokes (odds).
    """

    def __init__(
        self,
        pf_api: Optional[PuntingFormAPI] = None,
        lb_api: Optional[LadbrokeAPI] = None,
    ):
        self.pf_api = pf_api or PuntingFormAPI()
        self.lb_api = lb_api or LadbrokeAPI()

    def get_race_data(
        self,
        track: str,
        race_number: int,
        date: str,
        allow_finished: bool = False,
    ) -> tuple[Optional[RaceData], Optional[str]]:
        """
        Get complete race data for Claude.

        Args:
            track: Track name (PuntingForm format)
            race_number: Race number
            date: Date in PuntingForm format (dd-MMM-yyyy)

        Returns:
            Tuple of (RaceData, error_message)
            - If successful: (RaceData, None)
            - If failed: (None, "Human-readable error")
        """
        # 1. Get meeting ID from PuntingForm
        try:
            meetings = self.pf_api.get_meetings(date)
        except Exception as e:
            return None, f"Failed to get meetings: {str(e)}"

        meeting_id = None
        meeting_track = None
        for m in meetings:
            m_track = m.get("track", {}).get("name", "")
            if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                meeting_id = m.get("meetingId")
                meeting_track = m_track
                break

        if not meeting_id:
            return None, f"Track '{track}' not found in meetings for {date}"

        # 2. Fetch all data in parallel (form, fields, speedmaps, ratings, odds)
        try:
            fields_data = self.pf_api.get_fields(meeting_id, race_number)
            form_data = self.pf_api.get_form(meeting_id, race_number, runs=10)
            speedmap_data = self.pf_api.get_speedmaps(meeting_id, race_number)
        except Exception as e:
            return None, f"Failed to fetch PuntingForm data: {str(e)}"

        # Fetch PFAI ratings (non-critical, don't fail if unavailable)
        try:
            ratings_data = self.pf_api.get_ratings(meeting_id)
        except Exception:
            ratings_data = {}

        # 3. Get Ladbrokes odds
        # Convert PF date format (dd-MMM-yyyy) to Ladbrokes format (YYYY-MM-DD)
        try:
            pf_date = datetime.strptime(date, "%d-%b-%Y")
            lb_date = pf_date.strftime("%Y-%m-%d")
        except ValueError:
            lb_date = "today"  # Fallback if date parsing fails

        lb_odds, lb_error = self.lb_api.get_odds_for_pf_track(meeting_track, race_number, lb_date, allow_finished=allow_finished)

        # Check if race is closed/finished - return error immediately (unless admin replay)
        if not allow_finished and lb_error and any(x in lb_error for x in ["has started", "has finished", "been abandoned", "not available"]):
            return None, lb_error

        # 4. Find the race in fields data
        race_info = None
        runners_fields = []

        races = fields_data.get("races", [])
        for race in races:
            if race.get("number") == race_number:
                race_info = race
                runners_fields = race.get("runners", [])
                break

        if not race_info:
            return None, f"Race {race_number} not found at {track}"

        # 5. Get form data indexed by runner ID
        form_by_runner = {}
        if isinstance(form_data, list):
            for runner in form_data:
                runner_id = runner.get("runnerId")
                if runner_id:
                    form_by_runner[runner_id] = runner.get("forms", [])

        # 6. Get speedmap data indexed by tab number
        # (runner IDs differ between fields and speedmap APIs, so match by tabNo)
        speedmap_by_tab = {}
        if isinstance(speedmap_data, list):
            for sm_entry in speedmap_data:
                for item in sm_entry.get("items", []):
                    tab_no = item.get("tabNo")
                    if tab_no is not None:
                        speedmap_by_tab[tab_no] = item

        # 7. Get race-level condition
        # Try conditions endpoint first (for upcoming races), then results (for past races)
        condition = None
        condition_num = None

        # Try conditions endpoint (upcoming races)
        try:
            conditions = self.pf_api.get_conditions()
            for c in conditions:
                if c.get("meetingId") == int(meeting_id):
                    condition = c.get("trackCondition")  # e.g., "Good", "Soft (5)"
                    cond_num_str = c.get("trackConditionNumber")  # e.g., "4", "5"
                    if cond_num_str:
                        try:
                            condition_num = int(cond_num_str)
                        except ValueError:
                            pass
                    break
        except Exception:
            pass

        # Try results endpoint (past races)
        if not condition:
            try:
                results = self.pf_api.get_results(meeting_id, race_number)
                if results and len(results) > 0:
                    race_results = results[0].get("raceResults", [])
                    for rr in race_results:
                        if rr.get("raceNumber") == race_number:
                            cond_label = rr.get("trackConditionLabel")  # e.g., "Good", "Unknown"
                            cond_num = rr.get("trackConditionNumber")
                            # Skip if condition is "Unknown" or number is 0 (not yet assessed)
                            if cond_label and cond_label.lower() != "unknown" and cond_num and cond_num != 0:
                                condition = cond_label
                                try:
                                    condition_num = int(cond_num)
                                except ValueError:
                                    pass
                            break
            except Exception:
                pass

        # Try Ladbrokes meetings endpoint (has track_condition on races)
        if not condition:
            try:
                # Convert date format (dd-MMM-yyyy -> YYYY-MM-DD)
                dt = datetime.strptime(date, "%d-%b-%Y")
                lb_date = dt.strftime("%Y-%m-%d")

                lb_meetings = self.lb_api.get_meetings(date_from=lb_date)
                for m in lb_meetings:
                    # Match by track name
                    lb_track = m.get("name", "")
                    if lb_track.lower() == meeting_track.lower() or meeting_track.lower() in lb_track.lower():
                        races = m.get("races", [])
                        for r in races:
                            if r.get("race_number") == race_number:
                                lb_condition = r.get("track_condition")  # e.g., "Heavy8", "Good4"
                                if lb_condition:
                                    condition = lb_condition
                                    condition_num = parse_condition_number(lb_condition)
                                break
                        break
            except Exception:
                pass

        # Fallback to fields_data - but FAIL if still missing (no silent defaults)
        if not condition:
            condition = fields_data.get("expectedCondition")
        if not condition:
            return None, f"Track condition not available for {meeting_track} R{race_number}. Cannot make accurate prediction."

        if not condition_num:
            condition_num = parse_condition_number(condition)
        if not condition_num:
            return None, f"Could not parse track condition '{condition}' for {meeting_track} R{race_number}. Cannot make accurate prediction."

        # Validate distance - must be present and non-zero
        distance = race_info.get("distance", 0)
        if not distance or distance == 0:
            return None, f"Race distance not available for {meeting_track} R{race_number}. Cannot make accurate prediction."

        race_data = RaceData(
            track=meeting_track,
            race_number=race_number,
            race_name=race_info.get("name", ""),
            distance=distance,
            condition=condition,
            condition_num=condition_num,
            class_=race_info.get("raceClass", ""),
            prize_money=race_info.get("prizeMoney", 0),
            start_time=race_info.get("startTime", ""),
            rail_position=fields_data.get("railPosition", ""),
        )

        # 8. Process each runner
        for runner in runners_fields:
            runner_id = runner.get("runnerId")
            horse_name = runner.get("name", "")

            # Skip if scratched (check PuntingForm first)
            if runner.get("scratched", False):
                continue

            # Get odds from Ladbrokes
            normalized_name = normalize_horse_name(horse_name)
            lb_runner_odds = lb_odds.get(normalized_name, {})
            odds = lb_runner_odds.get("fixed_win")
            place_odds = lb_runner_odds.get("fixed_place")
            odds_source = "ladbrokes" if odds else "none"
            implied_prob = (100 / odds) if odds and odds > 0 else None

            # Skip if scratched (also check Ladbrokes is_scratched)
            if lb_runner_odds.get("scratched", False):
                continue

            # Skip if jockey is blank (late scratching indicator)
            pf_jockey = runner.get("jockey", {})
            if isinstance(pf_jockey, dict):
                jockey_name = pf_jockey.get("fullName", "")
            else:
                jockey_name = ""
            if not jockey_name or jockey_name.strip() == "":
                continue

            # Get A/E data
            jockey_a2e = None
            trainer_a2e = None
            jt_a2e = None

            jockey_a2e_data = runner.get("jockeyA2E_Last100") or {}
            trainer_a2e_data = runner.get("trainerA2E_Last100") or {}
            jt_a2e_data = runner.get("trainerJockeyA2E_Last100") or {}

            if jockey_a2e_data.get("a2E"):
                jockey_a2e = jockey_a2e_data["a2E"]
            if trainer_a2e_data.get("a2E"):
                trainer_a2e = trainer_a2e_data["a2E"]
            if jt_a2e_data.get("a2E"):
                jt_a2e = jt_a2e_data["a2E"]

            # Get condition record based on today's condition
            condition_record = None
            if condition_num <= 4:
                condition_record = runner.get("goodRecord")
            elif condition_num <= 6:
                condition_record = runner.get("softRecord")
            else:
                condition_record = runner.get("heavyRecord")

            # Determine first-up/second-up from form history prepRuns + date gap
            # Form prepRuns: 0 = first-up run, 1 = second-up run, etc.
            # Fields endpoint prepRuns is always 0 (broken), so we use form data.
            SPELL_DAYS = 45  # Gap >= 45 days = new prep (first-up)
            runner_form_raw = form_by_runner.get(runner_id, [])
            # Filter to race runs only (not trials)
            race_runs_raw = [f for f in runner_form_raw if not f.get("isBarrierTrial", False)]
            if race_runs_raw:
                last_run = race_runs_raw[0]
                last_date_str = last_run.get("meetingDate", "")[:10]
                last_prep = last_run.get("prepRuns", 0)
                # Check date gap to detect new prep
                try:
                    last_date = datetime.fromisoformat(last_date_str)
                    race_date = datetime.strptime(date, "%d-%b-%Y")
                    days_gap = (race_date - last_date).days
                except (ValueError, TypeError):
                    days_gap = 0
                if days_gap >= SPELL_DAYS:
                    # Long gap = new prep, horse is first-up
                    first_up = True
                    second_up = False
                else:
                    # Same prep: today = last_prep + 1
                    current_prep = last_prep + 1
                    first_up = current_prep == 0  # Only if prepRuns was -1 (shouldn't happen)
                    second_up = current_prep == 1  # Last run was first-up (prepRuns=0)
            else:
                # No race form = first starter or only trials
                first_up = True
                second_up = False

            # Get first-up/second-up career records
            first_up_record = runner.get("firstUpRecord")
            second_up_record = runner.get("secondUpRecord")

            # Get speedmap data
            sm = speedmap_by_tab.get(runner.get("tabNo", 0), {})

            # Process form history
            form_runs = []
            runner_form = form_by_runner.get(runner_id, [])

            for run in runner_form[:10]:  # Max 10 runs
                # Calculate rating for this run
                rating = calculate_run_rating(run)

                # Parse date
                run_date = run.get("meetingDate", "")
                if run_date:
                    try:
                        dt = datetime.fromisoformat(run_date.replace("Z", "+00:00"))
                        run_date = dt.strftime("%d-%b")
                    except:
                        run_date = run_date[:10]

                run_track = run.get("track", {})
                if isinstance(run_track, dict):
                    run_track = run_track.get("name", "")

                # Calculate venue-adjusted rating
                rating_venue_adjusted = None
                if rating is not None:
                    track_rating = get_track_rating(run_track)
                    if track_rating:
                        rating_venue_adjusted = rating / track_rating

                # Get form run data - skip run if critical data is missing
                run_distance = run.get("distance")
                run_starters = run.get("starters")
                run_position = run.get("position")

                # Skip form runs with missing critical data (useless for analysis)
                if not run_distance or run_distance == 0:
                    continue  # Can't compare without distance
                if not run_starters or run_starters == 0:
                    continue  # Can't interpret position without field size

                # Condition - if missing, we can't calculate accurate rating
                run_condition = run.get("trackCondition")
                run_condition_num = None
                if run_condition:
                    run_condition_num = parse_condition_number(run_condition)
                else:
                    # Mark condition as unknown - rating will be None
                    run_condition = "UNK"
                    rating = None  # Can't calculate rating without condition
                    rating_venue_adjusted = None

                # Get prep run number (1 = first up, 2 = second up, etc.)
                run_prep = run.get("prepRuns")
                if run_prep is not None:
                    run_prep = run_prep + 1  # API returns 0-indexed, we want 1-indexed

                # Weight and margin - use None if missing, not 0
                run_weight = run.get("weight")
                run_margin = run.get("margin")

                # Validate margin for non-winners (should not be 0 or None unless winner)
                if run_position and run_position > 1 and (run_margin is None or run_margin == 0):
                    run_margin = None  # Mark as unknown rather than pretending 0

                form_run = FormRun(
                    date=run_date,
                    track=run_track,
                    distance=run_distance,
                    condition=run_condition,
                    condition_num=run_condition_num,
                    position=run_position or 0,
                    margin=run_margin,
                    weight=run_weight,
                    barrier=run.get("barrier"),
                    starters=run_starters,
                    class_=run.get("raceClass", ""),
                    prize_money=run.get("prizeMoney", 0) or 0,
                    rating=rating,
                    prep_run=run_prep,
                    is_barrier_trial=run.get("isBarrierTrial", False),
                    rating_venue_adjusted=rating_venue_adjusted,
                )
                form_runs.append(form_run)

            # Calculate days since last run
            days_since_last = None
            race_form_only = [f for f in runner_form_raw if not f.get("isBarrierTrial", False)]
            if race_form_only:
                last_run_date_str = race_form_only[0].get("meetingDate", "")[:10]
                try:
                    last_run_dt = datetime.fromisoformat(last_run_date_str.replace("Z", "+00:00"))
                    race_dt = datetime.strptime(date, "%d-%b-%Y")
                    days_since_last = (race_dt - last_run_dt).days
                except (ValueError, TypeError):
                    pass

            # Gear changes
            gear_changes_raw = runner.get("gearChanges")
            gear_changes = gear_changes_raw.strip() if gear_changes_raw and gear_changes_raw.strip() else None

            # Get PFAI rank for this runner
            tab_no = runner.get("tabNo", 0)
            pfai_rank = ratings_data.get(tab_no, {}).get("pfai_rank")

            runner_data = RunnerData(
                name=horse_name,
                tab_no=tab_no,
                barrier=runner.get("barrier", 0),
                weight=runner.get("weight", 0),
                age=runner.get("age", 0),
                sex=runner.get("sex", ""),
                odds=odds,
                place_odds=place_odds,
                odds_source=odds_source,
                implied_prob=implied_prob,
                jockey=runner.get("jockey", {}).get("fullName", "") if isinstance(runner.get("jockey"), dict) else "",
                trainer=runner.get("trainer", {}).get("fullName", "") if isinstance(runner.get("trainer"), dict) else "",
                jockey_a2e=jockey_a2e,
                trainer_a2e=trainer_a2e,
                jockey_trainer_a2e=jt_a2e,
                career_starts=runner.get("careerStarts", 0),
                career_wins=runner.get("careerWins", 0),
                career_seconds=runner.get("careerSeconds", 0),
                career_thirds=runner.get("careerThirds", 0),
                win_pct=runner.get("winPct", 0) or 0,
                place_pct=runner.get("placePct", 0) or 0,
                track_record=runner.get("trackRecord"),
                distance_record=runner.get("distanceRecord"),
                condition_record=condition_record,
                first_up=first_up,
                second_up=second_up,
                first_up_record=first_up_record,
                second_up_record=second_up_record,
                form=form_runs,
                early_speed_rank=sm.get("speed"),
                settling_position=sm.get("settle"),
                pfai_rank=pfai_rank,
                days_since_last=days_since_last,
                gear_changes=gear_changes,
            )

            race_data.runners.append(runner_data)

        # 9. Calculate pace scenario
        leaders = sum(1 for r in race_data.runners if r.early_speed_rank is not None and r.early_speed_rank <= 2)
        race_data.leaders_count = leaders
        if leaders >= 3:
            race_data.pace_scenario = "hot"
        elif leaders <= 1:
            race_data.pace_scenario = "soft"
        else:
            race_data.pace_scenario = "moderate"

        # 10. Add warnings
        if lb_error:
            race_data.warnings.append(f"Odds: {lb_error}")

        runners_without_odds = sum(1 for r in race_data.runners if not r.odds)
        if runners_without_odds > 0:
            race_data.warnings.append(f"{runners_without_odds} runners missing odds")

        runners_without_form = sum(1 for r in race_data.runners if not r.form)
        if runners_without_form > 0:
            race_data.warnings.append(f"{runners_without_form} runners missing form")

        # Count runners with limited form (< 3 actual race runs, excluding trials)
        # Note: First-up horses are NOT limited form - they have race history, just returning from spell
        field_size = len(race_data.runners)
        limited_form_count = sum(
            1 for r in race_data.runners
            if len([f for f in r.form if not f.is_barrier_trial]) < 3
        )
        if limited_form_count >= 3 and limited_form_count > field_size // 2:
            race_data.warnings.append(f"{limited_form_count}/{field_size} runners have < 3 race runs (limited form data)")

        return race_data, None

    def get_meeting_races(
        self,
        track: str,
        date: str,
    ) -> tuple[list[RaceData], list[str]]:
        """
        Get data for all races at a meeting.

        Args:
            track: Track name
            date: Date in PuntingForm format

        Returns:
            Tuple of (list of RaceData, list of errors)
        """
        races = []
        errors = []

        # Get meeting info to find race count
        try:
            meetings = self.pf_api.get_meetings(date)
        except Exception as e:
            return [], [f"Failed to get meetings: {str(e)}"]

        meeting_id = None
        for m in meetings:
            m_track = m.get("track", {}).get("name", "")
            if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                meeting_id = m.get("meetingId")
                break

        if not meeting_id:
            return [], [f"Track '{track}' not found"]

        # Get fields to find all race numbers
        try:
            fields = self.pf_api.get_fields(meeting_id, 0)
        except Exception as e:
            return [], [f"Failed to get fields: {str(e)}"]

        race_numbers = [r.get("number") for r in fields.get("races", [])]

        # Fetch each race
        for race_num in race_numbers:
            race_data, error = self.get_race_data(track, race_num, date)
            if race_data:
                races.append(race_data)
            if error:
                errors.append(f"R{race_num}: {error}")

        return races, errors
