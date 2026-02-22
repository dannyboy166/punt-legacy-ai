"""
FastAPI Server for Punt Legacy AI Predictor.

Exposes the predictor via HTTP API for the racing-tips-platform frontend.

Run:
    uvicorn server:app --reload --port 8000

Endpoints:
    GET  /meetings?date=09-Jan-2026     - List tracks racing on date
    GET  /races?track=Gosford&date=X    - List races at track
    POST /predict                        - Generate prediction for race
    POST /predict-test                   - TEST: Collateral form analysis (not tracked)
    POST /predict-test-adj               - TEST: Track-adjusted ratings (not tracked)
    POST /backtest                       - Run backtest on historical races
    GET  /health                         - Health check

Backtest Example:
    curl -X POST http://localhost:8000/backtest \\
        -H "Content-Type: application/json" \\
        -d '{"track": "Canterbury", "date": "16-Jan-2026", "race_start": 1, "race_end": 7}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional
import os
import re

from dotenv import load_dotenv
load_dotenv()

from core.race_data import RaceDataPipeline
from core.predictor import Predictor
from core.predictor_collateral import CollateralPredictor
from core.predictor_track_adjusted import TrackAdjustedPredictor
from core.tracking import PredictionTracker
from core.backtest import run_backtest, run_backtest_meeting, get_race_results
from api.puntingform import PuntingFormAPI

app = FastAPI(
    title="Punt Legacy AI Predictor",
    description="AI-powered horse racing predictions using Claude",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://puntlegacy.com",
        "https://racing-platform.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize once
pipeline = RaceDataPipeline()
predictor = Predictor()
collateral_predictor = CollateralPredictor()  # TEST: collateral form analysis
track_adjusted_predictor = TrackAdjustedPredictor()  # TEST: track-adjusted ratings
pf_api = PuntingFormAPI()
tracker = PredictionTracker()


# =============================================================================
# MODELS
# =============================================================================

DATE_PATTERN = re.compile(r'^\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}$')


class PredictionRequest(BaseModel):
    track: str
    race_number: int
    date: str  # Format: dd-MMM-yyyy (e.g., "09-Jan-2026")
    mode: str = "normal"  # "normal" or "promo_bonus"
    allow_finished: bool = False  # Admin only: allow predictions on finished races
    include_admin_data: bool = False  # Admin only: include raw form data for contenders

    @field_validator('track')
    @classmethod
    def track_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Track name cannot be empty')
        return v.strip()

    @field_validator('race_number')
    @classmethod
    def race_number_valid(cls, v: int) -> int:
        if v < 1 or v > 12:
            raise ValueError('Race number must be between 1 and 12')
        return v

    @field_validator('date')
    @classmethod
    def date_format_valid(cls, v: str) -> str:
        if not DATE_PATTERN.match(v):
            raise ValueError('Date must be in format dd-MMM-yyyy (e.g., 09-Jan-2026)')
        return v

    @field_validator('mode')
    @classmethod
    def mode_valid(cls, v: str) -> str:
        if v not in ("normal", "promo_bonus"):
            raise ValueError('Mode must be "normal" or "promo_bonus"')
        return v


class Contender(BaseModel):
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float]
    tag: str  # "The one to beat", "Each-way chance", or "Value bet"
    analysis: str
    confidence: Optional[int] = None  # Deprecated - not used in new model
    tipsheet_pick: bool = False  # True if Claude would genuinely bet on this (admin only)
    pfai_rank: Optional[int] = None  # PuntingForm AI rank (1 = best)


class PromoBonusPick(BaseModel):
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float]
    pick_type: str  # "bonus_bet" or "promo_play"
    analysis: str


class PredictionResponse(BaseModel):
    mode: str = "normal"  # "normal" or "promo_bonus"
    track: str
    race_number: int
    race_name: str
    distance: int
    condition: str
    class_: str
    contenders: list[Contender] = []  # Used in normal mode (can be empty = no picks)
    bonus_pick: Optional[PromoBonusPick] = None  # Used in promo_bonus mode
    promo_pick: Optional[PromoBonusPick] = None  # Used in promo_bonus mode
    summary: str
    runner_notes: dict = {}  # Notes for non-selected runners (1 sentence each)
    race_confidence: Optional[int] = None  # Deprecated - not used in new model
    confidence_reason: Optional[str] = None  # Deprecated - not used in new model
    skipped: bool = False  # True if race was skipped due to insufficient form data
    skip_reason: Optional[str] = None  # Reason for skipping (shown to user)
    warnings: list[str] = []  # Form/data warnings (e.g., "5/8 runners have limited form")
    tracking_stored: bool = False  # True if prediction was stored to tracking DB
    tracking_error: Optional[str] = None  # Error message if tracking failed
    admin_data: Optional[dict] = None  # Admin only: raw form data for contenders


class MeetingResponse(BaseModel):
    track: str
    meeting_id: int
    race_count: int


class RaceResponse(BaseModel):
    race_number: int
    race_name: str
    distance: int
    start_time: str
    runners_count: int


# =============================================================================
# BACKTEST MODELS
# =============================================================================

class MeetingPredictionRequest(BaseModel):
    """Request to generate predictions for an entire meeting."""
    track: str
    date: str  # Format: dd-MMM-yyyy
    race_start: int = 1
    race_end: int = 12
    include_admin_data: bool = True  # Default True since it's admin-only anyway

    @field_validator('track')
    @classmethod
    def track_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Track name cannot be empty')
        return v.strip()

    @field_validator('date')
    @classmethod
    def date_format_valid(cls, v: str) -> str:
        if not DATE_PATTERN.match(v):
            raise ValueError('Date must be in format dd-MMM-yyyy (e.g., 21-Jan-2026)')
        return v


class MeetingRaceResult(BaseModel):
    """Result of a single race prediction within a meeting."""
    race_number: int
    race_name: str
    distance: int
    condition: str
    class_: str
    contenders: list[Contender]
    summary: str
    error: Optional[str] = None
    admin_data: Optional[dict] = None  # Raw form data for admin


class TipsheetPick(BaseModel):
    """A tipsheet-worthy pick from the meeting."""
    race_number: int
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float]
    tag: str
    analysis: str
    pfai_rank: Optional[int] = None


class MeetingPredictionResponse(BaseModel):
    """Response from generating predictions for an entire meeting."""
    track: str
    date: str
    races: list[MeetingRaceResult]
    tipsheet_picks: list[TipsheetPick]
    total_races: int
    races_with_picks: int
    estimated_cost: float


class BacktestRequest(BaseModel):
    """Request to run a backtest on historical races."""
    track: str
    date: str  # Format: dd-MMM-yyyy
    race_start: int = 1
    race_end: int = 12
    auto_sync: bool = True  # Automatically sync outcomes after backtest

    @field_validator('track')
    @classmethod
    def track_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Track name cannot be empty')
        return v.strip()

    @field_validator('date')
    @classmethod
    def date_format_valid(cls, v: str) -> str:
        if not DATE_PATTERN.match(v):
            raise ValueError('Date must be in format dd-MMM-yyyy (e.g., 16-Jan-2026)')
        return v

    @field_validator('race_start', 'race_end')
    @classmethod
    def race_number_valid(cls, v: int) -> int:
        if v < 1 or v > 12:
            raise ValueError('Race number must be between 1 and 12')
        return v


class BacktestContenderResponse(BaseModel):
    """A contender from a backtest."""
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float] = None
    tag: str
    analysis: str


class BacktestRaceResult(BaseModel):
    """Result of a single race backtest."""
    track: str
    race_number: int
    date: str
    race_name: str
    distance: int
    condition: str
    field_size: int
    horses_with_form: int
    scratched_count: int
    contenders: list[BacktestContenderResponse]
    summary: str
    results: Optional[dict] = None  # {horse: position} - actual race results
    error: Optional[str] = None


class BacktestResponse(BaseModel):
    """Response from running a backtest."""
    track: str
    date: str
    races: list[BacktestRaceResult]
    total_races: int
    total_contenders: int
    outcomes_synced: int
    stats: Optional[dict] = None  # Performance stats by tag


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_admin_data(race_data, contenders) -> dict:
    """
    Build admin-only data showing ALL form runs for each runner.

    Marks runs as "relevant" if they match:
    - Within ±20% of today's race distance
    - Similar track condition (within ±2 condition levels)
    """
    from core.speed import parse_condition_number

    today_distance = race_data.distance
    today_condition_num = parse_condition_number(race_data.condition)
    distance_tolerance = today_distance * 0.20  # ±20%

    # Get contender tab numbers for highlighting
    contender_tabs = {c.tab_no for c in contenders}

    all_runners_form = {}

    # Process ALL runners
    for runner in race_data.runners:
        all_runs = []

        for run in runner.form:
            if run.is_barrier_trial:
                continue

            # Check if this run is "relevant" (similar distance/condition)
            is_relevant = True

            # Check distance (within ±20%)
            distance_diff = abs(run.distance - today_distance)
            if distance_diff > distance_tolerance:
                is_relevant = False

            # Check condition (within ±2 levels)
            if today_condition_num is not None and run.condition_num is not None:
                condition_diff = abs(run.condition_num - today_condition_num)
                if condition_diff > 2:
                    is_relevant = False

            all_runs.append({
                "date": run.date,
                "track": run.track,
                "distance": run.distance,
                "condition": run.condition,
                "position": f"{run.position}/{run.starters}",
                "margin": run.margin,
                "weight": run.weight,
                "barrier": run.barrier,
                "rating": round(run.rating, 3) if run.rating else None,
                "is_relevant": is_relevant,  # Highlight flag
            })

        all_runners_form[runner.name] = {
            "tab_no": runner.tab_no,
            "odds": runner.odds,
            "weight": runner.weight,  # Today's weight
            "barrier": runner.barrier,  # Today's barrier
            "total_form_runs": runner.race_runs_count,
            "is_contender": runner.tab_no in contender_tabs,
            "all_runs": all_runs,  # ALL runs with is_relevant flag
        }

    return {
        "race_distance": today_distance,
        "race_condition": race_data.condition,
        "distance_tolerance": "±20%",
        "condition_tolerance": "±2 levels",
        "runners": all_runners_form,
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "punt-legacy-ai"}


def validate_date(date: str) -> None:
    """Validate date format, raise HTTPException if invalid."""
    if not DATE_PATTERN.match(date):
        raise HTTPException(
            status_code=400,
            detail="Date must be in format dd-MMM-yyyy (e.g., 09-Jan-2026)"
        )


@app.get("/meetings", response_model=list[MeetingResponse])
def get_meetings(date: str):
    """
    Get all tracks racing on a given date.

    Args:
        date: Date in format dd-MMM-yyyy (e.g., "09-Jan-2026")

    Returns:
        List of tracks with meeting IDs and race counts
    """
    validate_date(date)
    try:
        meetings = pf_api.get_meetings(date)

        # Filter to Australian meetings only
        aus_meetings = [
            m for m in meetings
            if m.get("track", {}).get("name")
            and m.get("track", {}).get("country") == "AUS"
        ]

        # Fetch race counts in parallel for all meetings
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def get_race_count(meeting_id: int) -> int:
            """Get race count for a meeting by fetching fields."""
            try:
                fields = pf_api.get_fields(meeting_id, 0)
                races = fields.get("races", [])
                return len(races) if races else 0
            except Exception:
                return 0

        race_counts = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_meeting = {
                executor.submit(get_race_count, int(m.get("meetingId", 0))): m.get("meetingId")
                for m in aus_meetings
            }
            for future in as_completed(future_to_meeting):
                meeting_id = future_to_meeting[future]
                race_counts[meeting_id] = future.result()

        return [
            MeetingResponse(
                track=m.get("track", {}).get("name", "Unknown"),
                meeting_id=int(m.get("meetingId", 0)),
                race_count=race_counts.get(m.get("meetingId"), 0)
            )
            for m in aus_meetings
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/races", response_model=list[RaceResponse])
def get_races(track: str, date: str):
    """
    Get all races at a track on a given date.

    Args:
        track: Track name (e.g., "Gosford")
        date: Date in format dd-MMM-yyyy

    Returns:
        List of races with basic info
    """
    validate_date(date)
    if not track or not track.strip():
        raise HTTPException(status_code=400, detail="Track name cannot be empty")
    track = track.strip()
    try:
        meetings = pf_api.get_meetings(date)

        # Find meeting ID for track
        meeting_id = None
        for m in meetings:
            m_track = m.get("track", {}).get("name", "")
            if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                meeting_id = m.get("meetingId")
                break

        if not meeting_id:
            raise HTTPException(status_code=404, detail=f"Track '{track}' not found on {date}")

        # Get fields for all races
        fields = pf_api.get_fields(meeting_id, 0)
        races = fields.get("races", [])

        return [
            RaceResponse(
                race_number=r.get("number", 0),
                race_name=r.get("name", ""),
                distance=r.get("distance", 0),
                start_time=r.get("startTime", ""),
                runners_count=len(r.get("runners", []))
            )
            for r in races
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/odds")
def get_odds(track: str, race_number: int, date: str):
    """
    Get live Ladbrokes odds for a race.

    Args:
        track: Track name from PuntingForm (e.g., "Randwick")
        race_number: Race number
        date: Date in format dd-MMM-yyyy

    Returns:
        Dict of runner odds keyed by lowercase horse name.
    """
    validate_date(date)
    if not track or not track.strip():
        raise HTTPException(status_code=400, detail="Track name cannot be empty")
    if race_number < 1:
        raise HTTPException(status_code=400, detail="Race number must be >= 1")

    from api.ladbrokes import LadbrokeAPI
    from datetime import datetime as dt
    try:
        lb_api = LadbrokeAPI()
        try:
            lb_date = dt.strptime(date, "%d-%b-%Y").strftime("%Y-%m-%d")
        except ValueError:
            lb_date = "today"
        odds_dict, error = lb_api.get_odds_for_pf_track(track.strip(), race_number, lb_date)

        if error:
            return {"runners": {}, "error": error}

        return {"runners": odds_dict, "error": None}
    except Exception as e:
        return {"runners": {}, "error": str(e)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Generate AI prediction for a race.

    Args:
        track: Track name
        race_number: Race number
        date: Date in format dd-MMM-yyyy
        mode: "normal" or "promo_bonus"

    Returns:
        Normal mode: 1-3 contenders with analysis and summary
        Promo/Bonus mode: bonus_pick and promo_pick with analysis and summary
    """
    try:
        # Get race data
        race_data, error = pipeline.get_race_data(req.track, req.race_number, req.date, allow_finished=req.allow_finished)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Check if odds are available - we need odds to make predictions
        runners_with_odds = sum(1 for r in race_data.runners if r.odds)
        if runners_with_odds == 0:
            raise HTTPException(
                status_code=503,
                detail="Odds not available yet. Please wait for the market to open and try again."
            )

        # Check if >50% of field has no race form (only trials or first starters)
        # If so, skip the Claude API call - predictions would be unreliable
        total_runners = len(race_data.runners)
        runners_with_no_form = sum(1 for r in race_data.runners if r.race_runs_count == 0)
        no_form_percentage = (runners_with_no_form / total_runners * 100) if total_runners > 0 else 0

        if no_form_percentage > 50:
            # Return skipped response - doesn't call Claude, doesn't count against limit
            return PredictionResponse(
                mode=req.mode,
                track=race_data.track,
                race_number=race_data.race_number,
                race_name=race_data.race_name,
                distance=race_data.distance,
                condition=race_data.condition,
                class_=race_data.class_,
                contenders=[],
                summary=f"This race has insufficient form data for reliable predictions. {runners_with_no_form} of {total_runners} runners ({no_form_percentage:.0f}%) are first starters or have only barrier trial form.",
                skipped=True,
                skip_reason=f"{runners_with_no_form}/{total_runners} runners have no race history"
            )

        # Generate prediction with specified mode
        result = predictor.predict(race_data, mode=req.mode)

        # Build response based on mode
        if req.mode == "promo_bonus":
            # Promo/Bonus mode response
            bonus_pick_response = None
            promo_pick_response = None

            # Build bonus_pick response
            if result.bonus_pick:
                place_odds = None
                for r in race_data.runners:
                    if r.tab_no == result.bonus_pick.tab_no:
                        place_odds = r.place_odds
                        break
                bonus_pick_response = PromoBonusPick(
                    horse=result.bonus_pick.horse,
                    tab_no=result.bonus_pick.tab_no,
                    odds=result.bonus_pick.odds,
                    place_odds=place_odds,
                    pick_type=result.bonus_pick.pick_type,
                    analysis=result.bonus_pick.analysis
                )

            # Build promo_pick response
            if result.promo_pick:
                place_odds = None
                for r in race_data.runners:
                    if r.tab_no == result.promo_pick.tab_no:
                        place_odds = r.place_odds
                        break
                promo_pick_response = PromoBonusPick(
                    horse=result.promo_pick.horse,
                    tab_no=result.promo_pick.tab_no,
                    odds=result.promo_pick.odds,
                    place_odds=place_odds,
                    pick_type=result.promo_pick.pick_type,
                    analysis=result.promo_pick.analysis
                )

            # Check we got at least one pick
            if not bonus_pick_response and not promo_pick_response:
                raise HTTPException(
                    status_code=503,
                    detail="Could not generate promo/bonus picks. Odds may not be available yet."
                )

            # Store prediction for tracking
            tracking_stored = False
            tracking_error = None
            try:
                tracker.store_prediction(result, race_data, req.date)
                tracking_stored = True
            except Exception as e:
                tracking_error = str(e)
                print(f"WARNING: Failed to store prediction to tracking: {e}")
                import traceback
                traceback.print_exc()

            return PredictionResponse(
                mode="promo_bonus",
                track=race_data.track,
                race_number=race_data.race_number,
                race_name=race_data.race_name,
                distance=race_data.distance,
                condition=race_data.condition,
                class_=race_data.class_,
                contenders=[],
                bonus_pick=bonus_pick_response,
                promo_pick=promo_pick_response,
                summary=result.summary,
                race_confidence=result.race_confidence,
                confidence_reason=result.confidence_reason,
                warnings=race_data.warnings,
                tracking_stored=tracking_stored,
                tracking_error=tracking_error
            )

        else:
            # Normal mode response (0-3 contenders)
            contenders = []
            for c in result.contenders:
                # Get place odds from race data (fallback if not in response)
                place_odds = c.place_odds
                if not place_odds:
                    for r in race_data.runners:
                        if r.tab_no == c.tab_no:
                            place_odds = r.place_odds
                            break

                contenders.append(Contender(
                    horse=c.horse,
                    tab_no=c.tab_no,
                    odds=c.odds,
                    place_odds=place_odds,
                    tag=c.tag,
                    analysis=c.analysis,
                    confidence=c.confidence,  # Will be None in new model
                    tipsheet_pick=c.tipsheet_pick,
                    pfai_rank=c.pfai_rank,
                ))

            # Store prediction for tracking (even if 0 contenders)
            tracking_stored = False
            tracking_error = None
            try:
                tracker.store_prediction(result, race_data, req.date)
                tracking_stored = True
            except Exception as e:
                # Don't fail the request if tracking fails, but capture the error
                tracking_error = str(e)
                print(f"WARNING: Failed to store prediction to tracking: {e}")
                import traceback
                traceback.print_exc()

            # Build admin data if requested
            admin_data = None
            if req.include_admin_data and contenders:
                admin_data = build_admin_data(race_data, result.contenders)

            # Return response (contenders can be empty = no picks for this race)
            return PredictionResponse(
                mode="normal",
                track=race_data.track,
                race_number=race_data.race_number,
                race_name=race_data.race_name,
                distance=race_data.distance,
                condition=race_data.condition,
                class_=race_data.class_,
                contenders=contenders,
                summary=result.summary,
                race_confidence=result.race_confidence,
                confidence_reason=result.confidence_reason,
                warnings=race_data.warnings,
                tracking_stored=tracking_stored,
                tracking_error=tracking_error,
                admin_data=admin_data,
                runner_notes=result.runner_notes
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-test", response_model=PredictionResponse)
def predict_test(req: PredictionRequest):
    """
    TEST ENDPOINT: Generate prediction using collateral form analysis.

    Same as /predict but uses a different prompt focused on comparing horses
    through common tracks and opponents, rather than absolute speed ratings.

    Ideal for midweek/country racing where track speeds vary significantly.

    NOT tracked, NOT stored - for testing purposes only.
    """
    try:
        # Get race data
        race_data, error = pipeline.get_race_data(req.track, req.race_number, req.date, allow_finished=req.allow_finished)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Check if odds are available
        runners_with_odds = sum(1 for r in race_data.runners if r.odds)
        if runners_with_odds == 0:
            raise HTTPException(
                status_code=503,
                detail="Odds not available yet. Please wait for the market to open and try again."
            )

        # Check if >50% of field has no race form
        total_runners = len(race_data.runners)
        runners_with_no_form = sum(1 for r in race_data.runners if r.race_runs_count == 0)
        no_form_percentage = (runners_with_no_form / total_runners * 100) if total_runners > 0 else 0

        if no_form_percentage > 50:
            return PredictionResponse(
                mode=req.mode,
                track=race_data.track,
                race_number=race_data.race_number,
                race_name=race_data.race_name,
                distance=race_data.distance,
                condition=race_data.condition,
                class_=race_data.class_,
                contenders=[],
                summary=f"[TEST] This race has insufficient form data for reliable predictions. {runners_with_no_form} of {total_runners} runners ({no_form_percentage:.0f}%) are first starters or have only barrier trial form.",
                skipped=True,
                skip_reason=f"{runners_with_no_form}/{total_runners} runners have no race history"
            )

        # Generate prediction with COLLATERAL PREDICTOR (different prompt)
        result = collateral_predictor.predict(race_data)

        # Build contenders response
        contenders = []
        for c in result.contenders:
            place_odds = c.place_odds
            if not place_odds:
                for r in race_data.runners:
                    if r.tab_no == c.tab_no:
                        place_odds = r.place_odds
                        break

            contenders.append(Contender(
                horse=c.horse,
                tab_no=c.tab_no,
                odds=c.odds,
                place_odds=place_odds,
                tag=c.tag,
                analysis=c.analysis,
                confidence=c.confidence,
                tipsheet_pick=c.tipsheet_pick,
                pfai_rank=c.pfai_rank,
            ))

        # Build admin data if requested
        admin_data = None
        if req.include_admin_data and contenders:
            admin_data = build_admin_data(race_data, result.contenders)

        # NOT tracked - this is test only
        return PredictionResponse(
            mode="normal",
            track=race_data.track,
            race_number=race_data.race_number,
            race_name=race_data.race_name,
            distance=race_data.distance,
            condition=race_data.condition,
            class_=race_data.class_,
            contenders=contenders,
            summary=f"[TEST - Collateral Form] {result.summary}",
            race_confidence=result.race_confidence,
            confidence_reason=result.confidence_reason,
            warnings=race_data.warnings,
            tracking_stored=False,  # Test endpoint - not tracked
            admin_data=admin_data,
            runner_notes=result.runner_notes
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-test-adj", response_model=PredictionResponse)
def predict_test_adj(req: PredictionRequest):
    """
    TEST ENDPOINT: Generate prediction showing track-adjusted ratings.

    Same as /predict but shows an "Adj" column in form tables with
    ratings normalized by track speed (removes track bias).

    Uses the exact same prompt as live predictor.

    NOT tracked, NOT stored - for testing purposes only.
    """
    try:
        # Get race data
        race_data, error = pipeline.get_race_data(req.track, req.race_number, req.date, allow_finished=req.allow_finished)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Check if odds are available
        runners_with_odds = sum(1 for r in race_data.runners if r.odds)
        if runners_with_odds == 0:
            raise HTTPException(
                status_code=503,
                detail="Odds not available yet. Please wait for the market to open and try again."
            )

        # Check if >50% of field has no race form
        total_runners = len(race_data.runners)
        runners_with_no_form = sum(1 for r in race_data.runners if r.race_runs_count == 0)
        no_form_percentage = (runners_with_no_form / total_runners * 100) if total_runners > 0 else 0

        if no_form_percentage > 50:
            return PredictionResponse(
                mode=req.mode,
                track=race_data.track,
                race_number=race_data.race_number,
                race_name=race_data.race_name,
                distance=race_data.distance,
                condition=race_data.condition,
                class_=race_data.class_,
                contenders=[],
                summary=f"[TEST] This race has insufficient form data for reliable predictions. {runners_with_no_form} of {total_runners} runners ({no_form_percentage:.0f}%) are first starters or have only barrier trial form.",
                skipped=True,
                skip_reason=f"{runners_with_no_form}/{total_runners} runners have no race history"
            )

        # Generate prediction with TRACK-ADJUSTED PREDICTOR (shows Adj column)
        result = track_adjusted_predictor.predict(race_data)

        # Build contenders response
        contenders = []
        for c in result.contenders:
            place_odds = c.place_odds
            if not place_odds:
                for r in race_data.runners:
                    if r.tab_no == c.tab_no:
                        place_odds = r.place_odds
                        break

            contenders.append(Contender(
                horse=c.horse,
                tab_no=c.tab_no,
                odds=c.odds,
                place_odds=place_odds,
                tag=c.tag,
                analysis=c.analysis,
                confidence=c.confidence,
                tipsheet_pick=c.tipsheet_pick,
                pfai_rank=c.pfai_rank,
            ))

        # Build admin data if requested
        admin_data = None
        if req.include_admin_data and contenders:
            admin_data = build_admin_data(race_data, result.contenders)

        # NOT tracked - this is test only
        return PredictionResponse(
            mode="normal",
            track=race_data.track,
            race_number=race_data.race_number,
            race_name=race_data.race_name,
            distance=race_data.distance,
            condition=race_data.condition,
            class_=race_data.class_,
            contenders=contenders,
            summary=f"[TEST - Track-Adjusted] {result.summary}",
            race_confidence=result.race_confidence,
            confidence_reason=result.confidence_reason,
            warnings=race_data.warnings,
            tracking_stored=False,  # Test endpoint - not tracked
            admin_data=admin_data,
            runner_notes=result.runner_notes
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-meeting", response_model=MeetingPredictionResponse)
def predict_meeting(req: MeetingPredictionRequest):
    """
    Generate predictions for all races at a meeting.

    This is an admin-only convenience endpoint that runs the same prediction
    logic as /predict for each race sequentially. Useful for building tipsheets.

    Args:
        track: Track name
        date: Date in format dd-MMM-yyyy
        race_start: First race to predict (default 1)
        race_end: Last race to predict (default 12)

    Returns:
        All race predictions plus a summary of tipsheet_pick horses
    """
    try:
        # Get races at this track
        meetings = pf_api.get_meetings(req.date)

        # Find meeting ID for track
        meeting_id = None
        actual_track_name = req.track
        for m in meetings:
            m_track = m.get("track", {}).get("name", "")
            if m_track.lower() == req.track.lower() or req.track.lower() in m_track.lower():
                meeting_id = m.get("meetingId")
                actual_track_name = m_track
                break

        if not meeting_id:
            raise HTTPException(status_code=404, detail=f"Track '{req.track}' not found on {req.date}")

        # Get all races
        fields = pf_api.get_fields(meeting_id, 0)
        all_races = fields.get("races", [])

        # Filter to requested range
        race_numbers = [r.get("number") for r in all_races
                       if req.race_start <= r.get("number", 0) <= req.race_end]
        race_numbers.sort()

        if not race_numbers:
            raise HTTPException(status_code=404, detail=f"No races found in range R{req.race_start}-R{req.race_end}")

        # Run predictions for each race
        race_results = []
        tipsheet_picks = []
        races_with_picks = 0

        for race_num in race_numbers:
            try:
                # Get race data (same as /predict)
                race_data, error = pipeline.get_race_data(actual_track_name, race_num, req.date)

                if error:
                    race_results.append(MeetingRaceResult(
                        race_number=race_num,
                        race_name="",
                        distance=0,
                        condition="",
                        class_="",
                        contenders=[],
                        summary="",
                        error=error
                    ))
                    continue

                # Check if odds are available
                runners_with_odds = sum(1 for r in race_data.runners if r.odds)
                if runners_with_odds == 0:
                    race_results.append(MeetingRaceResult(
                        race_number=race_num,
                        race_name=race_data.race_name,
                        distance=race_data.distance,
                        condition=race_data.condition,
                        class_=race_data.class_,
                        contenders=[],
                        summary="",
                        error="Odds not available yet"
                    ))
                    continue

                # Check if >50% of field has no race form (auto-skip)
                total_runners = len(race_data.runners)
                runners_with_no_form = sum(1 for r in race_data.runners if r.race_runs_count == 0)
                no_form_pct = (runners_with_no_form / total_runners * 100) if total_runners > 0 else 0
                if no_form_pct > 50:
                    race_results.append(MeetingRaceResult(
                        race_number=race_num,
                        race_name=race_data.race_name,
                        distance=race_data.distance,
                        condition=race_data.condition,
                        class_=race_data.class_,
                        contenders=[],
                        summary=f"Skipped: {runners_with_no_form}/{total_runners} runners ({no_form_pct:.0f}%) have no race form.",
                        error=None
                    ))
                    continue

                # Generate prediction (same logic as /predict)
                result = predictor.predict(race_data, mode="normal")

                # Build contenders list
                contenders = []
                for c in result.contenders:
                    place_odds = c.place_odds
                    if not place_odds:
                        for r in race_data.runners:
                            if r.tab_no == c.tab_no:
                                place_odds = r.place_odds
                                break

                    contender = Contender(
                        horse=c.horse,
                        tab_no=c.tab_no,
                        odds=c.odds,
                        place_odds=place_odds,
                        tag=c.tag,
                        analysis=c.analysis,
                        tipsheet_pick=c.tipsheet_pick,
                        pfai_rank=c.pfai_rank,
                    )
                    contenders.append(contender)

                    # Collect tipsheet picks
                    if c.tipsheet_pick:
                        tipsheet_picks.append(TipsheetPick(
                            race_number=race_num,
                            horse=c.horse,
                            tab_no=c.tab_no,
                            odds=c.odds,
                            place_odds=place_odds,
                            tag=c.tag,
                            analysis=c.analysis,
                            pfai_rank=c.pfai_rank
                        ))

                if contenders:
                    races_with_picks += 1

                # Store prediction for tracking
                try:
                    tracker.store_prediction(result, race_data, req.date)
                except Exception as e:
                    print(f"Warning: Failed to store prediction for R{race_num}: {e}")

                # Build admin data if requested
                admin_data = None
                if req.include_admin_data:
                    admin_data = build_admin_data(race_data, result.contenders)

                race_results.append(MeetingRaceResult(
                    race_number=race_num,
                    race_name=race_data.race_name,
                    distance=race_data.distance,
                    condition=race_data.condition,
                    class_=race_data.class_,
                    contenders=contenders,
                    summary=result.summary,
                    error=None,
                    admin_data=admin_data
                ))

            except Exception as e:
                race_results.append(MeetingRaceResult(
                    race_number=race_num,
                    race_name="",
                    distance=0,
                    condition="",
                    class_="",
                    contenders=[],
                    summary="",
                    error=str(e)
                ))

        return MeetingPredictionResponse(
            track=actual_track_name,
            date=req.date,
            races=race_results,
            tipsheet_picks=tipsheet_picks,
            total_races=len(race_results),
            races_with_picks=races_with_picks,
            estimated_cost=len(race_numbers) * 0.025
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRACKING ENDPOINTS
# =============================================================================

class OutcomeRequest(BaseModel):
    track: str
    race_number: int
    race_date: str
    results: dict[str, int]  # {horse_name: finishing_position}


@app.get("/stats/summary")
def get_stats_summary():
    """Get overall prediction statistics."""
    return tracker.get_summary()


@app.get("/stats/by-tag")
def get_stats_by_tag():
    """Get performance statistics grouped by tag (e.g., 'Value pick' vs 'The one to beat')."""
    return tracker.get_stats_by_tag()


@app.get("/stats/by-confidence")
def get_stats_by_confidence():
    """Get performance statistics grouped by confidence level."""
    return tracker.get_stats_by_confidence()


@app.get("/stats/by-race-confidence")
def get_stats_by_race_confidence():
    """Get performance statistics grouped by race-level confidence."""
    return tracker.get_stats_by_race_confidence()


@app.get("/stats/by-mode")
def get_stats_by_mode():
    """Get performance statistics grouped by mode (normal vs promo_bonus) and pick type."""
    return tracker.get_stats_by_mode()


@app.get("/stats/by-tag-staking")
def get_stats_by_tag_with_staking():
    """
    Get performance statistics grouped by tag with staking calculations.

    Returns for each tag:
    - total, wins, places, win_rate, place_rate
    - flat_bet: 1u each horse ROI
    - fixed_return: $100 target return per bet ROI
    - each_way: 1u win + 2u place ROI (best for "Each-way chance" tag)
    """
    return tracker.get_stats_by_tag_with_staking(min_samples=1)


@app.get("/stats/by-meeting")
def get_stats_by_meeting():
    """
    Get performance statistics grouped by meeting (track + date).

    Returns a list of meetings with:
    - track, date
    - total_picks, wins, places, win_rate, place_rate
    - flat_profit (1u per pick)
    - by_tag: breakdown by tag with wins/places/profit for each
    """
    return tracker.get_stats_by_meeting()


@app.get("/stats/by-tipsheet")
def get_stats_by_tipsheet():
    """
    Get performance statistics comparing tipsheet_pick=true vs false.

    Returns dict with:
    - tipsheet: stats for picks where Claude would genuinely bet
    - non_tipsheet: stats for other picks
    """
    return tracker.get_stats_by_tipsheet()


@app.get("/stats/by-tag-tipsheet")
def get_stats_by_tag_and_tipsheet():
    """
    Get performance statistics grouped by tag, split by tipsheet_pick.

    Returns dict of tag -> {starred: {...}, regular: {...}}
    Each contains: total, wins, places, win_rate, place_rate, avg_odds, profit, roi
    """
    return tracker.get_stats_by_tag_and_tipsheet()


@app.get("/stats/by-day")
def get_stats_by_day():
    """
    Get performance statistics grouped by day (aggregated across all tracks).

    Returns a list of days with:
    - date, tracks[], total_picks, wins, places
    - win_rate, place_rate, flat_profit
    - starred_picks, by_tag breakdown
    """
    return tracker.get_stats_by_day()


@app.get("/stats/by-class")
def get_stats_by_class(tag: Optional[str] = None):
    """
    Get performance statistics grouped by race class.

    Args:
        tag: Optional filter by tag (e.g., "The one to beat")

    Returns dict of race_class -> stats with:
    - total, wins, places
    - win_rate, place_rate (percentages)
    - avg_odds, flat_profit, roi

    Classes are normalized to groups like:
    - Maiden
    - Class 1-3, Class 4-6
    - BM45-58, BM58-72, BM72-85, BM85+
    - Group 1, Group 2, Group 3, Listed
    """
    return tracker.get_stats_by_class(tag=tag)


@app.get("/stats/by-pfai-rank")
def get_stats_by_pfai_rank(tag: Optional[str] = None):
    """
    Get performance statistics grouped by PFAI rank.

    Args:
        tag: Optional filter by tag (e.g., "The one to beat")

    Returns dict of pfai_rank -> stats with:
    - total, wins, places
    - win_rate, place_rate (percentages)
    - avg_odds, flat_profit, roi
    """
    return tracker.get_stats_by_pfai_rank(tag=tag)


@app.get("/stats/by-tag-pfai")
def get_stats_by_tag_and_pfai(pfai_rank: int = 1):
    """
    Get performance of each tag filtered to a specific PFAI rank.

    This shows how "The one to beat" picks perform when they're also
    PFAI Rank 1 (consensus picks where both AIs agree).

    Args:
        pfai_rank: PFAI rank to filter (default 1 = PFAI's top pick)

    Returns dict of tag -> stats for picks where pfai_rank matches.
    """
    return tracker.get_stats_by_tag_and_pfai(pfai_rank=pfai_rank)


@app.get("/stats/consensus")
def get_consensus_picks_stats():
    """
    Get performance of "The one to beat" picks that are also PFAI Rank 1.

    These are "consensus" picks where both AIs agree - potential tipsheet picks.

    Returns stats for:
    - consensus: "The one to beat" + PFAI Rank 1
    - our_ai_only: "The one to beat" but NOT PFAI Rank 1
    - pfai_only: PFAI Rank 1 but NOT "The one to beat"
    """
    return tracker.get_consensus_picks_stats()


@app.get("/stats/by-metro")
def get_stats_by_metro(tag: Optional[str] = None):
    """
    Get performance split by metro vs non-metro tracks.

    Args:
        tag: Optional tag filter (e.g., "The one to beat")

    Returns dict with:
    - metro: stats for metro tracks (Sydney, Melbourne, Brisbane, Perth, Adelaide)
    - non_metro: stats for country/provincial tracks
    """
    return tracker.get_stats_by_metro(tag=tag)


@app.get("/stats/by-odds")
def get_stats_by_odds(tag: Optional[str] = None, starred_only: bool = False):
    """
    Get performance split by odds range.

    Args:
        tag: Optional tag filter (e.g., "The one to beat")
        starred_only: If True, only include tipsheet_pick=1

    Returns dict of odds_range -> stats
    """
    return tracker.get_stats_by_odds_range(tag=tag, starred_only=starred_only)


@app.get("/picks/by-day")
def get_picks_for_day(race_date: str):
    """
    Get all individual picks for a specific day.

    Args:
        race_date: Date in dd-MMM-yyyy format (e.g., "04-Feb-2026")

    Returns:
        List of picks with track, race, horse, odds, result
    """
    validate_date(race_date)
    return tracker.get_picks_for_day(race_date)


@app.get("/tracking/health")
def tracking_health():
    """
    Health check for the tracking system.
    Returns database status, file location, and recent activity.
    """
    import sqlite3
    import os
    from datetime import datetime

    db_path = tracker.db_path

    result = {
        "status": "unknown",
        "db_path": db_path,
        "db_exists": os.path.exists(db_path),
        "db_size_bytes": 0,
        "total_predictions": 0,
        "recent_predictions": [],
        "last_stored": None,
        "error": None,
    }

    try:
        if os.path.exists(db_path):
            result["db_size_bytes"] = os.path.getsize(db_path)

            # Test connection and get counts
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            result["total_predictions"] = cursor.fetchone()[0]

            # Last 5 predictions (most recent first)
            cursor.execute("""
                SELECT track, race_number, race_date, horse, tag, timestamp
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            recent = cursor.fetchall()
            result["recent_predictions"] = [
                {
                    "track": r[0],
                    "race_number": r[1],
                    "race_date": r[2],
                    "horse": r[3],
                    "tag": r[4],
                    "timestamp": r[5]
                }
                for r in recent
            ]

            if recent:
                result["last_stored"] = result["recent_predictions"][0]["timestamp"]

            conn.close()
            result["status"] = "healthy"
        else:
            result["status"] = "no_database"
            result["error"] = f"Database file not found at {db_path}"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


@app.delete("/tracking/clear")
def clear_tracking():
    """
    Delete all prediction records from the tracking database.
    WARNING: This permanently deletes all tracking data!
    Use this to reset and start fresh with new predictor version.
    """
    count = tracker.clear_all()
    return {"success": True, "deleted_count": count, "message": f"Deleted {count} prediction records"}


@app.post("/fix-place-odds")
def fix_place_odds():
    """
    Fix place odds in database by fetching real Ladbrokes place odds.

    This corrects place_odds that were previously estimated with a formula
    by replacing them with actual Ladbrokes fixed_place prices.
    """
    from core.backtest import get_ladbrokes_place_odds
    from core.normalize import normalize_horse_name
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent / "data" / "predictions.db"

    if not db_path.exists():
        return {"fixed": 0, "error": "Database not found"}

    fixed_count = 0
    errors = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Get all unique track/race/date combinations
        races = conn.execute("""
            SELECT DISTINCT track, race_number, race_date
            FROM predictions
            WHERE mode = 'backtest'
        """).fetchall()

        for race in races:
            track = race['track']
            race_num = race['race_number']
            date = race['race_date']

            try:
                # Fetch real Ladbrokes place odds
                lb_place_odds = get_ladbrokes_place_odds(track, race_num, date)

                if not lb_place_odds:
                    errors.append(f"No Ladbrokes data for {track} R{race_num} {date}")
                    continue

                # Get predictions for this race
                predictions = conn.execute("""
                    SELECT id, horse, place_odds
                    FROM predictions
                    WHERE track = ? AND race_number = ? AND race_date = ? AND mode = 'backtest'
                """, (track, race_num, date)).fetchall()

                for pred in predictions:
                    horse = pred['horse']
                    old_place = pred['place_odds']
                    normalized = normalize_horse_name(horse)

                    if normalized in lb_place_odds:
                        new_place = lb_place_odds[normalized]

                        if abs(new_place - old_place) > 0.01:  # Only update if different
                            conn.execute("""
                                UPDATE predictions
                                SET place_odds = ?
                                WHERE id = ?
                            """, (new_place, pred['id']))
                            fixed_count += 1

            except Exception as e:
                errors.append(f"Error processing {track} R{race_num}: {str(e)}")

        conn.commit()

    return {
        "fixed": fixed_count,
        "races_processed": len(races),
        "errors": errors if errors else None
    }


@app.get("/predictions/pending")
def get_pending_outcomes(race_date: Optional[str] = None):
    """Get predictions that haven't had outcomes recorded yet."""
    return tracker.get_pending_outcomes(race_date)


@app.get("/predictions/recent")
def get_recent_predictions(limit: int = 50):
    """Get recent predictions for display."""
    return tracker.get_recent_predictions(limit)


@app.post("/outcomes")
def record_outcomes(req: OutcomeRequest):
    """
    Record race outcomes for predictions.

    Args:
        track: Track name
        race_number: Race number
        race_date: Race date in format dd-MMM-yyyy
        results: Dict mapping horse names to finishing positions

    Returns:
        Number of predictions updated
    """
    count = tracker.record_outcomes_bulk(
        req.track,
        req.race_number,
        req.race_date,
        req.results
    )
    return {"updated": count, "message": f"Recorded {count} outcomes"}


@app.post("/outcomes/sync")
def sync_outcomes(race_date: str):
    """
    Auto-fetch results from PuntingForm for all pending predictions on a date.

    Call this after races finish to automatically update outcomes.

    Args:
        race_date: Date in format dd-MMM-yyyy

    Returns:
        Summary of synced outcomes
    """
    validate_date(race_date)

    # Get ALL pending predictions (ignore stored dates due to timezone issues)
    all_pending = tracker.get_pending_outcomes()
    if not all_pending:
        return {
            "synced": 0,
            "message": "No pending predictions",
            "debug": {"requested_date": race_date, "total_pending": 0}
        }

    # Group by track (from all pending, not filtered by date)
    tracks = {}
    for p in all_pending:
        track = p["track"]
        if track not in tracks:
            tracks[track] = set()
        tracks[track].add(p["race_number"])

    # Helper to parse and adjust dates
    def adjust_date(date_str: str, days: int) -> str:
        """Add/subtract days from dd-MMM-yyyy date string."""
        from datetime import datetime, timedelta
        months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        parts = date_str.split('-')
        dt = datetime(int(parts[2]), months[parts[1]], int(parts[0]))
        dt = dt + timedelta(days=days)
        return f"{dt.day:02d}-{month_names[dt.month-1]}-{dt.year}"

    def find_meeting(track: str, date_str: str):
        """Find meeting ID for track, returns (meeting_id, actual_date) or (None, None)."""
        meetings = pf_api.get_meetings(date_str)
        for m in meetings:
            m_track = m.get("track", {}).get("name", "")
            if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                return m.get("meetingId"), date_str
        return None, None

    # Fetch results for each track
    synced = 0
    errors = []

    for track, race_numbers in tracks.items():
        try:
            # Try to find meeting - check original date, then +/- 1-2 days
            # (handles timezone issues where stored date might be off)
            meeting_id = None
            actual_date = race_date

            # Try original date first, then adjacent dates
            for day_offset in [0, 1, -1, 2, -2]:
                check_date = adjust_date(race_date, day_offset) if day_offset != 0 else race_date
                meeting_id, actual_date = find_meeting(track, check_date)
                if meeting_id:
                    break

            if not meeting_id:
                errors.append(f"Track not found: {track}")
                continue

            # Fetch results for this meeting
            results_data = pf_api.get_results(meeting_id, 0)  # 0 = all races

            # PuntingForm returns [{meetingId, raceResults: [...]}]
            # Extract the actual race results
            races = []
            if results_data and len(results_data) > 0:
                meeting_data = results_data[0]
                races = meeting_data.get("raceResults", [])

            # Process each race
            for race in races:
                race_num = race.get("raceNumber")
                if race_num not in race_numbers:
                    continue

                # Build results dict {horse_name: position}
                results = {}
                for runner in race.get("runners", []):
                    position = runner.get("position", 0)
                    horse_name = runner.get("runner", "")
                    if position > 0 and horse_name:  # 0 = scratched
                        results[horse_name] = position

                if results:
                    # Try to record with actual date first (where meeting was found)
                    # Then try original stored date as fallback
                    count = tracker.record_outcomes_bulk(
                        track, race_num, actual_date, results
                    )
                    if count == 0 and actual_date != race_date:
                        # Try original date if actual date didn't match any predictions
                        count = tracker.record_outcomes_bulk(
                            track, race_num, race_date, results
                        )
                    synced += count

        except Exception as e:
            errors.append(f"{track}: {str(e)}")

    return {
        "synced": synced,
        "errors": errors if errors else None,
        "message": f"Synced {synced} prediction outcomes",
        "debug": {
            "tracks_processed": list(tracks.keys()),
            "pending_count": len(all_pending)
        }
    }


@app.post("/outcomes/sync/all")
def sync_all_outcomes():
    """
    Auto-fetch results for ALL dates with pending predictions.

    Convenience endpoint that:
    1. Gets all unique dates from pending predictions
    2. Syncs each date sequentially
    3. Returns summary of all syncs

    Returns:
        Summary with dates_synced, total_updated, per-date details
    """
    # Get all pending predictions
    all_pending = tracker.get_pending_outcomes()
    if not all_pending:
        return {
            "success": True,
            "dates_synced": 0,
            "total_updated": 0,
            "message": "No pending predictions",
            "details": []
        }

    # Extract unique dates
    dates = set()
    for p in all_pending:
        if p.get("race_date"):
            dates.add(p["race_date"])

    dates_list = sorted(dates)

    # Sync each date
    details = []
    total_updated = 0

    for date in dates_list:
        try:
            # Call the existing sync function directly
            result = sync_outcomes(date)

            synced = result.get("synced", 0)
            total_updated += synced

            details.append({
                "date": date,
                "synced": synced,
                "errors": result.get("errors")
            })
        except Exception as e:
            details.append({
                "date": date,
                "synced": 0,
                "errors": [str(e)]
            })

    return {
        "success": True,
        "dates_synced": len(dates_list),
        "total_updated": total_updated,
        "message": f"Synced {total_updated} predictions across {len(dates_list)} dates",
        "details": details
    }


# =============================================================================
# BACKTEST ENDPOINT
# =============================================================================

@app.post("/backtest", response_model=BacktestResponse)
def run_backtest_endpoint(req: BacktestRequest):
    """
    Run backtest on historical races.

    This endpoint:
    1. Runs predictions on finished races using Starting Prices (SP)
    2. Stores predictions in the tracking database
    3. Auto-syncs actual race outcomes from PuntingForm
    4. Returns results and performance stats by tag

    Args:
        track: Track name (e.g., "Canterbury")
        date: Date in dd-MMM-yyyy format (e.g., "16-Jan-2026")
        race_start: First race number (default 1)
        race_end: Last race number (default 12)
        auto_sync: Whether to auto-sync outcomes (default True)

    Returns:
        Backtest results with contenders, outcomes, and stats
    """
    from datetime import datetime
    import sqlite3
    from pathlib import Path

    try:
        # Run backtests
        results = run_backtest_meeting(
            track=req.track,
            date=req.date,
            race_start=req.race_start,
            race_end=req.race_end
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No races found at {req.track} on {req.date}"
            )

        race_results = []
        total_contenders = 0
        outcomes_synced = 0

        # Process each race result
        for result in results:
            contenders_response = []

            for c in result.contenders:
                contenders_response.append(BacktestContenderResponse(
                    horse=c.horse,
                    tab_no=c.tab_no,
                    odds=c.odds,
                    place_odds=c.place_odds,
                    tag=c.tag,
                    analysis=c.analysis
                ))
                total_contenders += 1

            # Store predictions in tracking DB
            if result.contenders:
                db_path = Path(__file__).parent / "data" / "predictions.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)

                with sqlite3.connect(db_path) as conn:
                    for c in result.contenders:
                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO predictions
                                (timestamp, track, race_number, race_date, horse, tab_no,
                                 odds, place_odds, tag, confidence, race_confidence,
                                 confidence_reason, mode, pick_type, analysis)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                datetime.now().isoformat(),
                                result.track,
                                result.race_number,
                                result.date,
                                c.horse,
                                c.tab_no,
                                c.odds,
                                c.place_odds,
                                c.tag,
                                5,  # Default confidence
                                5,  # Default race confidence
                                "",
                                "backtest",  # Mark as backtest
                                "contender",
                                c.analysis,
                            ))
                        except Exception as e:
                            print(f"Warning: Failed to store prediction: {e}")
                    conn.commit()

            # Get actual race results if auto_sync enabled
            actual_results = None
            if req.auto_sync:
                actual_results = get_race_results(result.track, result.race_number, result.date)
                if actual_results:
                    # Determine place paying positions based on field size
                    # Australian rules: 8+ runners = 1st/2nd/3rd pay place
                    #                   5-7 runners = 1st/2nd only
                    actual_field_size = len(actual_results)

                    # Record outcomes for our contenders
                    for c in result.contenders:
                        # Find the horse in results (fuzzy match)
                        for horse_name, position in actual_results.items():
                            if horse_name.lower() == c.horse.lower() or \
                               c.horse.lower() in horse_name.lower() or \
                               horse_name.lower() in c.horse.lower():
                                won = position == 1
                                # Place only pays 3rd if 8+ runners
                                if actual_field_size >= 8:
                                    placed = position <= 3
                                else:
                                    placed = position <= 2  # Only 1st/2nd pay place
                                if tracker.record_outcome(
                                    result.track, result.race_number, result.date,
                                    c.horse, won, placed, position
                                ):
                                    outcomes_synced += 1
                                break

            race_results.append(BacktestRaceResult(
                track=result.track,
                race_number=result.race_number,
                date=result.date,
                race_name=result.race_name,
                distance=result.distance,
                condition=result.condition,
                field_size=result.total_horses,
                horses_with_form=result.horses_with_form,
                scratched_count=result.scratched_count,
                contenders=contenders_response,
                summary=result.summary,
                results=actual_results,
                error=result.error
            ))

        # Get updated stats by tag with staking calculations
        stats = tracker.get_stats_by_tag_with_staking(min_samples=1) if req.auto_sync else None

        return BacktestResponse(
            track=req.track,
            date=req.date,
            races=race_results,
            total_races=len(race_results),
            total_contenders=total_contenders,
            outcomes_synced=outcomes_synced,
            stats=stats
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BackfillPrediction(BaseModel):
    track: str
    race_number: int
    race_date: str  # dd-MMM-yyyy format
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float] = None
    tag: str
    mode: str  # "normal" or "promo_bonus"
    pick_type: str  # "contender", "bonus_bet", "promo_play"
    analysis: str = ""
    confidence: int = 5
    race_confidence: int = 5
    tipsheet_pick: bool = False  # True if Claude would genuinely bet on this
    pfai_rank: Optional[int] = None  # PuntingForm AI rank (1 = best)


class BackfillRequest(BaseModel):
    predictions: list[BackfillPrediction]


@app.post("/backfill/update-tipsheet")
def update_tipsheet_picks(req: BackfillRequest):
    """
    Update tipsheet_pick field for existing predictions.

    Use this after backfill to sync tipsheet picks that were added later.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent / "data" / "predictions.db"

    updated = 0
    with sqlite3.connect(db_path) as conn:
        for pred in req.predictions:
            if pred.tipsheet_pick:
                cursor = conn.execute("""
                    UPDATE predictions
                    SET tipsheet_pick = 1
                    WHERE track = ? AND race_number = ? AND horse = ?
                    AND tipsheet_pick = 0
                """, (pred.track, pred.race_number, pred.horse))
                updated += cursor.rowcount

        conn.commit()

    return {
        "updated": updated,
        "message": f"Updated {updated} predictions with tipsheet_pick=true"
    }


@app.post("/backfill/race-class")
def backfill_race_class():
    """
    Backfill race_class for existing predictions that don't have it.

    Queries PuntingForm API for each unique race and updates the predictions.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent / "data" / "predictions.db"

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    # Get unique races without race_class
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        races = conn.execute("""
            SELECT DISTINCT track, race_number, race_date
            FROM predictions
            WHERE race_class IS NULL
            ORDER BY race_date DESC, track, race_number
        """).fetchall()

    if not races:
        return {"message": "No predictions need backfill", "updated": 0, "failed": 0}

    # Cache for fields data
    fields_cache: dict[str, dict] = {}
    updated_total = 0
    failed_races = []

    for race in races:
        track = race['track']
        race_number = race['race_number']
        race_date = race['race_date']
        cache_key = f"{track}|{race_date}"

        # Fetch meeting data if not cached
        if cache_key not in fields_cache:
            try:
                meetings = pf_api.get_meetings(race_date)
                meeting_id = None
                for m in meetings:
                    if m.get('track', {}).get('name', '').lower() == track.lower():
                        meeting_id = m.get('meetingId')
                        break

                if not meeting_id:
                    failed_races.append(f"{track} R{race_number} ({race_date}): Meeting not found")
                    continue

                fields = pf_api.get_fields(meeting_id)
                fields_cache[cache_key] = fields
            except Exception as e:
                failed_races.append(f"{track} R{race_number} ({race_date}): {str(e)}")
                continue

        fields = fields_cache[cache_key]

        # Find the race class
        race_class = None
        for r in fields.get('races', []):
            if r.get('number') == race_number:
                race_class = r.get('raceClass', '').strip().rstrip(';')
                break

        if not race_class:
            failed_races.append(f"{track} R{race_number} ({race_date}): Race class not found")
            continue

        # Update predictions
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                UPDATE predictions
                SET race_class = ?
                WHERE track = ? AND race_number = ? AND race_date = ?
            """, (race_class, track, race_number, race_date))
            conn.commit()
            updated_total += cursor.rowcount

    return {
        "message": f"Backfilled race_class for {len(races)} races",
        "races_processed": len(races),
        "predictions_updated": updated_total,
        "failed": len(failed_races),
        "failed_details": failed_races[:20]  # Limit to first 20 failures
    }


@app.post("/backfill/pfai-rank")
def backfill_pfai_rank(limit: int = 0):
    """
    Backfill PFAI rank for existing predictions that don't have it.

    Queries PuntingForm API for each unique meeting and updates predictions.

    Args:
        limit: Max meetings to process (0 = all). Use for testing.

    Returns:
        Summary of updated predictions.
    """
    import sqlite3
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed

    db_path = Path(__file__).parent / "data" / "predictions.db"

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    # Get unique meetings without pfai_rank
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        meetings = conn.execute("""
            SELECT DISTINCT track, race_date
            FROM predictions
            WHERE pfai_rank IS NULL
            ORDER BY race_date DESC
        """).fetchall()

    if not meetings:
        return {"message": "No predictions need PFAI rank backfill", "updated": 0}

    meetings_list = [(m['track'], m['race_date']) for m in meetings]
    if limit > 0:
        meetings_list = meetings_list[:limit]

    # Process meetings (can parallelize)
    updated_total = 0
    failed_meetings = []
    meeting_cache: dict[str, int] = {}  # track|date -> meeting_id

    def get_meeting_id(track: str, date: str) -> Optional[int]:
        """Get meeting ID from PuntingForm."""
        cache_key = f"{track}|{date}"
        if cache_key in meeting_cache:
            return meeting_cache[cache_key]

        try:
            meetings_data = pf_api.get_meetings(date)
            for m in meetings_data:
                m_track = m.get('track', {}).get('name', '')
                if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                    meeting_id = m.get('meetingId')
                    meeting_cache[cache_key] = meeting_id
                    return meeting_id
        except Exception:
            pass
        return None

    def process_meeting(track: str, race_date: str) -> dict:
        """Process a single meeting - fetch ratings and update predictions."""
        result = {"track": track, "date": race_date, "updated": 0, "error": None}

        try:
            meeting_id = get_meeting_id(track, race_date)
            if not meeting_id:
                result["error"] = "Meeting not found"
                return result

            # Fetch PFAI ratings
            ratings_data = pf_api.get_ratings(meeting_id)
            if not ratings_data:
                result["error"] = "No ratings data"
                return result

            # Update predictions
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get predictions for this meeting (include race_number for lookup)
                predictions = conn.execute("""
                    SELECT id, tab_no, race_number
                    FROM predictions
                    WHERE track = ? AND race_date = ? AND pfai_rank IS NULL
                """, (track, race_date)).fetchall()

                for pred in predictions:
                    tab_no = pred['tab_no']
                    race_num = pred['race_number']
                    # Ratings keyed by (race_no, tab_no)
                    pfai_rank = ratings_data.get((race_num, tab_no), {}).get('pfai_rank')

                    if pfai_rank is not None:
                        conn.execute("""
                            UPDATE predictions
                            SET pfai_rank = ?
                            WHERE id = ?
                        """, (pfai_rank, pred['id']))
                        result["updated"] += 1

                conn.commit()

        except Exception as e:
            result["error"] = str(e)

        return result

    # Process meetings (parallel for speed)
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_meeting = {
            executor.submit(process_meeting, track, date): (track, date)
            for track, date in meetings_list
        }
        for future in as_completed(future_to_meeting):
            result = future.result()
            results.append(result)
            if result["updated"] > 0:
                updated_total += result["updated"]
            if result["error"]:
                failed_meetings.append(f"{result['track']} ({result['date']}): {result['error']}")

    return {
        "message": f"Backfilled PFAI rank for {len(meetings_list)} meetings",
        "meetings_processed": len(meetings_list),
        "predictions_updated": updated_total,
        "failed": len(failed_meetings),
        "failed_details": failed_meetings[:20],
        "results": results[:20]  # Show first 20 results
    }


@app.post("/backfill")
def backfill_predictions(req: BackfillRequest):
    """
    Backfill predictions from external source (e.g., Prisma database).

    Used to import historical predictions for tracking.
    Duplicates are automatically handled (same track/race/date/horse/mode = one entry).
    """
    import sqlite3
    from datetime import datetime
    from pathlib import Path

    db_path = Path(__file__).parent / "data" / "predictions.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0

    with sqlite3.connect(db_path) as conn:
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                track TEXT NOT NULL,
                race_number INTEGER NOT NULL,
                race_date TEXT NOT NULL,
                horse TEXT NOT NULL,
                tab_no INTEGER NOT NULL,
                odds REAL NOT NULL,
                place_odds REAL,
                tag TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                race_confidence INTEGER NOT NULL,
                confidence_reason TEXT,
                mode TEXT NOT NULL,
                pick_type TEXT NOT NULL,
                analysis TEXT,
                tipsheet_pick INTEGER DEFAULT 0,
                won INTEGER,
                placed INTEGER,
                finishing_position INTEGER,
                outcome_recorded INTEGER DEFAULT 0,
                UNIQUE(track, race_number, race_date, horse, mode, pick_type)
            )
        """)

        # Add tipsheet_pick column if it doesn't exist (for existing DBs)
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN tipsheet_pick INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists

        for pred in req.predictions:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO predictions
                    (timestamp, track, race_number, race_date, horse, tab_no,
                     odds, place_odds, tag, confidence, race_confidence,
                     confidence_reason, mode, pick_type, analysis, tipsheet_pick, pfai_rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    pred.track,
                    pred.race_number,
                    pred.race_date,
                    pred.horse,
                    pred.tab_no,
                    pred.odds,
                    pred.place_odds,
                    pred.tag,
                    pred.confidence,
                    pred.race_confidence,
                    "",
                    pred.mode,
                    pred.pick_type,
                    pred.analysis,
                    1 if pred.tipsheet_pick else 0,
                    pred.pfai_rank,
                ))
                if conn.total_changes > 0:
                    imported += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1

        conn.commit()

    return {
        "imported": imported,
        "skipped": skipped,
        "total": len(req.predictions),
        "message": f"Imported {imported} predictions ({skipped} duplicates skipped)"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
