"""
FastAPI Server for Punt Legacy AI Predictor.

Exposes the predictor via HTTP API for the racing-tips-platform frontend.

Run:
    uvicorn server:app --reload --port 8000

Endpoints:
    GET  /meetings?date=09-Jan-2026     - List tracks racing on date
    GET  /races?track=Gosford&date=X    - List races at track
    POST /predict                        - Generate prediction for race
    GET  /health                         - Health check
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
from core.tracking import PredictionTracker
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
    tag: str
    analysis: str
    confidence: int = 5  # 1-10 scale


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
    contenders: list[Contender] = []  # Used in normal mode
    bonus_pick: Optional[PromoBonusPick] = None  # Used in promo_bonus mode
    promo_pick: Optional[PromoBonusPick] = None  # Used in promo_bonus mode
    summary: str
    race_confidence: int = 5  # 1-10 overall confidence
    confidence_reason: str = ""  # Why confidence is high/low


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
        List of tracks with meeting IDs
    """
    validate_date(date)
    try:
        meetings = pf_api.get_meetings(date)
        return [
            MeetingResponse(
                track=m.get("track", {}).get("name", "Unknown"),
                meeting_id=m.get("meetingId", 0),
                race_count=m.get("numberOfRaces", 0)
            )
            for m in meetings
            if m.get("track", {}).get("name")
            and m.get("track", {}).get("country") == "AUS"  # Australian races only
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
        race_data, error = pipeline.get_race_data(req.track, req.race_number, req.date)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Check if odds are available - we need odds to make predictions
        runners_with_odds = sum(1 for r in race_data.runners if r.odds)
        if runners_with_odds == 0:
            raise HTTPException(
                status_code=503,
                detail="Odds not available yet. Please wait for the market to open and try again."
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
            try:
                tracker.store_prediction(result, race_data, req.date)
            except Exception as e:
                print(f"Warning: Failed to store prediction: {e}")

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
                confidence_reason=result.confidence_reason
            )

        else:
            # Normal mode response
            contenders = []
            for c in result.contenders:
                # Get place odds from race data
                place_odds = None
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
                    confidence=c.confidence
                ))

            # Check we got at least one contender with odds
            if not contenders:
                raise HTTPException(
                    status_code=503,
                    detail="Could not generate predictions. Odds may not be available yet."
                )

            # Store prediction for tracking
            try:
                tracker.store_prediction(result, race_data, req.date)
            except Exception as e:
                # Don't fail the request if tracking fails
                print(f"Warning: Failed to store prediction: {e}")

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
                confidence_reason=result.confidence_reason
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

    # Get pending predictions for this date
    pending = tracker.get_pending_outcomes(race_date)
    if not pending:
        # Check if there are any pending at all (debug info)
        all_pending = tracker.get_pending_outcomes()
        unique_dates = set(p["race_date"] for p in all_pending)
        return {
            "synced": 0,
            "message": f"No pending predictions for {race_date}",
            "debug": {
                "requested_date": race_date,
                "total_pending": len(all_pending),
                "available_dates": sorted(list(unique_dates))[:10]  # Show first 10 dates
            }
        }

    # Group by track
    tracks = {}
    for p in pending:
        track = p["track"]
        if track not in tracks:
            tracks[track] = set()
        tracks[track].add(p["race_number"])

    # Fetch results for each track
    synced = 0
    errors = []

    for track, race_numbers in tracks.items():
        try:
            # Find meeting ID for track
            meetings = pf_api.get_meetings(race_date)
            meeting_id = None
            for m in meetings:
                m_track = m.get("track", {}).get("name", "")
                if m_track.lower() == track.lower() or track.lower() in m_track.lower():
                    meeting_id = m.get("meetingId")
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
            "pending_count": len(pending)
        }
    }


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


class BackfillRequest(BaseModel):
    predictions: list[BackfillPrediction]


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
                won INTEGER,
                placed INTEGER,
                finishing_position INTEGER,
                outcome_recorded INTEGER DEFAULT 0,
                UNIQUE(track, race_number, race_date, horse, mode, pick_type)
            )
        """)

        for pred in req.predictions:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO predictions
                    (timestamp, track, race_number, race_date, horse, tab_no,
                     odds, place_odds, tag, confidence, race_confidence,
                     confidence_reason, mode, pick_type, analysis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
