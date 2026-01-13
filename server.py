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
                summary=result.summary
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
                    analysis=c.analysis
                ))

            # Check we got at least one contender with odds
            if not contenders:
                raise HTTPException(
                    status_code=503,
                    detail="Could not generate predictions. Odds may not be available yet."
                )

            return PredictionResponse(
                mode="normal",
                track=race_data.track,
                race_number=race_data.race_number,
                race_name=race_data.race_name,
                distance=race_data.distance,
                condition=race_data.condition,
                class_=race_data.class_,
                contenders=contenders,
                summary=result.summary
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
