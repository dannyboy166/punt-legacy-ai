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
from pydantic import BaseModel
from typing import Optional
import os

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

class PredictionRequest(BaseModel):
    track: str
    race_number: int
    date: str  # Format: dd-MMM-yyyy (e.g., "09-Jan-2026")


class Contender(BaseModel):
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float]
    tag: str
    analysis: str


class PredictionResponse(BaseModel):
    track: str
    race_number: int
    race_name: str
    distance: int
    condition: str
    class_: str
    contenders: list[Contender]
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


@app.get("/meetings", response_model=list[MeetingResponse])
def get_meetings(date: str):
    """
    Get all tracks racing on a given date.

    Args:
        date: Date in format dd-MMM-yyyy (e.g., "09-Jan-2026")

    Returns:
        List of tracks with meeting IDs
    """
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

    Returns:
        1-3 contenders with analysis and summary
    """
    try:
        # Get race data
        race_data, error = pipeline.get_race_data(req.track, req.race_number, req.date)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Generate prediction
        result = predictor.predict(race_data)

        # Build response
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

        return PredictionResponse(
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
