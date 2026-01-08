"""
Prediction results and error handling.

Provides structured results that track success/failure reasons,
making it easy to surface clear messages to users.

Usage:
    from core.results import PredictionResult, RaceStatus

    # Success
    result = PredictionResult.success(
        horse="Fast Horse",
        odds=3.50,
        confidence=0.75,
    )

    # Failure with clear reason
    result = PredictionResult.no_odds(
        horse="Fast Horse",
        reason="Ladbrokes doesn't cover Pioneer Park (Alice Springs)",
    )

    # Check and display
    if result.ok:
        print(f"Bet: {result.horse} @ ${result.odds}")
    else:
        print(f"Skipped: {result.message}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from datetime import datetime


class RaceStatus(Enum):
    """Status of a race for prediction purposes."""

    # Success states
    OK = "ok"  # Race can be predicted

    # Data availability issues
    NO_LADBROKES_ODDS = "no_ladbrokes_odds"  # Track not on Ladbrokes
    NO_PUNTINGFORM_DATA = "no_puntingform_data"  # Track not on PuntingForm
    NO_FORM_DATA = "no_form_data"  # No historical form available
    INSUFFICIENT_RUNNERS = "insufficient_runners"  # Not enough runners

    # Track/race issues
    TRACK_NOT_SUPPORTED = "track_not_supported"  # NZ, HK, etc.
    RACE_ABANDONED = "race_abandoned"
    ALL_SCRATCHED = "all_scratched"

    # Data quality issues
    ODDS_UNRELIABLE = "odds_unreliable"  # PF odds don't match LB
    STALE_DATA = "stale_data"  # Data too old

    # Processing errors
    API_ERROR = "api_error"
    UNKNOWN_ERROR = "unknown_error"


# User-friendly messages for each status
STATUS_MESSAGES = {
    RaceStatus.OK: "Race available for prediction",
    RaceStatus.NO_LADBROKES_ODDS: "Live odds not available (track not covered by Ladbrokes)",
    RaceStatus.NO_PUNTINGFORM_DATA: "Form data not available for this track",
    RaceStatus.NO_FORM_DATA: "No historical form data available for runners",
    RaceStatus.INSUFFICIENT_RUNNERS: "Not enough runners with data to make prediction",
    RaceStatus.TRACK_NOT_SUPPORTED: "This track is not currently supported",
    RaceStatus.RACE_ABANDONED: "Race has been abandoned",
    RaceStatus.ALL_SCRATCHED: "All runners have been scratched",
    RaceStatus.ODDS_UNRELIABLE: "Odds data appears unreliable - skipping",
    RaceStatus.STALE_DATA: "Data is too old to make reliable prediction",
    RaceStatus.API_ERROR: "Error fetching data from API",
    RaceStatus.UNKNOWN_ERROR: "An unexpected error occurred",
}


@dataclass
class PredictionResult:
    """
    Result of attempting to generate a prediction.

    Can represent either a successful prediction or a clear reason
    why prediction wasn't possible.
    """

    ok: bool
    status: RaceStatus
    message: str

    # Prediction data (when ok=True)
    horse: Optional[str] = None
    odds: Optional[float] = None
    odds_source: Optional[str] = None  # "ladbrokes" or "puntingform"
    confidence: Optional[float] = None
    stake: Optional[float] = None
    edge: Optional[float] = None

    # Context
    track: Optional[str] = None
    race_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional details for debugging
    details: dict = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        horse: str,
        odds: float,
        track: str,
        race_number: int,
        odds_source: str = "ladbrokes",
        confidence: Optional[float] = None,
        stake: Optional[float] = None,
        edge: Optional[float] = None,
        **details,
    ) -> "PredictionResult":
        """Create a successful prediction result."""
        return cls(
            ok=True,
            status=RaceStatus.OK,
            message=f"Prediction: {horse} @ ${odds:.2f}",
            horse=horse,
            odds=odds,
            odds_source=odds_source,
            confidence=confidence,
            stake=stake,
            edge=edge,
            track=track,
            race_number=race_number,
            details=details,
        )

    @classmethod
    def no_odds(
        cls,
        track: str,
        race_number: int,
        reason: str,
        **details,
    ) -> "PredictionResult":
        """Create result when odds aren't available."""
        return cls(
            ok=False,
            status=RaceStatus.NO_LADBROKES_ODDS,
            message=reason,
            track=track,
            race_number=race_number,
            details=details,
        )

    @classmethod
    def no_form(
        cls,
        track: str,
        race_number: int,
        reason: str,
        **details,
    ) -> "PredictionResult":
        """Create result when form data isn't available."""
        return cls(
            ok=False,
            status=RaceStatus.NO_FORM_DATA,
            message=reason,
            track=track,
            race_number=race_number,
            details=details,
        )

    @classmethod
    def track_not_supported(
        cls,
        track: str,
        race_number: int,
        reason: Optional[str] = None,
        **details,
    ) -> "PredictionResult":
        """Create result for unsupported tracks (NZ, HK, etc.)."""
        msg = reason or f"{track} is not currently supported (no Ladbrokes coverage)"
        return cls(
            ok=False,
            status=RaceStatus.TRACK_NOT_SUPPORTED,
            message=msg,
            track=track,
            race_number=race_number,
            details=details,
        )

    @classmethod
    def api_error(
        cls,
        track: str,
        race_number: int,
        error: str,
        **details,
    ) -> "PredictionResult":
        """Create result for API errors."""
        return cls(
            ok=False,
            status=RaceStatus.API_ERROR,
            message=f"API error: {error}",
            track=track,
            race_number=race_number,
            details={"error": error, **details},
        )

    @classmethod
    def error(
        cls,
        track: str,
        race_number: int,
        error: str,
        status: RaceStatus = RaceStatus.UNKNOWN_ERROR,
        **details,
    ) -> "PredictionResult":
        """Create result for generic errors."""
        return cls(
            ok=False,
            status=status,
            message=error,
            track=track,
            race_number=race_number,
            details=details,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ok": self.ok,
            "status": self.status.value,
            "message": self.message,
            "horse": self.horse,
            "odds": self.odds,
            "odds_source": self.odds_source,
            "confidence": self.confidence,
            "stake": self.stake,
            "edge": self.edge,
            "track": self.track,
            "race_number": self.race_number,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class RaceResult:
    """
    Result of processing a race (may contain multiple predictions or none).
    """

    track: str
    race_number: int
    status: RaceStatus
    message: str
    predictions: list[PredictionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def ok(self) -> bool:
        """True if race was successfully processed."""
        return self.status == RaceStatus.OK

    @property
    def has_predictions(self) -> bool:
        """True if at least one prediction was made."""
        return len(self.predictions) > 0 and any(p.ok for p in self.predictions)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "track": self.track,
            "race_number": self.race_number,
            "status": self.status.value,
            "message": self.message,
            "ok": self.ok,
            "has_predictions": self.has_predictions,
            "predictions": [p.to_dict() for p in self.predictions],
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MeetingResult:
    """
    Result of processing an entire meeting (all races at a track).
    """

    track: str
    date: str
    races: list[RaceResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def ok(self) -> bool:
        """True if at least one race was successfully processed."""
        return any(r.ok for r in self.races)

    @property
    def total_predictions(self) -> int:
        """Count of successful predictions across all races."""
        return sum(
            1 for r in self.races for p in r.predictions if p.ok
        )

    @property
    def skipped_races(self) -> list[RaceResult]:
        """Races that couldn't be processed."""
        return [r for r in self.races if not r.ok]

    def add_race(self, race: RaceResult) -> None:
        """Add a race result."""
        self.races.append(race)

    def add_warning(self, message: str) -> None:
        """Add a meeting-level warning."""
        self.warnings.append(message)

    def summary(self) -> str:
        """Get a summary string."""
        ok_races = sum(1 for r in self.races if r.ok)
        return (
            f"{self.track} {self.date}: "
            f"{self.total_predictions} predictions from {ok_races}/{len(self.races)} races"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "track": self.track,
            "date": self.date,
            "ok": self.ok,
            "total_predictions": self.total_predictions,
            "races": [r.to_dict() for r in self.races],
            "skipped_races": [
                {"race": r.race_number, "reason": r.message}
                for r in self.skipped_races
            ],
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }
