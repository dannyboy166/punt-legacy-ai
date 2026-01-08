"""
Tests for prediction results and error handling.

Run with: python -m pytest tests/test_results.py -v
"""

import pytest
from core.results import (
    PredictionResult,
    RaceResult,
    MeetingResult,
    RaceStatus,
    STATUS_MESSAGES,
)


class TestPredictionResult:
    """Tests for PredictionResult."""

    def test_success(self):
        """Test creating successful prediction."""
        result = PredictionResult.success(
            horse="Fast Horse",
            odds=3.50,
            track="Randwick",
            race_number=1,
            confidence=0.75,
            stake=2.0,
        )

        assert result.ok is True
        assert result.status == RaceStatus.OK
        assert result.horse == "Fast Horse"
        assert result.odds == 3.50
        assert result.confidence == 0.75
        assert "Fast Horse" in result.message

    def test_no_odds(self):
        """Test creating no-odds result."""
        result = PredictionResult.no_odds(
            track="Pioneer Park",
            race_number=1,
            reason="Ladbrokes doesn't cover Pioneer Park (Alice Springs)",
        )

        assert result.ok is False
        assert result.status == RaceStatus.NO_LADBROKES_ODDS
        assert "Ladbrokes" in result.message
        assert result.track == "Pioneer Park"

    def test_track_not_supported(self):
        """Test creating unsupported track result."""
        result = PredictionResult.track_not_supported(
            track="Tauranga",
            race_number=3,
        )

        assert result.ok is False
        assert result.status == RaceStatus.TRACK_NOT_SUPPORTED
        assert "Tauranga" in result.message

    def test_api_error(self):
        """Test creating API error result."""
        result = PredictionResult.api_error(
            track="Randwick",
            race_number=1,
            error="Connection timeout",
        )

        assert result.ok is False
        assert result.status == RaceStatus.API_ERROR
        assert "Connection timeout" in result.message

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = PredictionResult.success(
            horse="Fast Horse",
            odds=3.50,
            track="Randwick",
            race_number=1,
        )

        data = result.to_dict()

        assert data["ok"] is True
        assert data["status"] == "ok"
        assert data["horse"] == "Fast Horse"
        assert data["odds"] == 3.50
        assert "timestamp" in data


class TestRaceResult:
    """Tests for RaceResult."""

    def test_ok_race(self):
        """Test race with predictions."""
        race = RaceResult(
            track="Randwick",
            race_number=1,
            status=RaceStatus.OK,
            message="Race processed",
        )

        prediction = PredictionResult.success(
            horse="Fast Horse",
            odds=3.50,
            track="Randwick",
            race_number=1,
        )
        race.predictions.append(prediction)

        assert race.ok is True
        assert race.has_predictions is True

    def test_skipped_race(self):
        """Test race that was skipped."""
        race = RaceResult(
            track="Tauranga",
            race_number=1,
            status=RaceStatus.TRACK_NOT_SUPPORTED,
            message="NZ tracks not supported",
        )

        assert race.ok is False
        assert race.has_predictions is False

    def test_warnings(self):
        """Test adding warnings."""
        race = RaceResult(
            track="Randwick",
            race_number=1,
            status=RaceStatus.OK,
            message="Race processed",
        )
        race.add_warning("2 horses have unreliable odds")
        race.add_warning("Wet track conditions")

        assert len(race.warnings) == 2

    def test_to_dict(self):
        """Test dictionary serialization."""
        race = RaceResult(
            track="Randwick",
            race_number=1,
            status=RaceStatus.OK,
            message="Race processed",
        )

        data = race.to_dict()

        assert data["track"] == "Randwick"
        assert data["race_number"] == 1
        assert data["status"] == "ok"


class TestMeetingResult:
    """Tests for MeetingResult."""

    def test_meeting_with_races(self):
        """Test meeting with multiple races."""
        meeting = MeetingResult(track="Randwick", date="08-Jan-2026")

        # Add successful race
        race1 = RaceResult(
            track="Randwick",
            race_number=1,
            status=RaceStatus.OK,
            message="OK",
        )
        race1.predictions.append(
            PredictionResult.success(
                horse="Horse 1",
                odds=3.0,
                track="Randwick",
                race_number=1,
            )
        )
        meeting.add_race(race1)

        # Add skipped race
        race2 = RaceResult(
            track="Randwick",
            race_number=2,
            status=RaceStatus.INSUFFICIENT_RUNNERS,
            message="Only 2 runners",
        )
        meeting.add_race(race2)

        assert meeting.ok is True
        assert meeting.total_predictions == 1
        assert len(meeting.skipped_races) == 1

    def test_meeting_all_skipped(self):
        """Test meeting where all races were skipped."""
        meeting = MeetingResult(track="Tauranga", date="08-Jan-2026")

        race = RaceResult(
            track="Tauranga",
            race_number=1,
            status=RaceStatus.TRACK_NOT_SUPPORTED,
            message="NZ not supported",
        )
        meeting.add_race(race)

        assert meeting.ok is False
        assert meeting.total_predictions == 0

    def test_summary(self):
        """Test summary generation."""
        meeting = MeetingResult(track="Randwick", date="08-Jan-2026")

        race = RaceResult(
            track="Randwick",
            race_number=1,
            status=RaceStatus.OK,
            message="OK",
        )
        race.predictions.append(
            PredictionResult.success(
                horse="Fast Horse",
                odds=3.0,
                track="Randwick",
                race_number=1,
            )
        )
        meeting.add_race(race)

        summary = meeting.summary()
        assert "Randwick" in summary
        assert "1 predictions" in summary


class TestStatusMessages:
    """Tests for status messages."""

    def test_all_statuses_have_messages(self):
        """Every status should have a user-friendly message."""
        for status in RaceStatus:
            assert status in STATUS_MESSAGES
            assert len(STATUS_MESSAGES[status]) > 0


class TestRealWorldScenarios:
    """Test real-world usage patterns."""

    def test_nz_track_flow(self):
        """
        Scenario: User requests predictions for Tauranga (NZ).
        System should clearly explain why it can't provide odds.
        """
        track = "Tauranga"
        race_number = 1

        # System detects NZ track
        result = PredictionResult.track_not_supported(
            track=track,
            race_number=race_number,
            reason="Tauranga (NZ) is not supported - Ladbrokes doesn't cover NZ racing",
        )

        assert result.ok is False
        assert "NZ" in result.message or "Tauranga" in result.message
        # Frontend can display: result.message

    def test_odds_unavailable_flow(self):
        """
        Scenario: PuntingForm has data but Ladbrokes doesn't have odds yet.
        """
        result = PredictionResult.no_odds(
            track="Randwick",
            race_number=1,
            reason="Ladbrokes odds not yet available - markets may not be open",
        )

        assert result.ok is False
        assert result.status == RaceStatus.NO_LADBROKES_ODDS
        # Frontend can show: "Check back later - markets not open"

    def test_successful_prediction_display(self):
        """
        Scenario: Successful prediction ready for display.
        """
        result = PredictionResult.success(
            horse="Fast Horse",
            odds=3.50,
            track="Randwick",
            race_number=5,
            odds_source="ladbrokes",
            confidence=0.82,
            stake=1.5,
            edge=0.15,
        )

        # Frontend can display all details
        assert result.horse == "Fast Horse"
        assert result.odds == 3.50
        assert result.odds_source == "ladbrokes"
        assert result.confidence == 0.82
        assert result.stake == 1.5

        # Can serialize to JSON for API response
        data = result.to_dict()
        assert data["ok"] is True
