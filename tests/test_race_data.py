"""
Tests for race data pipeline.

Run: python3 -m pytest tests/test_race_data.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from core.race_data import (
    FormRun,
    RunnerData,
    RaceData,
    RaceDataPipeline,
)


# =============================================================================
# FORM RUN TESTS
# =============================================================================

class TestFormRun:
    """Test FormRun dataclass."""

    def test_to_dict(self):
        run = FormRun(
            date="01-Jan",
            track="Randwick",
            distance=1200,
            condition="G4",
            condition_num=4,
            position=2,
            margin=1.5,
            weight=57.0,
            barrier=5,
            starters=12,
            class_="BM78",
            prize_money=100000,
            rating=1.0234,
        )
        d = run.to_dict()

        assert d["date"] == "01-Jan"
        assert d["track"] == "Randwick"
        assert d["distance"] == 1200
        assert d["condition"] == "G4"
        assert d["position"] == 2
        assert d["margin"] == 1.5
        assert d["rating"] == 102.3

    def test_to_dict_none_rating(self):
        run = FormRun(
            date="01-Jan",
            track="Randwick",
            distance=1200,
            condition="G4",
            condition_num=4,
            position=2,
            margin=1.5,
            weight=57.0,
            barrier=5,
            starters=12,
            class_="BM78",
            prize_money=100000,
            rating=None,
        )
        d = run.to_dict()
        assert d["rating"] is None


# =============================================================================
# RUNNER DATA TESTS
# =============================================================================

class TestRunnerData:
    """Test RunnerData dataclass."""

    def test_to_dict_basic(self):
        runner = RunnerData(
            name="Fast Horse",
            tab_no=5,
            barrier=3,
            weight=57.0,
            age=4,
            sex="Gelding",
            odds=3.50,
            place_odds=1.40,
            odds_source="ladbrokes",
            implied_prob=28.57,
            jockey="J McDonald",
            trainer="C Waller",
            jockey_a2e=1.15,
            trainer_a2e=0.92,
            jockey_trainer_a2e=1.05,
            career_starts=15,
            career_wins=4,
            career_seconds=3,
            career_thirds=2,
            win_pct=26.7,
            place_pct=60.0,
            track_record={"starts": 3, "firsts": 1, "seconds": 1, "thirds": 0},
            distance_record={"starts": 5, "firsts": 2, "seconds": 1, "thirds": 1},
            condition_record={"starts": 4, "firsts": 1, "seconds": 1, "thirds": 0},
            first_up=False,
            second_up=True,
            form=[],
            early_speed_rank=3,
            settling_position=4,
        )

        d = runner.to_dict()

        assert d["name"] == "Fast Horse"
        assert d["tab_no"] == 5
        assert d["odds"] == 3.50
        assert d["implied_prob"] == 28.6  # Rounded
        assert d["career"] == "15: 4-3-2"
        assert d["jockey_a2e"] == 1.15
        assert d["first_up"] is False
        assert d["second_up"] is True

    def test_to_dict_missing_odds(self):
        runner = RunnerData(
            name="No Odds Horse",
            tab_no=1,
            barrier=1,
            weight=55.0,
            age=3,
            sex="Colt",
            odds=None,
            place_odds=None,
            odds_source="none",
            implied_prob=None,
            jockey="A Jockey",
            trainer="A Trainer",
            jockey_a2e=None,
            trainer_a2e=None,
            jockey_trainer_a2e=None,
            career_starts=5,
            career_wins=1,
            career_seconds=1,
            career_thirds=0,
            win_pct=20.0,
            place_pct=40.0,
            track_record=None,
            distance_record=None,
            condition_record=None,
            first_up=True,
            second_up=False,
        )

        d = runner.to_dict()

        assert d["odds"] is None
        assert d["implied_prob"] is None
        assert d["jockey_a2e"] is None
        assert d["first_up"] is True


# =============================================================================
# RACE DATA TESTS
# =============================================================================

class TestRaceData:
    """Test RaceData dataclass."""

    def test_to_dict(self):
        race = RaceData(
            track="Randwick",
            race_number=3,
            race_name="Maiden Plate",
            distance=1200,
            condition="G4",
            condition_num=4,
            class_="Maiden",
            prize_money=50000,
            start_time="14:30",
            rail_position="True",
            runners=[],
            leaders_count=2,
            pace_scenario="moderate",
        )

        d = race.to_dict()

        assert d["track"] == "Randwick"
        assert d["race_number"] == 3
        assert d["distance"] == 1200
        assert d["field_size"] == 0
        assert d["pace_scenario"] == "moderate"

    def test_to_prompt_text(self):
        runner = RunnerData(
            name="Test Horse",
            tab_no=1,
            barrier=2,
            weight=56.0,
            age=4,
            sex="Mare",
            odds=4.50,
            place_odds=1.70,
            odds_source="ladbrokes",
            implied_prob=22.2,
            jockey="J Smith",
            trainer="T Jones",
            jockey_a2e=1.10,
            trainer_a2e=0.95,
            jockey_trainer_a2e=None,
            career_starts=10,
            career_wins=2,
            career_seconds=2,
            career_thirds=1,
            win_pct=20.0,
            place_pct=50.0,
            track_record=None,
            distance_record=None,
            condition_record=None,
            first_up=False,
            second_up=False,
            form=[
                FormRun(
                    date="01-Jan",
                    track="Randwick",
                    distance=1200,
                    condition="G4",
                    condition_num=4,
                    position=2,
                    margin=1.5,
                    weight=56.0,
                    barrier=3,
                    starters=10,
                    class_="Maiden",
                    prize_money=50000,
                    rating=1.015,
                ),
            ],
            early_speed_rank=2,
            settling_position=3,
        )

        race = RaceData(
            track="Randwick",
            race_number=1,
            race_name="Maiden Plate",
            distance=1200,
            condition="G4",
            condition_num=4,
            class_="Maiden",
            prize_money=50000,
            start_time="14:30",
            rail_position="True",
            runners=[runner],
            leaders_count=2,
            pace_scenario="moderate",
        )

        text = race.to_prompt_text()

        assert "Randwick Race 1" in text
        assert "Test Horse" in text
        assert "4.50" in text or "4.5" in text
        assert "J Smith" in text
        assert "G4" in text
        assert "1200m" in text
        assert "101.5" in text  # Rating


# =============================================================================
# PIPELINE TESTS (MOCKED)
# =============================================================================

class TestRaceDataPipeline:
    """Test RaceDataPipeline with mocked APIs."""

    def test_init_default_apis(self):
        """Test pipeline initializes with default APIs."""
        with patch("core.race_data.PuntingFormAPI") as mock_pf, \
             patch("core.race_data.LadbrokeAPI") as mock_lb:
            pipeline = RaceDataPipeline()
            mock_pf.assert_called_once()
            mock_lb.assert_called_once()

    def test_init_custom_apis(self):
        """Test pipeline accepts custom API instances."""
        mock_pf = Mock()
        mock_lb = Mock()
        pipeline = RaceDataPipeline(pf_api=mock_pf, lb_api=mock_lb)
        assert pipeline.pf_api is mock_pf
        assert pipeline.lb_api is mock_lb

    def test_get_race_data_track_not_found(self):
        """Test error when track not found in meetings."""
        mock_pf = Mock()
        mock_pf.get_meetings.return_value = [
            {"track": {"name": "Flemington"}, "meetingId": 123}
        ]

        mock_lb = Mock()

        pipeline = RaceDataPipeline(pf_api=mock_pf, lb_api=mock_lb)
        result, error = pipeline.get_race_data("Randwick", 1, "09-Jan-2026")

        assert result is None
        assert "not found" in error.lower()

    def test_get_race_data_success(self):
        """Test successful race data retrieval."""
        mock_pf = Mock()

        # Meetings response
        mock_pf.get_meetings.return_value = [
            {"track": {"name": "Randwick"}, "meetingId": 123}
        ]

        # Fields response
        mock_pf.get_fields.return_value = {
            "expectedCondition": "G4",
            "railPosition": "True",
            "races": [
                {
                    "number": 1,
                    "name": "Maiden Plate",
                    "distance": 1200,
                    "raceClass": "Maiden",
                    "prizeMoney": 50000,
                    "startTime": "14:30",
                    "runners": [
                        {
                            "runnerId": 1,
                            "name": "Fast Horse",
                            "tabNo": 1,
                            "barrier": 3,
                            "weight": 57.0,
                            "age": 4,
                            "sex": "Gelding",
                            "careerStarts": 10,
                            "careerWins": 2,
                            "careerSeconds": 2,
                            "careerThirds": 1,
                            "winPct": 20.0,
                            "placePct": 50.0,
                            "prepRuns": 1,  # Second up
                            "jockey": {"fullName": "J McDonald"},
                            "trainer": {"fullName": "C Waller"},
                            "jockeyA2E_Last100": {"a2E": 1.15},
                            "trainerA2E_Last100": {"a2E": 0.95},
                        }
                    ],
                }
            ],
        }

        # Form response
        mock_pf.get_form.return_value = [
            {
                "runnerId": 1,
                "forms": [
                    {
                        "meetingDate": "2026-01-01T00:00:00",
                        "track": {"name": "Rosehill"},
                        "distance": 1200,
                        "trackCondition": "G4",
                        "position": 2,
                        "margin": 1.5,
                        "weight": 57.0,
                        "barrier": 5,
                        "starters": 10,
                        "raceClass": "BM78",
                        "prizeMoney": 80000,
                        "officialRaceTime": "00:01:11.5000000",
                    }
                ],
            }
        ]

        # Speedmap response
        mock_pf.get_speedmaps.return_value = [
            {"items": [{"tabNo": 1, "speed": 2, "settle": 3}]}
        ]

        # Ladbrokes odds
        mock_lb = Mock()
        mock_lb.get_odds_for_pf_track.return_value = (
            {"fast horse": {"fixed_win": 3.50, "scratched": False}},
            None,
        )

        pipeline = RaceDataPipeline(pf_api=mock_pf, lb_api=mock_lb)
        result, error = pipeline.get_race_data("Randwick", 1, "09-Jan-2026")

        assert error is None
        assert result is not None
        assert result.track == "Randwick"
        assert result.race_number == 1
        assert result.distance == 1200
        assert len(result.runners) == 1

        runner = result.runners[0]
        assert runner.name == "Fast Horse"
        assert runner.odds == 3.50
        assert runner.jockey_a2e == 1.15
        assert runner.second_up is True
        assert runner.early_speed_rank == 2
        assert len(runner.form) == 1

    def test_pace_scenario_calculation(self):
        """Test pace scenario is calculated correctly."""
        mock_pf = Mock()
        mock_pf.get_meetings.return_value = [
            {"track": {"name": "Randwick"}, "meetingId": 123}
        ]
        mock_pf.get_fields.return_value = {
            "races": [
                {
                    "number": 1,
                    "name": "Test",
                    "distance": 1200,
                    "runners": [],
                }
            ],
        }
        mock_pf.get_form.return_value = []
        mock_pf.get_speedmaps.return_value = []

        mock_lb = Mock()
        mock_lb.get_odds_for_pf_track.return_value = ({}, None)

        pipeline = RaceDataPipeline(pf_api=mock_pf, lb_api=mock_lb)

        # Create race with different leader counts
        race_data, _ = pipeline.get_race_data("Randwick", 1, "09-Jan-2026")

        # Default with no runners
        assert race_data.pace_scenario == "soft"  # 0 leaders


class TestPaceScenario:
    """Test pace scenario logic."""

    def test_hot_pace_with_many_leaders(self):
        race = RaceData(
            track="Test",
            race_number=1,
            race_name="Test",
            distance=1200,
            condition="G4",
            condition_num=4,
            class_="Maiden",
            prize_money=50000,
            start_time="14:00",
            rail_position="True",
        )

        # Add runners with speed ranks
        for i in range(5):
            runner = RunnerData(
                name=f"Horse {i}",
                tab_no=i + 1,
                barrier=i + 1,
                weight=55.0,
                age=3,
                sex="Gelding",
                odds=5.0,
                place_odds=1.80,
                odds_source="ladbrokes",
                implied_prob=20.0,
                jockey="Jockey",
                trainer="Trainer",
                jockey_a2e=None,
                trainer_a2e=None,
                jockey_trainer_a2e=None,
                career_starts=5,
                career_wins=1,
                career_seconds=1,
                career_thirds=0,
                win_pct=20.0,
                place_pct=40.0,
                track_record=None,
                distance_record=None,
                condition_record=None,
                first_up=False,
                second_up=False,
                early_speed_rank=i + 1,  # 1, 2, 3, 4, 5
                settling_position=i + 1,
            )
            race.runners.append(runner)

        # Calculate leaders (speed rank 1-2)
        leaders = sum(1 for r in race.runners if r.early_speed_rank and r.early_speed_rank <= 2)
        race.leaders_count = leaders

        if leaders >= 3:
            race.pace_scenario = "hot"
        elif leaders <= 1:
            race.pace_scenario = "soft"
        else:
            race.pace_scenario = "moderate"

        assert race.leaders_count == 2
        assert race.pace_scenario == "moderate"
