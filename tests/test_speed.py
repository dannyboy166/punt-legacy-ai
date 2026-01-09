"""
Tests for speed rating calculations.

Run: python3 -m pytest tests/test_speed.py -v
"""

import pytest
from core.speed import (
    parse_race_time,
    parse_condition_number,
    get_baseline_speed,
    get_condition_multiplier,
    calculate_raw_speed,
    calculate_speed_rating,
    calculate_run_rating,
    calculate_horse_rating,
    calculate_race_ratings,
    get_speed_ranks,
    METRES_PER_LENGTH,
)


# =============================================================================
# TIME PARSING
# =============================================================================

class TestParseRaceTime:
    """Test race time parsing from various formats."""

    def test_float_input(self):
        assert parse_race_time(71.5) == 71.5

    def test_int_input(self):
        assert parse_race_time(71) == 71.0

    def test_zero_returns_none(self):
        assert parse_race_time(0) is None

    def test_negative_returns_none(self):
        assert parse_race_time(-5) is None

    def test_none_returns_none(self):
        assert parse_race_time(None) is None

    def test_mm_ss_format(self):
        # "1:11.50" = 71.5 seconds
        assert parse_race_time("1:11.50") == 71.5

    def test_hh_mm_ss_format(self):
        # "00:01:11.5000000" = 71.5 seconds
        assert parse_race_time("00:01:11.5000000") == 71.5

    def test_string_number(self):
        assert parse_race_time("71.5") == 71.5

    def test_dict_total_seconds(self):
        assert parse_race_time({"totalSeconds": 71.5}) == 71.5

    def test_dict_components(self):
        assert parse_race_time({"minutes": 1, "seconds": 11.5}) == 71.5

    def test_invalid_string(self):
        assert parse_race_time("invalid") is None


# =============================================================================
# CONDITION PARSING
# =============================================================================

class TestParseConditionNumber:
    """Test condition number extraction."""

    def test_g4(self):
        assert parse_condition_number("G4") == 4

    def test_s5(self):
        assert parse_condition_number("S5") == 5

    def test_h8(self):
        assert parse_condition_number("H8") == 8

    def test_good_4(self):
        assert parse_condition_number("Good 4") == 4

    def test_soft_6(self):
        assert parse_condition_number("Soft 6") == 6

    def test_heavy_9(self):
        assert parse_condition_number("Heavy 9") == 9

    def test_text_only_good(self):
        assert parse_condition_number("Good") == 4

    def test_text_only_soft(self):
        assert parse_condition_number("Soft") == 5

    def test_text_only_heavy(self):
        assert parse_condition_number("Heavy") == 8

    def test_empty(self):
        assert parse_condition_number("") is None

    def test_none(self):
        assert parse_condition_number(None) is None


# =============================================================================
# BASELINE SPEED
# =============================================================================

class TestGetBaselineSpeed:
    """Test baseline speed lookup with interpolation."""

    def test_exact_1200m(self):
        # From CSV: 1200m = 16.9105 m/s
        assert get_baseline_speed(1200) == pytest.approx(16.9105, rel=1e-4)

    def test_exact_1600m(self):
        # From CSV: 1600m = 16.4374 m/s
        assert get_baseline_speed(1600) == pytest.approx(16.4374, rel=1e-4)

    def test_interpolation_1150m(self):
        # Between 1100 (17.0669) and 1200 (16.9105)
        # 1150 is halfway, so expect ~16.9887
        speed_1100 = 17.0669
        speed_1200 = 16.9105
        expected = speed_1100 + 0.5 * (speed_1200 - speed_1100)
        assert get_baseline_speed(1150) == pytest.approx(expected, rel=1e-4)

    def test_interpolation_1350m(self):
        # Between 1300 (16.7682) and 1400 (16.6481)
        # 1350 is halfway
        speed_1300 = 16.7682
        speed_1400 = 16.6481
        expected = speed_1300 + 0.5 * (speed_1400 - speed_1300)
        assert get_baseline_speed(1350) == pytest.approx(expected, rel=1e-4)

    def test_clamp_below_min(self):
        # Below 900m should return 900m baseline
        assert get_baseline_speed(800) == get_baseline_speed(900)

    def test_clamp_above_max(self):
        # Above 2400m should return 2400m baseline
        assert get_baseline_speed(2600) == get_baseline_speed(2400)


# =============================================================================
# CONDITION MULTIPLIER
# =============================================================================

class TestGetConditionMultiplier:
    """Test condition multiplier lookup."""

    def test_g4_is_baseline(self):
        assert get_condition_multiplier("G4") == pytest.approx(1.0, rel=1e-4)

    def test_g3_faster(self):
        # G3 should be > 1.0 (faster than G4)
        mult = get_condition_multiplier("G3")
        assert mult > 1.0

    def test_s5_slower(self):
        # S5 should be < 1.0 (slower than G4)
        mult = get_condition_multiplier("S5")
        assert mult < 1.0
        assert mult == pytest.approx(0.990678, rel=1e-4)

    def test_h8_much_slower(self):
        # H8 should be much < 1.0
        mult = get_condition_multiplier("H8")
        assert mult < 0.97
        assert mult == pytest.approx(0.965511, rel=1e-4)

    def test_heavy_10_slowest(self):
        # H10 is slowest
        mult = get_condition_multiplier("H10")
        assert mult < 0.95

    def test_text_format(self):
        # "Good 4" should map to G4
        assert get_condition_multiplier("Good 4") == pytest.approx(1.0, rel=1e-4)

    def test_unknown_defaults_to_g4(self):
        # Unknown condition defaults to 1.0
        assert get_condition_multiplier("Unknown") == pytest.approx(1.0, rel=1e-4)


# =============================================================================
# RAW SPEED CALCULATION
# =============================================================================

class TestCalculateRawSpeed:
    """Test raw speed calculation."""

    def test_winner_speed(self):
        # 1200m in 71.5s = 16.78 m/s
        speed = calculate_raw_speed(1200, 71.5, margin=0, position=1)
        assert speed == pytest.approx(1200 / 71.5, rel=1e-4)

    def test_non_winner_speed(self):
        # 2.5 lengths behind at 1200m
        # winner_speed = 1200/71.5 = 16.78 m/s
        # margin_metres = 2.5 * 2.45 = 6.125m
        # margin_seconds = 6.125 / 16.78 = 0.365s
        # horse_time = 71.5 + 0.365 = 71.865s
        # horse_speed = 1200 / 71.865 = 16.70 m/s
        speed = calculate_raw_speed(1200, 71.5, margin=2.5, position=3)

        winner_speed = 1200 / 71.5
        margin_seconds = (2.5 * METRES_PER_LENGTH) / winner_speed
        expected = 1200 / (71.5 + margin_seconds)

        assert speed == pytest.approx(expected, rel=1e-4)

    def test_invalid_distance(self):
        assert calculate_raw_speed(0, 71.5) is None
        assert calculate_raw_speed(-100, 71.5) is None

    def test_invalid_time(self):
        assert calculate_raw_speed(1200, 0) is None
        assert calculate_raw_speed(1200, -5) is None
        assert calculate_raw_speed(1200, None) is None


# =============================================================================
# SPEED RATING
# =============================================================================

class TestCalculateSpeedRating:
    """Test normalized speed rating calculation."""

    def test_average_run_near_1(self):
        # A run at expected pace should be ~1.0
        # 1200m baseline at G4 = 16.9105 m/s
        # Time for baseline = 1200 / 16.9105 = 70.96s
        baseline_time = 1200 / 16.9105
        rating = calculate_speed_rating(1200, baseline_time, margin=0, position=1, condition="G4")
        assert rating == pytest.approx(1.0, rel=0.01)

    def test_fast_run_above_1(self):
        # Faster than expected = rating > 1.0
        fast_time = 69.0  # Very fast 1200m
        rating = calculate_speed_rating(1200, fast_time, margin=0, position=1, condition="G4")
        assert rating > 1.0

    def test_slow_run_below_1(self):
        # Slower than expected = rating < 1.0
        slow_time = 75.0  # Slow 1200m
        rating = calculate_speed_rating(1200, slow_time, margin=0, position=1, condition="G4")
        assert rating < 1.0

    def test_heavy_track_adjustment(self):
        # Same raw time on heavy track should rate higher
        # because heavy tracks are expected to be slower
        time = 72.0
        rating_g4 = calculate_speed_rating(1200, time, condition="G4")
        rating_h8 = calculate_speed_rating(1200, time, condition="H8")
        assert rating_h8 > rating_g4

    def test_distance_out_of_range(self):
        # Too short
        assert calculate_speed_rating(800, 50.0) is None
        # Too long
        assert calculate_speed_rating(3000, 200.0) is None


# =============================================================================
# RUN RATING FROM FORM DATA
# =============================================================================

class TestCalculateRunRating:
    """Test rating calculation from form data dict."""

    def test_valid_run(self):
        run = {
            "distance": 1200,
            "officialRaceTime": "00:01:11.5000000",
            "margin": 0,
            "position": 1,
            "trackCondition": "G4",
        }
        rating = calculate_run_rating(run)
        assert rating is not None
        assert 0.9 < rating < 1.1

    def test_non_winner(self):
        run = {
            "distance": 1200,
            "officialRaceTime": "00:01:11.5000000",
            "margin": 3.5,
            "position": 4,
            "trackCondition": "G4",
        }
        rating = calculate_run_rating(run)
        assert rating is not None

    def test_barrier_trial_skipped(self):
        run = {
            "distance": 1200,
            "officialRaceTime": "00:01:11.5000000",
            "position": 1,
            "isBarrierTrial": True,
        }
        assert calculate_run_rating(run) is None

    def test_scratched_skipped(self):
        run = {
            "distance": 1200,
            "officialRaceTime": "00:01:11.5000000",
            "position": 0,  # Scratched
        }
        assert calculate_run_rating(run) is None

    def test_dnf_skipped(self):
        run = {
            "distance": 1200,
            "officialRaceTime": "00:01:11.5000000",
            "position": 99,  # DNF
        }
        assert calculate_run_rating(run) is None

    def test_missing_time_skipped(self):
        run = {
            "distance": 1200,
            "position": 1,
        }
        assert calculate_run_rating(run) is None

    def test_prefers_pf_race_time(self):
        # Should use pfRaceTime over officialRaceTime
        run = {
            "distance": 1200,
            "pfRaceTime": "00:01:10.0000000",
            "officialRaceTime": "00:01:12.0000000",
            "position": 1,
            "trackCondition": "G4",
        }
        rating = calculate_run_rating(run)
        # pfRaceTime is faster, so rating should be higher
        expected_speed = 1200 / 70.0
        baseline = 16.9105
        expected_rating = expected_speed / baseline
        assert rating == pytest.approx(expected_rating, rel=0.01)


# =============================================================================
# HORSE RATING FROM FORM HISTORY
# =============================================================================

class TestCalculateHorseRating:
    """Test average rating from form history."""

    def test_single_run(self):
        form = [
            {"distance": 1200, "officialRaceTime": 71.5, "position": 1, "trackCondition": "G4"}
        ]
        rating = calculate_horse_rating(form)
        assert rating is not None

    def test_multiple_runs_averaged(self):
        form = [
            {"distance": 1200, "officialRaceTime": 71.0, "position": 1, "trackCondition": "G4"},
            {"distance": 1200, "officialRaceTime": 72.0, "position": 2, "margin": 1, "trackCondition": "G4"},
            {"distance": 1200, "officialRaceTime": 71.5, "position": 1, "trackCondition": "G4"},
        ]
        rating = calculate_horse_rating(form, num_runs=3)
        assert rating is not None

    def test_respects_num_runs_limit(self):
        form = [
            {"distance": 1200, "officialRaceTime": 70.0, "position": 1, "trackCondition": "G4"},  # Fast
            {"distance": 1200, "officialRaceTime": 75.0, "position": 1, "trackCondition": "G4"},  # Slow
            {"distance": 1200, "officialRaceTime": 75.0, "position": 1, "trackCondition": "G4"},  # Slow
        ]
        # Only use first run
        rating_1 = calculate_horse_rating(form, num_runs=1)
        # Use all 3 runs
        rating_3 = calculate_horse_rating(form, num_runs=3)
        # First run only should have higher rating (faster)
        assert rating_1 > rating_3

    def test_filters_by_distance(self):
        form = [
            {"distance": 1200, "officialRaceTime": 75.0, "position": 1, "trackCondition": "G4"},  # Different distance
            {"distance": 2000, "officialRaceTime": 125.0, "position": 1, "trackCondition": "G4"},  # Too far from 1200
        ]
        # With 20% tolerance, 2000m should be filtered out for a 1200m race
        rating = calculate_horse_rating(form, race_distance=1200, distance_tolerance=0.2, num_runs=5)
        # Should only use the 1200m run
        single_rating = calculate_horse_rating(form[:1], num_runs=1)
        assert rating == pytest.approx(single_rating, rel=0.01)

    def test_filters_by_condition(self):
        form = [
            {"distance": 1200, "officialRaceTime": 71.5, "position": 1, "trackCondition": "H10"},  # Very different
            {"distance": 1200, "officialRaceTime": 71.5, "position": 1, "trackCondition": "G4"},  # Match
        ]
        # With tolerance of 2, H10 (10) vs G4 (4) = 6 difference, should be filtered
        rating = calculate_horse_rating(form, race_condition="G4", condition_tolerance=2, num_runs=5)
        # Should only use the G4 run
        single_rating = calculate_horse_rating(form[1:], num_runs=1)
        assert rating == pytest.approx(single_rating, rel=0.01)

    def test_empty_form_returns_none(self):
        assert calculate_horse_rating([]) is None

    def test_no_valid_runs_returns_none(self):
        form = [
            {"distance": 1200, "position": 0},  # Scratched
            {"distance": 1200, "position": 1, "isBarrierTrial": True},  # Trial
        ]
        assert calculate_horse_rating(form) is None


# =============================================================================
# RACE RATINGS
# =============================================================================

class TestCalculateRaceRatings:
    """Test ratings for all runners in a race."""

    def test_basic_race(self):
        runners = [
            {"runnerId": 1, "forms": [{"distance": 1200, "officialRaceTime": 70.0, "position": 1, "trackCondition": "G4"}]},
            {"runnerId": 2, "forms": [{"distance": 1200, "officialRaceTime": 72.0, "position": 1, "trackCondition": "G4"}]},
            {"runnerId": 3, "forms": [{"distance": 1200, "officialRaceTime": 74.0, "position": 1, "trackCondition": "G4"}]},
        ]
        ratings = calculate_race_ratings(runners, 1200, "G4")

        assert len(ratings) == 3
        # Runner 1 should have highest rating (fastest)
        assert ratings[1] > ratings[2] > ratings[3]

    def test_excludes_unrateable_runners(self):
        runners = [
            {"runnerId": 1, "forms": [{"distance": 1200, "officialRaceTime": 71.5, "position": 1, "trackCondition": "G4"}]},
            {"runnerId": 2, "forms": []},  # No form
        ]
        ratings = calculate_race_ratings(runners, 1200, "G4")

        assert 1 in ratings
        assert 2 not in ratings


# =============================================================================
# SPEED RANKS
# =============================================================================

class TestGetSpeedRanks:
    """Test conversion of ratings to ranks."""

    def test_basic_ranking(self):
        ratings = {1: 1.05, 2: 0.98, 3: 1.02}
        ranks = get_speed_ranks(ratings)

        assert ranks[1] == 1  # Highest rating
        assert ranks[3] == 2
        assert ranks[2] == 3  # Lowest rating

    def test_empty_ratings(self):
        assert get_speed_ranks({}) == {}
