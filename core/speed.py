"""
Speed Rating Calculations.

Calculates normalized speed ratings from past race data.
These ratings are used as inputs to the Claude AI predictor.

Key concepts:
- Raw speed: distance / time (m/s)
- Normalized rating: actual_speed / expected_speed
  - Rating > 1.0 = faster than expected
  - Rating < 1.0 = slower than expected

Usage:
    from core.speed import calculate_speed_rating, calculate_horse_rating

    # Single run
    rating = calculate_speed_rating(
        distance=1200,
        winner_time=71.5,
        margin=2.5,
        position=3,
        condition="G4"
    )

    # Horse's average rating from form history
    rating = calculate_horse_rating(form_history, race_distance=1200, race_condition="G4")
"""

import csv
import os
import re
from typing import Optional

# =============================================================================
# CONSTANTS
# =============================================================================

METRES_PER_LENGTH = 2.45  # Standard length measurement

# Distance range for valid calculations
MIN_DISTANCE = 900
MAX_DISTANCE = 2400

# Paths to normalization data
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DISTANCE_CSV = os.path.join(_BASE_DIR, "normalization", "distance.csv")
_CONDITION_CSV = os.path.join(_BASE_DIR, "normalization", "condition.csv")

# Cached normalization data
_distance_baselines: dict[int, float] | None = None
_condition_multipliers: dict[str, float] | None = None


# =============================================================================
# NORMALIZATION DATA LOADING
# =============================================================================

def _load_distance_baselines() -> dict[int, float]:
    """Load distance -> baseline speed (m/s) mapping at G4 condition."""
    global _distance_baselines
    if _distance_baselines is not None:
        return _distance_baselines

    _distance_baselines = {}
    with open(_DISTANCE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            distance = int(row["distance"])
            avg_speed = float(row["avg_speed_mps"])
            _distance_baselines[distance] = avg_speed

    return _distance_baselines


def _load_condition_multipliers() -> dict[str, float]:
    """Load condition -> speed multiplier mapping."""
    global _condition_multipliers
    if _condition_multipliers is not None:
        return _condition_multipliers

    _condition_multipliers = {}
    with open(_CONDITION_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["track_condition"]
            multiplier = float(row["speed_multiplier"])
            _condition_multipliers[condition] = multiplier

    return _condition_multipliers


def get_baseline_speed(distance: int) -> float:
    """
    Get expected speed (m/s) for a distance at G4 condition.

    Uses linear interpolation for distances between fixed points.

    Args:
        distance: Race distance in metres

    Returns:
        Expected speed in m/s

    Example:
        >>> get_baseline_speed(1200)
        16.9105
        >>> get_baseline_speed(1150)  # Interpolates between 1100 and 1200
        16.9887
    """
    baselines = _load_distance_baselines()

    # Exact match
    if distance in baselines:
        return baselines[distance]

    # Get sorted distances for interpolation
    distances = sorted(baselines.keys())

    # Clamp to range
    if distance <= distances[0]:
        return baselines[distances[0]]
    if distance >= distances[-1]:
        return baselines[distances[-1]]

    # Find surrounding distances for interpolation
    lower_dist = max(d for d in distances if d < distance)
    upper_dist = min(d for d in distances if d > distance)

    # Linear interpolation
    lower_speed = baselines[lower_dist]
    upper_speed = baselines[upper_dist]

    ratio = (distance - lower_dist) / (upper_dist - lower_dist)
    return lower_speed + ratio * (upper_speed - lower_speed)


def get_condition_multiplier(condition: str) -> float:
    """
    Get speed multiplier for a track condition.

    Args:
        condition: Track condition string (e.g., "G4", "S5", "Heavy 8")

    Returns:
        Speed multiplier (1.0 = G4 baseline, <1.0 = slower conditions)

    Example:
        >>> get_condition_multiplier("G4")
        1.0
        >>> get_condition_multiplier("H8")
        0.965511
    """
    multipliers = _load_condition_multipliers()

    # Try exact match
    if condition in multipliers:
        return multipliers[condition]

    # Try to extract condition code from various formats
    # e.g., "Good 4" -> "G4", "Soft 5" -> "S5", "Heavy 8" -> "H8"
    condition_map = {
        "Good": "G4",
        "Good 3": "G3",
        "Good 4": "G4",
        "Soft": "S5",
        "Soft 5": "S5",
        "Soft 6": "S6",
        "Soft 7": "S7",
        "Heavy": "H8",
        "Heavy 8": "H8",
        "Heavy 9": "H9",
        "Heavy 10": "H10",
        "Synthetic": "Syn",
    }

    if condition in condition_map:
        mapped = condition_map[condition]
        if mapped in multipliers:
            return multipliers[mapped]

    # Try to parse condition number and map to closest
    cond_num = parse_condition_number(condition)
    if cond_num is not None:
        # Map condition number to standard code
        if cond_num <= 3:
            return multipliers.get("G3", 1.0)
        elif cond_num == 4:
            return multipliers.get("G4", 1.0)
        elif cond_num == 5:
            return multipliers.get("S5", 1.0)
        elif cond_num == 6:
            return multipliers.get("S6", 1.0)
        elif cond_num == 7:
            return multipliers.get("S7", 1.0)
        elif cond_num == 8:
            return multipliers.get("H8", 1.0)
        elif cond_num == 9:
            return multipliers.get("H9", 1.0)
        elif cond_num >= 10:
            return multipliers.get("H10", 1.0)

    # Default to G4 if can't parse
    return 1.0


def parse_condition_number(condition: str) -> Optional[int]:
    """
    Extract numeric value from condition string.

    Args:
        condition: e.g., "G4", "S5", "H8", "Good 4", "5"

    Returns:
        Integer condition number (1-10), or None if can't parse

    Example:
        >>> parse_condition_number("G4")
        4
        >>> parse_condition_number("Heavy 9")
        9
    """
    if not condition:
        return None

    # Try to extract number from end of string
    match = re.search(r"(\d+)$", str(condition))
    if match:
        return int(match.group(1))

    # Map text-only conditions to default numbers
    text_map = {"Good": 4, "Soft": 5, "Heavy": 8, "Synthetic": 5}
    for text, num in text_map.items():
        if text.lower() in str(condition).lower():
            return num

    return None


# =============================================================================
# TIME PARSING
# =============================================================================

def parse_race_time(time_value) -> Optional[float]:
    """
    Parse race time from various API formats to seconds.

    Handles:
    - float/int: Already in seconds
    - string: "1:23.45" or "00:01:23.4500000"
    - dict: Timespan object with hours/minutes/seconds

    Args:
        time_value: Time in any format from API

    Returns:
        Time in seconds, or None if can't parse

    Example:
        >>> parse_race_time(71.5)
        71.5
        >>> parse_race_time("1:11.50")
        71.5
        >>> parse_race_time("00:01:11.5000000")
        71.5
    """
    if time_value is None:
        return None

    # Already a number
    if isinstance(time_value, (int, float)):
        return float(time_value) if time_value > 0 else None

    # String format
    if isinstance(time_value, str):
        time_str = time_value.strip()
        if ":" in time_str:
            parts = time_str.split(":")
            try:
                if len(parts) == 3:
                    # "00:01:23.45" format
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    total = hours * 3600 + minutes * 60 + seconds
                    return total if total > 0 else None
                elif len(parts) == 2:
                    # "1:23.45" format
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
            except (ValueError, IndexError):
                return None
        else:
            # Plain number as string
            try:
                val = float(time_str)
                return val if val > 0 else None
            except ValueError:
                return None

    # Dict format (timespan object)
    if isinstance(time_value, dict):
        # Try totalSeconds first (complete value)
        for key in ["totalSeconds", "TotalSeconds"]:
            if key in time_value:
                val = float(time_value[key])
                return val if val > 0 else None

        # Build from components (hours/minutes/seconds)
        hours = time_value.get("hours", time_value.get("Hours", 0)) or 0
        minutes = time_value.get("minutes", time_value.get("Minutes", 0)) or 0
        seconds = time_value.get("seconds", time_value.get("Seconds", 0)) or 0
        if hours or minutes or seconds:
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    return None


# =============================================================================
# SPEED CALCULATION
# =============================================================================

def calculate_raw_speed(
    distance: int,
    winner_time: float,
    margin: float = 0,
    position: int = 1,
) -> Optional[float]:
    """
    Calculate raw speed (m/s) from race data.

    Uses winner's time and margin to calculate horse's actual time:
        winner_speed = distance / winner_time
        margin_seconds = (margin_lengths × 2.45m) / winner_speed
        horse_time = winner_time + margin_seconds

    Args:
        distance: Race distance in metres
        winner_time: Winner's time in seconds
        margin: Lengths behind winner (0 for winner)
        position: Finishing position (1 for winner)

    Returns:
        Speed in m/s, or None if invalid inputs

    Example:
        >>> calculate_raw_speed(1200, 71.5, margin=0, position=1)  # Winner
        16.78
        >>> calculate_raw_speed(1200, 71.5, margin=2.5, position=3)  # 2.5L behind
        16.54
    """
    if not distance or distance <= 0:
        return None
    if not winner_time or winner_time <= 0:
        return None

    # Winner case
    if position == 1 or margin == 0:
        return distance / winner_time

    # Non-winner: add margin to time
    winner_speed = distance / winner_time
    margin_metres = float(margin) * METRES_PER_LENGTH
    margin_seconds = margin_metres / winner_speed
    horse_time = winner_time + margin_seconds

    if horse_time <= 0:
        return None

    return distance / horse_time


def calculate_speed_rating(
    distance: int,
    winner_time: float,
    margin: float = 0,
    position: int = 1,
    condition: str = "G4",
) -> Optional[float]:
    """
    Calculate normalized speed rating from race data.

    Rating = actual_speed / expected_speed
    - Rating > 1.0 = faster than expected for distance/condition
    - Rating < 1.0 = slower than expected

    Args:
        distance: Race distance in metres
        winner_time: Winner's time in seconds
        margin: Lengths behind winner (0 for winner)
        position: Finishing position (1 for winner)
        condition: Track condition (e.g., "G4", "S5")

    Returns:
        Normalized rating, or None if can't calculate

    Example:
        >>> calculate_speed_rating(1200, 71.5, margin=0, position=1, condition="G4")
        0.992  # Slightly slower than expected for 1200m on Good4
    """
    # Check distance is in valid range
    if not distance or distance < MIN_DISTANCE or distance > MAX_DISTANCE:
        return None

    # Calculate raw speed
    raw_speed = calculate_raw_speed(distance, winner_time, margin, position)
    if raw_speed is None:
        return None

    # Get expected speed for this distance and condition
    baseline = get_baseline_speed(distance)
    multiplier = get_condition_multiplier(condition)
    expected_speed = baseline * multiplier

    # Return normalized rating
    return raw_speed / expected_speed


# =============================================================================
# FORM HISTORY PROCESSING
# =============================================================================

def calculate_run_rating(run: dict) -> Optional[float]:
    """
    Calculate speed rating from a single past run.

    Extracts data from PuntingForm form history format.

    Args:
        run: Past run dict from PuntingForm API

    Returns:
        Normalized rating, or None if can't calculate
    """
    # Skip barrier trials
    if run.get("isBarrierTrial", False):
        return None

    # Skip scratched/DNF (position 0 or 99+ means didn't finish)
    position = run.get("position")
    if position is None or position <= 0 or position >= 90:
        return None

    # Get distance
    distance = run.get("distance")
    if not distance or distance < MIN_DISTANCE or distance > MAX_DISTANCE:
        return None

    # Get winner's time (prefer pfRaceTime, fallback to officialRaceTime)
    winner_time = parse_race_time(run.get("pfRaceTime"))
    if winner_time is None:
        winner_time = parse_race_time(run.get("officialRaceTime"))
    if winner_time is None:
        return None

    # Get margin
    margin = run.get("margin", 0) or 0

    # Get condition
    condition = run.get("trackCondition") or "G4"

    return calculate_speed_rating(distance, winner_time, margin, position, condition)


def calculate_horse_rating(
    form_history: list[dict],
    race_distance: Optional[int] = None,
    race_condition: Optional[str] = None,
    num_runs: int = 3,
    distance_tolerance: float = 0.2,
    condition_tolerance: int = 2,
) -> Optional[float]:
    """
    Calculate average speed rating from a horse's form history.

    Args:
        form_history: List of past runs from PuntingForm API
        race_distance: Today's race distance (for filtering similar runs)
        race_condition: Today's track condition (for filtering similar runs)
        num_runs: Maximum number of recent runs to use
        distance_tolerance: Max distance difference as fraction (0.2 = ±20%)
        condition_tolerance: Max condition number difference (2 = ±2 conditions)

    Returns:
        Average normalized rating, or None if no valid runs

    Example:
        >>> rating = calculate_horse_rating(
        ...     form_history,
        ...     race_distance=1200,
        ...     race_condition="G4",
        ...     num_runs=3
        ... )
        >>> print(f"Rating: {rating:.3f}")
        Rating: 1.012
    """
    ratings = []

    # Pre-calculate distance bounds
    if race_distance and distance_tolerance:
        min_dist = race_distance * (1 - distance_tolerance)
        max_dist = race_distance * (1 + distance_tolerance)
    else:
        min_dist = None
        max_dist = None

    # Pre-calculate condition bounds
    if race_condition and condition_tolerance:
        race_cond_num = parse_condition_number(race_condition)
    else:
        race_cond_num = None

    for run in form_history:
        # Filter by distance similarity
        if min_dist is not None:
            run_distance = run.get("distance")
            if run_distance and (run_distance < min_dist or run_distance > max_dist):
                continue

        # Filter by condition similarity
        if race_cond_num is not None:
            run_condition = run.get("trackCondition") or "G4"
            run_cond_num = parse_condition_number(run_condition)
            if run_cond_num and abs(run_cond_num - race_cond_num) > condition_tolerance:
                continue

        # Calculate rating for this run
        rating = calculate_run_rating(run)
        if rating is not None:
            ratings.append(rating)

        # Stop once we have enough runs
        if len(ratings) >= num_runs:
            break

    if not ratings:
        return None

    return sum(ratings) / len(ratings)


def calculate_race_ratings(
    runners: list[dict],
    race_distance: int,
    race_condition: str,
    num_runs: int = 3,
) -> dict[int, float]:
    """
    Calculate speed ratings for all runners in a race.

    Args:
        runners: List of runner dicts with 'runnerId' and 'forms'
        race_distance: Today's race distance
        race_condition: Today's track condition
        num_runs: Number of recent runs to use per horse

    Returns:
        Dict mapping runner_id -> rating (only includes runners with valid ratings)

    Example:
        >>> ratings = calculate_race_ratings(runners, 1200, "G4")
        >>> for runner_id, rating in sorted(ratings.items(), key=lambda x: -x[1]):
        ...     print(f"Runner {runner_id}: {rating:.3f}")
    """
    ratings = {}

    for runner in runners:
        runner_id = runner.get("runnerId")
        form_history = runner.get("forms") or []

        rating = calculate_horse_rating(
            form_history,
            race_distance=race_distance,
            race_condition=race_condition,
            num_runs=num_runs,
        )

        if rating is not None:
            ratings[runner_id] = rating

    return ratings


def get_speed_ranks(ratings: dict[int, float]) -> dict[int, int]:
    """
    Convert ratings to ranks (1 = highest rated).

    Args:
        ratings: Dict mapping runner_id -> rating

    Returns:
        Dict mapping runner_id -> rank
    """
    if not ratings:
        return {}

    sorted_runners = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    return {runner_id: rank for rank, (runner_id, _) in enumerate(sorted_runners, 1)}
