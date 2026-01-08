"""
Horse name normalization utilities.

Provides consistent name matching across different data sources (PuntingForm, Ladbrokes).
"""

import re
from typing import Optional


def normalize_horse_name(name: Optional[str]) -> str:
    """
    Normalize horse name for matching across APIs.

    Handles:
    - Case differences ("FAST HORSE" vs "Fast Horse")
    - Apostrophes ("O'Brien" vs "OBrien")
    - Special characters ("D'Artagnan (NZ)" vs "Dartagnan NZ")
    - Extra whitespace ("Fast  Horse" vs "Fast Horse")

    Args:
        name: Horse name from any source

    Returns:
        Normalized lowercase name with special chars removed

    Examples:
        >>> normalize_horse_name("O'Brien's Pride")
        'obriens pride'
        >>> normalize_horse_name("FAST HORSE")
        'fast horse'
        >>> normalize_horse_name("D'Artagnan (NZ)")
        'dartagnan nz'
        >>> normalize_horse_name(None)
        ''
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower()

    # Remove apostrophes and special characters (keep letters, numbers, spaces)
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Collapse multiple spaces to single space
    name = re.sub(r"\s+", " ", name)

    # Strip whitespace
    return name.strip()


def normalize_track_name(name: Optional[str]) -> str:
    """
    Normalize track name for matching.

    Args:
        name: Track name from any source

    Returns:
        Normalized lowercase track name

    Examples:
        >>> normalize_track_name("Eagle Farm")
        'eagle farm'
        >>> normalize_track_name("RANDWICK")
        'randwick'
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower()

    # Remove special characters (keep letters, numbers, spaces)
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def tracks_match(track1: Optional[str], track2: Optional[str]) -> bool:
    """
    Check if two track names refer to the same track.

    Uses substring matching for flexibility (e.g., "Randwick" matches "Randwick Kensington").

    Args:
        track1: First track name
        track2: Second track name

    Returns:
        True if tracks match

    Examples:
        >>> tracks_match("Eagle Farm", "EAGLE FARM")
        True
        >>> tracks_match("Randwick", "Randwick Kensington")
        True
        >>> tracks_match("Flemington", "Eagle Farm")
        False
    """
    norm1 = normalize_track_name(track1)
    norm2 = normalize_track_name(track2)

    if not norm1 or not norm2:
        return False

    # Exact match or substring match
    return norm1 == norm2 or norm1 in norm2 or norm2 in norm1


def horses_match(name1: Optional[str], name2: Optional[str]) -> bool:
    """
    Check if two horse names refer to the same horse.

    Args:
        name1: First horse name
        name2: Second horse name

    Returns:
        True if names match after normalization

    Examples:
        >>> horses_match("O'Brien's Pride", "obriens pride")
        True
        >>> horses_match("Fast Horse", "FAST HORSE")
        True
        >>> horses_match("Horse A", "Horse B")
        False
    """
    return normalize_horse_name(name1) == normalize_horse_name(name2)
