"""
Track name mapping between PuntingForm and Ladbrokes APIs.

Different APIs use different names for the same tracks. This module
provides bidirectional mapping to ensure correct data linking.

Verified mappings as of Jan 2026:
- Sandown-Lakeside (PF) = Sandown (LB)
- Yarra Glen (PF) = Yarra Valley (LB)
- Murray Bridge GH (PF) = Murray Bridge (LB)
- Geelong (PF) = Ladbrokes Geelong (LB)
- Fannie Bay (PF) = Darwin (LB)

Known coverage gaps:
- Pioneer Park (Alice Springs) - PF only, no Ladbrokes coverage
- Newcastle - LB only, not in PuntingForm data
- NZ tracks - PF only, no Ladbrokes coverage
- HK tracks - PF only, no Ladbrokes coverage
"""

from typing import Optional
from core.normalize import normalize_track_name

# Human-readable mappings (for documentation)
# PuntingForm name -> Ladbrokes name
_TRACK_MAPPINGS = [
    ("Sandown-Lakeside", "Sandown"),
    ("Yarra Glen", "Yarra Valley"),
    ("Murray Bridge GH", "Murray Bridge"),
    ("Geelong", "Ladbrokes Geelong"),
    ("Fannie Bay", "Darwin"),
]

# Build normalized lookup dictionaries
PF_TO_LB_MAPPING = {
    normalize_track_name(pf): normalize_track_name(lb)
    for pf, lb in _TRACK_MAPPINGS
}

LB_TO_PF_MAPPING = {
    normalize_track_name(lb): normalize_track_name(pf)
    for pf, lb in _TRACK_MAPPINGS
}

# Tracks that only exist in one API (no odds available in the other)
# All stored as normalized names
PF_ONLY_TRACKS = {
    normalize_track_name(t) for t in [
        # NT
        "Pioneer Park",  # Alice Springs - no Ladbrokes coverage
        # NZ tracks
        "Tauranga", "Ellerslie", "Te Rapa", "Matamata", "Ruakaka",
        "Te Aroha", "Trentham", "Otaki", "Tauherenikau", "Taupo",
        "New Plymouth Raceway", "Phar Lap Raceway", "Arawa Park",
        "Ascot Park", "Ashburton", "Gore", "Greymouth", "Kurow",
        "Reefton", "Riverton", "Wingatui",
        # HK tracks
        "Sha Tin", "Happy Valley",
    ]
}

LB_ONLY_TRACKS = {
    normalize_track_name(t) for t in [
        "Newcastle",  # Not in PuntingForm data
    ]
}


def pf_to_lb_track(pf_name: str) -> str:
    """
    Convert PuntingForm track name to Ladbrokes equivalent.

    Args:
        pf_name: Track name from PuntingForm API

    Returns:
        Equivalent Ladbrokes track name (normalized)

    Examples:
        >>> pf_to_lb_track("Sandown-Lakeside")
        'sandown'
        >>> pf_to_lb_track("Randwick")
        'randwick'
    """
    normalized = normalize_track_name(pf_name)
    return PF_TO_LB_MAPPING.get(normalized, normalized)


def lb_to_pf_track(lb_name: str) -> str:
    """
    Convert Ladbrokes track name to PuntingForm equivalent.

    Args:
        lb_name: Track name from Ladbrokes API

    Returns:
        Equivalent PuntingForm track name (normalized)

    Examples:
        >>> lb_to_pf_track("Sandown")
        'sandownlakeside'
        >>> lb_to_pf_track("Randwick")
        'randwick'
    """
    normalized = normalize_track_name(lb_name)
    return LB_TO_PF_MAPPING.get(normalized, normalized)


def is_pf_only_track(track_name: str) -> bool:
    """
    Check if track is only available in PuntingForm (no Ladbrokes odds).

    Args:
        track_name: Track name from either API

    Returns:
        True if track has no Ladbrokes coverage
    """
    normalized = normalize_track_name(track_name)
    return normalized in PF_ONLY_TRACKS


def is_lb_only_track(track_name: str) -> bool:
    """
    Check if track is only available in Ladbrokes (not in PuntingForm).

    Args:
        track_name: Track name from either API

    Returns:
        True if track has no PuntingForm coverage
    """
    normalized = normalize_track_name(track_name)
    return normalized in LB_ONLY_TRACKS


def tracks_equivalent(pf_name: str, lb_name: str) -> bool:
    """
    Check if a PuntingForm track and Ladbrokes track are the same.

    Handles known name differences between APIs.

    Args:
        pf_name: Track name from PuntingForm
        lb_name: Track name from Ladbrokes

    Returns:
        True if they refer to the same track

    Examples:
        >>> tracks_equivalent("Sandown-Lakeside", "Sandown")
        True
        >>> tracks_equivalent("Yarra Glen", "Yarra Valley")
        True
        >>> tracks_equivalent("Randwick", "Randwick")
        True
        >>> tracks_equivalent("Eagle Farm", "Flemington")
        False
    """
    pf_normalized = normalize_track_name(pf_name)
    lb_normalized = normalize_track_name(lb_name)

    # Direct match
    if pf_normalized == lb_normalized:
        return True

    # Check mapping
    mapped_lb = PF_TO_LB_MAPPING.get(pf_normalized)
    if mapped_lb and mapped_lb == lb_normalized:
        return True

    # Check reverse mapping
    mapped_pf = LB_TO_PF_MAPPING.get(lb_normalized)
    if mapped_pf and mapped_pf == pf_normalized:
        return True

    # Substring match as fallback (e.g., "Geelong" in "Ladbrokes Geelong")
    if pf_normalized in lb_normalized or lb_normalized in pf_normalized:
        return True

    return False


def get_lb_track_for_odds(pf_track: str) -> Optional[str]:
    """
    Get the Ladbrokes track name to use for fetching odds.

    Returns None if the track has no Ladbrokes coverage.

    Args:
        pf_track: PuntingForm track name

    Returns:
        Ladbrokes track name to search for, or None if not available

    Examples:
        >>> get_lb_track_for_odds("Sandown-Lakeside")
        'sandown'
        >>> get_lb_track_for_odds("Pioneer Park")  # Alice Springs
        None
        >>> get_lb_track_for_odds("Randwick")
        'randwick'
    """
    if is_pf_only_track(pf_track):
        return None

    return pf_to_lb_track(pf_track)


# Known track aliases and common misspellings
_TRACK_ALIASES_RAW = {
    # Common variations
    "Flemington": ["Flemington Racecourse"],
    "Caulfield": ["Caulfield Racecourse"],
    "Moonee Valley": ["Moonee Valley Racecourse", "The Valley"],
    "Rosehill": ["Rosehill Gardens"],
    "Morphettville": ["Morphettville Racecourse", "Morphettville Parks"],
    "Doomben": ["Doomben Racecourse"],
    "Eagle Farm": ["Eagle Farm Racecourse"],
    "Ascot": ["Ascot Racecourse"],  # Perth
}

# Build normalized alias lookup
TRACK_ALIASES = {}
for canonical, aliases in _TRACK_ALIASES_RAW.items():
    normalized_canonical = normalize_track_name(canonical)
    for alias in aliases:
        TRACK_ALIASES[normalize_track_name(alias)] = normalized_canonical


def normalize_with_aliases(track_name: str) -> str:
    """
    Normalize track name, resolving common aliases.

    Args:
        track_name: Track name from any source

    Returns:
        Canonical normalized name
    """
    normalized = normalize_track_name(track_name)

    # Check if this is an alias
    if normalized in TRACK_ALIASES:
        return TRACK_ALIASES[normalized]

    return normalized
