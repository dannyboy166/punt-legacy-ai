"""
Tests for horse and track name normalization.

Run with: python -m pytest tests/test_normalize.py -v
"""

import pytest
from core.normalize import (
    normalize_horse_name,
    normalize_track_name,
    tracks_match,
    horses_match,
)


class TestNormalizeHorseName:
    """Tests for normalize_horse_name()."""

    def test_lowercase(self):
        assert normalize_horse_name("FAST HORSE") == "fast horse"
        assert normalize_horse_name("Fast Horse") == "fast horse"

    def test_apostrophe_removal(self):
        assert normalize_horse_name("O'Brien's Pride") == "obriens pride"
        assert normalize_horse_name("It's A Winner") == "its a winner"

    def test_special_characters(self):
        assert normalize_horse_name("D'Artagnan (NZ)") == "dartagnan nz"
        assert normalize_horse_name("Horse-Name") == "horsename"
        assert normalize_horse_name("Horse.Name") == "horsename"

    def test_extra_whitespace(self):
        assert normalize_horse_name("Fast  Horse") == "fast horse"
        assert normalize_horse_name("  Fast Horse  ") == "fast horse"
        assert normalize_horse_name("Fast   Horse   Name") == "fast horse name"

    def test_empty_input(self):
        assert normalize_horse_name("") == ""
        assert normalize_horse_name(None) == ""

    def test_numbers_preserved(self):
        assert normalize_horse_name("Horse 123") == "horse 123"
        assert normalize_horse_name("Winner2Be") == "winner2be"

    def test_real_horse_names(self):
        """Test with actual horse names from racing."""
        assert normalize_horse_name("Yoyo Yeezy") == "yoyo yeezy"
        assert normalize_horse_name("Shalhavmusik") == "shalhavmusik"
        assert normalize_horse_name("French Love") == "french love"
        assert normalize_horse_name("Hard To Go Wrong") == "hard to go wrong"


class TestNormalizeTrackName:
    """Tests for normalize_track_name()."""

    def test_lowercase(self):
        assert normalize_track_name("Eagle Farm") == "eagle farm"
        assert normalize_track_name("RANDWICK") == "randwick"

    def test_special_characters(self):
        assert normalize_track_name("Track (Synthetic)") == "track synthetic"

    def test_empty_input(self):
        assert normalize_track_name("") == ""
        assert normalize_track_name(None) == ""

    def test_real_track_names(self):
        """Test with actual track names."""
        assert normalize_track_name("Eagle Farm") == "eagle farm"
        assert normalize_track_name("Warwick Farm") == "warwick farm"
        assert normalize_track_name("Randwick") == "randwick"
        assert normalize_track_name("Flemington") == "flemington"
        assert normalize_track_name("Rockhampton") == "rockhampton"


class TestTracksMatch:
    """Tests for tracks_match()."""

    def test_exact_match(self):
        assert tracks_match("Eagle Farm", "Eagle Farm") is True

    def test_case_insensitive(self):
        assert tracks_match("Eagle Farm", "EAGLE FARM") is True
        assert tracks_match("randwick", "RANDWICK") is True

    def test_substring_match(self):
        """Substring matching for tracks with variants."""
        assert tracks_match("Randwick", "Randwick Kensington") is True
        assert tracks_match("Randwick Kensington", "Randwick") is True

    def test_no_match(self):
        assert tracks_match("Eagle Farm", "Flemington") is False
        assert tracks_match("Randwick", "Doomben") is False

    def test_empty_input(self):
        assert tracks_match("", "Eagle Farm") is False
        assert tracks_match("Eagle Farm", "") is False
        assert tracks_match(None, "Eagle Farm") is False
        assert tracks_match("Eagle Farm", None) is False


class TestHorsesMatch:
    """Tests for horses_match()."""

    def test_exact_match(self):
        assert horses_match("Fast Horse", "Fast Horse") is True

    def test_case_insensitive(self):
        assert horses_match("Fast Horse", "FAST HORSE") is True
        assert horses_match("fast horse", "FAST HORSE") is True

    def test_apostrophe_handling(self):
        assert horses_match("O'Brien's Pride", "obriens pride") is True
        assert horses_match("It's A Winner", "Its A Winner") is True

    def test_special_characters(self):
        assert horses_match("D'Artagnan (NZ)", "Dartagnan NZ") is True

    def test_no_match(self):
        assert horses_match("Horse A", "Horse B") is False
        assert horses_match("Fast Horse", "Slow Horse") is False

    def test_empty_input(self):
        assert horses_match("", "Horse") is False
        assert horses_match("Horse", "") is False
        assert horses_match(None, "Horse") is False


class TestCrossAPIMatching:
    """
    Test matching between PuntingForm and Ladbrokes naming conventions.

    These tests simulate real-world matching scenarios.
    """

    def test_puntingform_to_ladbrokes_horse(self):
        """Test horse name matching between APIs."""
        # PuntingForm format -> Ladbrokes format
        test_cases = [
            ("Yoyo Yeezy", "YOYO YEEZY"),
            ("Hard To Go Wrong", "hard to go wrong"),
            ("O'Brien's Star", "Obriens Star"),
        ]

        for pf_name, lb_name in test_cases:
            assert horses_match(pf_name, lb_name), f"Failed: {pf_name} vs {lb_name}"

    def test_puntingform_to_ladbrokes_track(self):
        """Test track name matching between APIs."""
        test_cases = [
            ("Eagle Farm", "eagle farm"),
            ("Warwick Farm", "WARWICK FARM"),
            ("Randwick", "randwick"),
        ]

        for pf_name, lb_name in test_cases:
            assert tracks_match(pf_name, lb_name), f"Failed: {pf_name} vs {lb_name}"
