"""
Tests for track name mapping between APIs.

Run with: python -m pytest tests/test_track_mapping.py -v
"""

import pytest
from core.track_mapping import (
    pf_to_lb_track,
    lb_to_pf_track,
    is_pf_only_track,
    is_lb_only_track,
    tracks_equivalent,
    get_lb_track_for_odds,
    normalize_with_aliases,
)


class TestPfToLbTrack:
    """Tests for pf_to_lb_track()."""

    def test_sandown(self):
        """Sandown-Lakeside (PF) -> Sandown (LB)."""
        assert pf_to_lb_track("Sandown-Lakeside") == "sandown"
        assert pf_to_lb_track("sandown-lakeside") == "sandown"
        assert pf_to_lb_track("SANDOWN-LAKESIDE") == "sandown"

    def test_yarra_glen(self):
        """Yarra Glen (PF) -> Yarra Valley (LB)."""
        assert pf_to_lb_track("Yarra Glen") == "yarra valley"

    def test_murray_bridge(self):
        """Murray Bridge GH (PF) -> Murray Bridge (LB)."""
        assert pf_to_lb_track("Murray Bridge GH") == "murray bridge"

    def test_geelong(self):
        """Geelong (PF) -> Ladbrokes Geelong (LB)."""
        assert pf_to_lb_track("Geelong") == "ladbrokes geelong"

    def test_fannie_bay(self):
        """Fannie Bay (PF) -> Darwin (LB)."""
        assert pf_to_lb_track("Fannie Bay") == "darwin"

    def test_no_mapping_needed(self):
        """Tracks with same name in both APIs."""
        assert pf_to_lb_track("Randwick") == "randwick"
        assert pf_to_lb_track("Eagle Farm") == "eagle farm"
        assert pf_to_lb_track("Flemington") == "flemington"


class TestLbToPfTrack:
    """Tests for lb_to_pf_track()."""

    def test_sandown(self):
        """Sandown (LB) -> Sandown-Lakeside (PF)."""
        assert lb_to_pf_track("Sandown") == "sandownlakeside"  # Normalized

    def test_yarra_valley(self):
        """Yarra Valley (LB) -> Yarra Glen (PF)."""
        assert lb_to_pf_track("Yarra Valley") == "yarra glen"

    def test_murray_bridge(self):
        """Murray Bridge (LB) -> Murray Bridge GH (PF)."""
        assert lb_to_pf_track("Murray Bridge") == "murray bridge gh"

    def test_ladbrokes_geelong(self):
        """Ladbrokes Geelong (LB) -> Geelong (PF)."""
        assert lb_to_pf_track("Ladbrokes Geelong") == "geelong"

    def test_darwin(self):
        """Darwin (LB) -> Fannie Bay (PF)."""
        assert lb_to_pf_track("Darwin") == "fannie bay"

    def test_no_mapping_needed(self):
        """Tracks with same name in both APIs."""
        assert lb_to_pf_track("Randwick") == "randwick"
        assert lb_to_pf_track("Eagle Farm") == "eagle farm"


class TestIsPfOnlyTrack:
    """Tests for is_pf_only_track()."""

    def test_pioneer_park(self):
        """Pioneer Park (Alice Springs) is PF only."""
        assert is_pf_only_track("Pioneer Park") is True

    def test_nz_tracks(self):
        """NZ tracks are PF only."""
        assert is_pf_only_track("Tauranga") is True
        assert is_pf_only_track("Ellerslie") is True
        assert is_pf_only_track("Te Rapa") is True

    def test_hk_tracks(self):
        """HK tracks are PF only."""
        assert is_pf_only_track("Sha Tin") is True
        assert is_pf_only_track("Happy Valley") is True

    def test_aus_tracks(self):
        """Australian tracks are NOT PF only."""
        assert is_pf_only_track("Randwick") is False
        assert is_pf_only_track("Eagle Farm") is False
        assert is_pf_only_track("Sandown-Lakeside") is False


class TestIsLbOnlyTrack:
    """Tests for is_lb_only_track()."""

    def test_newcastle(self):
        """Newcastle is LB only (not in PuntingForm data)."""
        assert is_lb_only_track("Newcastle") is True

    def test_normal_tracks(self):
        """Normal tracks are NOT LB only."""
        assert is_lb_only_track("Randwick") is False
        assert is_lb_only_track("Sandown") is False


class TestTracksEquivalent:
    """Tests for tracks_equivalent()."""

    def test_exact_match(self):
        """Same name matches."""
        assert tracks_equivalent("Randwick", "Randwick") is True
        assert tracks_equivalent("Eagle Farm", "eagle farm") is True

    def test_sandown_mapping(self):
        """Sandown-Lakeside (PF) = Sandown (LB)."""
        assert tracks_equivalent("Sandown-Lakeside", "Sandown") is True
        assert tracks_equivalent("SANDOWN-LAKESIDE", "sandown") is True

    def test_yarra_mapping(self):
        """Yarra Glen (PF) = Yarra Valley (LB)."""
        assert tracks_equivalent("Yarra Glen", "Yarra Valley") is True

    def test_murray_bridge_mapping(self):
        """Murray Bridge GH (PF) = Murray Bridge (LB)."""
        assert tracks_equivalent("Murray Bridge GH", "Murray Bridge") is True

    def test_geelong_mapping(self):
        """Geelong (PF) = Ladbrokes Geelong (LB)."""
        assert tracks_equivalent("Geelong", "Ladbrokes Geelong") is True

    def test_darwin_mapping(self):
        """Fannie Bay (PF) = Darwin (LB)."""
        assert tracks_equivalent("Fannie Bay", "Darwin") is True

    def test_no_match(self):
        """Different tracks don't match."""
        assert tracks_equivalent("Randwick", "Flemington") is False
        assert tracks_equivalent("Eagle Farm", "Doomben") is False


class TestGetLbTrackForOdds:
    """Tests for get_lb_track_for_odds()."""

    def test_normal_track(self):
        """Normal tracks return mapped name."""
        assert get_lb_track_for_odds("Randwick") == "randwick"
        assert get_lb_track_for_odds("Sandown-Lakeside") == "sandown"
        assert get_lb_track_for_odds("Yarra Glen") == "yarra valley"

    def test_pf_only_track_returns_none(self):
        """PF-only tracks return None."""
        assert get_lb_track_for_odds("Pioneer Park") is None
        assert get_lb_track_for_odds("Tauranga") is None
        assert get_lb_track_for_odds("Sha Tin") is None


class TestNormalizeWithAliases:
    """Tests for normalize_with_aliases()."""

    def test_common_aliases(self):
        """Common track aliases resolve to canonical name."""
        assert normalize_with_aliases("The Valley") == "moonee valley"
        assert normalize_with_aliases("Rosehill Gardens") == "rosehill"

    def test_no_alias(self):
        """Non-aliased names pass through."""
        assert normalize_with_aliases("Randwick") == "randwick"
        assert normalize_with_aliases("Eagle Farm") == "eagle farm"


class TestRealWorldScenarios:
    """Test real-world data linking scenarios."""

    def test_fetch_odds_for_sandown_race(self):
        """
        Scenario: We have a race at "Sandown-Lakeside" from PuntingForm.
        We need to fetch odds from Ladbrokes.
        """
        pf_track = "Sandown-Lakeside"
        lb_track = get_lb_track_for_odds(pf_track)

        assert lb_track == "sandown"
        # Now we would call: ladbrokes_api.get_odds_for_race(lb_track, race_num)

    def test_match_runners_between_apis(self):
        """
        Scenario: We have runner data from PuntingForm at "Yarra Glen"
        and odds data from Ladbrokes at "Yarra Valley".
        Verify they're the same track.
        """
        pf_track = "Yarra Glen"
        lb_track = "Yarra Valley"

        assert tracks_equivalent(pf_track, lb_track) is True

    def test_skip_nz_races(self):
        """
        Scenario: We have a race at "Tauranga" from PuntingForm.
        We should skip fetching Ladbrokes odds (not available).
        """
        pf_track = "Tauranga"
        lb_track = get_lb_track_for_odds(pf_track)

        assert lb_track is None
        # Don't try to fetch odds - will fail
