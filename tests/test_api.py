"""
Tests for API clients.

Run with: python -m pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import Mock, patch

from api.puntingform import PuntingFormAPI, APIError
from api.ladbrokes import LadbrokeAPI


class TestPuntingFormAPI:
    """Tests for PuntingFormAPI client."""

    def test_init_with_key(self):
        """Test initialization with API key."""
        api = PuntingFormAPI(api_key="test-key")
        assert api.api_key == "test-key"

    def test_init_without_key_raises(self):
        """Test initialization without key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                PuntingFormAPI(api_key=None)

    @patch("api.puntingform.requests.get")
    def test_get_meetings(self, mock_get):
        """Test get_meetings returns meeting list."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "payLoad": [
                {"meetingId": 123, "track": {"name": "Randwick"}},
                {"meetingId": 456, "track": {"name": "Flemington"}},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = PuntingFormAPI(api_key="test-key")
        meetings = api.get_meetings("08-Jan-2026")

        assert len(meetings) == 2
        assert meetings[0]["meetingId"] == 123
        mock_get.assert_called_once()

    @patch("api.puntingform.requests.get")
    def test_get_fields(self, mock_get):
        """Test get_fields returns runner data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "payLoad": {
                "races": [
                    {
                        "number": 1,
                        "runners": [
                            {"runnerId": 1, "name": "Fast Horse"},
                        ],
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = PuntingFormAPI(api_key="test-key")
        fields = api.get_fields(meeting_id=123, race_number=0)

        assert "races" in fields
        assert fields["races"][0]["runners"][0]["name"] == "Fast Horse"

    @patch("api.puntingform.requests.get")
    def test_get_form(self, mock_get):
        """Test get_form returns form history."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "payLoad": [
                {
                    "runnerId": 1,
                    "name": "Fast Horse",
                    "forms": [
                        {"position": 1, "margin": 0, "officialRaceTime": "00:01:11.34"},
                    ],
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = PuntingFormAPI(api_key="test-key")
        form = api.get_form(meeting_id=123, race_number=1, runs=10)

        assert len(form) == 1
        assert form[0]["forms"][0]["position"] == 1

    @patch("api.puntingform.requests.get")
    def test_strikerate_jockeys(self, mock_get):
        """Test get_strikerate for jockeys."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "payLoad": [
                {"entityId": 1, "entityName": "J. McDonald", "last100Wins": 15},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = PuntingFormAPI(api_key="test-key")
        stats = api.get_jockey_stats()

        assert stats[0]["entityName"] == "J. McDonald"

    def test_strikerate_invalid_entity_type(self):
        """Test invalid entity type raises ValueError."""
        api = PuntingFormAPI(api_key="test-key")
        with pytest.raises(ValueError, match="entity_type must be 1"):
            api.get_strikerate(entity_type=3)


class TestLadbrokeAPI:
    """Tests for LadbrokeAPI client."""

    def test_init_default_headers(self):
        """Test initialization with default headers."""
        api = LadbrokeAPI()
        assert "From" in api.headers
        assert "X-Partner" in api.headers

    def test_init_custom_headers(self):
        """Test initialization with custom headers."""
        api = LadbrokeAPI(email="test@test.com", partner="TestOrg")
        assert api.headers["From"] == "test@test.com"
        assert api.headers["X-Partner"] == "TestOrg"

    @patch("api.ladbrokes.requests.get")
    def test_get_meetings(self, mock_get):
        """Test get_meetings returns meeting list."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "meetings": [
                    {"name": "Randwick", "races": []},
                    {"name": "Flemington", "races": []},
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = LadbrokeAPI()
        meetings = api.get_meetings()

        assert len(meetings) == 2
        assert meetings[0]["name"] == "Randwick"

    @patch("api.ladbrokes.requests.get")
    def test_get_race(self, mock_get):
        """Test get_race returns race with runners."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "race": {"race_number": 1},
                "runners": [
                    {"name": "Fast Horse", "odds": {"fixed_win": 3.50}},
                ],
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = LadbrokeAPI()
        race = api.get_race("race-uuid-123")

        assert race["race"]["race_number"] == 1
        assert race["runners"][0]["odds"]["fixed_win"] == 3.50

    def test_build_odds_dict(self):
        """Test odds dict building with name normalization."""
        api = LadbrokeAPI()

        runners = [
            {"name": "Fast Horse", "odds": {"fixed_win": 3.50, "fixed_place": 1.40}, "barrier": 5},
            {"name": "O'Brien's Star", "odds": {"fixed_win": 5.00, "fixed_place": 1.80}, "barrier": 2},
            {"name": "YOYO YEEZY", "odds": {"fixed_win": 1.30, "fixed_place": 1.06}, "scratched": True},
        ]

        odds_dict = api._build_odds_dict(runners)

        # Check normalized names
        assert "fast horse" in odds_dict
        assert "obriens star" in odds_dict
        assert "yoyo yeezy" in odds_dict

        # Check data
        assert odds_dict["fast horse"]["fixed_win"] == 3.50
        assert odds_dict["obriens star"]["fixed_win"] == 5.00
        assert odds_dict["yoyo yeezy"]["scratched"] is True

        # Check original name preserved
        assert odds_dict["obriens star"]["original_name"] == "O'Brien's Star"

    @patch("api.ladbrokes.requests.get")
    def test_get_odds_for_race(self, mock_get):
        """Test get_odds_for_race with track matching."""
        # First call: get_meetings
        meetings_response = Mock()
        meetings_response.json.return_value = {
            "data": {
                "meetings": [
                    {
                        "name": "Eagle Farm",
                        "races": [{"id": "race-123", "race_number": 1}],
                    }
                ]
            }
        }
        meetings_response.raise_for_status = Mock()

        # Second call: get_race
        race_response = Mock()
        race_response.json.return_value = {
            "data": {
                "race": {"race_number": 1},
                "runners": [
                    {"name": "Fast Horse", "odds": {"fixed_win": 3.50}},
                ],
            }
        }
        race_response.raise_for_status = Mock()

        mock_get.side_effect = [meetings_response, race_response]

        api = LadbrokeAPI()
        odds = api.get_odds_for_race("Eagle Farm", 1)

        assert "fast horse" in odds
        assert odds["fast horse"]["fixed_win"] == 3.50

    def test_get_odds_for_race_no_match(self):
        """Test get_odds_for_race when track not found."""
        api = LadbrokeAPI()

        with patch.object(api, "get_meetings", return_value=[]):
            odds = api.get_odds_for_race("Nonexistent Track", 1)
            assert odds == {}


class TestCrossAPIIntegration:
    """Test integration between PuntingForm and Ladbrokes APIs."""

    def test_price_validation(self):
        """Test price validation logic."""
        api = LadbrokeAPI()

        # Mock the get_horse_odds method
        with patch.object(
            api,
            "get_horse_odds",
            return_value={"fixed_win": 3.50, "scratched": False},
        ):
            # Valid: PF price within threshold
            is_valid, lb_price = api.validate_price(
                horse_name="Fast Horse",
                pf_price=4.00,  # Within 2x of 3.50
                track_name="Randwick",
                race_number=1,
            )
            assert is_valid is True
            assert lb_price == 3.50

            # Invalid: PF price too high
            is_valid, lb_price = api.validate_price(
                horse_name="Fast Horse",
                pf_price=10.00,  # More than 2x of 3.50
                track_name="Randwick",
                race_number=1,
            )
            assert is_valid is False
            assert lb_price == 3.50

    def test_price_validation_scratched(self):
        """Test price validation returns invalid for scratched horse."""
        api = LadbrokeAPI()

        with patch.object(
            api,
            "get_horse_odds",
            return_value={"fixed_win": 3.50, "scratched": True},
        ):
            is_valid, lb_price = api.validate_price(
                horse_name="Scratched Horse",
                pf_price=3.50,
                track_name="Randwick",
                race_number=1,
            )
            assert is_valid is False

    def test_price_validation_not_found(self):
        """Test price validation when horse not found."""
        api = LadbrokeAPI()

        with patch.object(api, "get_horse_odds", return_value=None):
            is_valid, lb_price = api.validate_price(
                horse_name="Unknown Horse",
                pf_price=5.00,
                track_name="Randwick",
                race_number=1,
            )
            assert is_valid is False
            assert lb_price is None
