"""
Ladbrokes API Client.

Provides live racing odds. Use this instead of PuntingForm's bestPrice_Current
which can be unreliable.

Usage:
    from api.ladbrokes import LadbrokeAPI

    api = LadbrokeAPI()
    meetings = api.get_meetings()
    odds = api.get_odds_for_race("Eagle Farm", 1)

    # With track mapping (handles name differences)
    odds = api.get_odds_for_pf_track("Sandown-Lakeside", 1)  # Searches "Sandown"
"""

from typing import Optional
from dataclasses import dataclass
import requests

from core.normalize import normalize_horse_name, normalize_track_name, tracks_match
from core.track_mapping import (
    get_lb_track_for_odds,
    is_pf_only_track,
    tracks_equivalent,
)
from core.logging import get_logger, log_prediction_skip, log_odds_mismatch

logger = get_logger(__name__)

BASE_URL = "https://api.ladbrokes.com.au/affiliates/v1"
DEFAULT_TIMEOUT = 10

# Required headers
DEFAULT_HEADERS = {
    "From": "contact@puntlegacy.com",
    "X-Partner": "PuntLegacy",
}


@dataclass
class APIError(Exception):
    """Ladbrokes API error."""
    status_code: int
    message: str


class LadbrokeAPI:
    """
    Ladbrokes API client for live racing odds.

    Provides accurate market prices for Australian thoroughbred racing.
    """

    def __init__(
        self,
        email: Optional[str] = None,
        partner: Optional[str] = None,
    ):
        """
        Initialize API client.

        Args:
            email: Contact email for From header
            partner: Partner name for X-Partner header
        """
        self.headers = {
            "From": email or DEFAULT_HEADERS["From"],
            "X-Partner": partner or DEFAULT_HEADERS["X-Partner"],
        }

    def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> dict:
        """
        Make API request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response data

        Raises:
            APIError: If API returns error
        """
        url = f"{BASE_URL}{endpoint}"

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})

        except requests.exceptions.HTTPError as e:
            raise APIError(
                status_code=e.response.status_code,
                message=f"HTTP {e.response.status_code}: {str(e)}",
            ) from e
        except requests.exceptions.RequestException as e:
            raise APIError(status_code=0, message=str(e)) from e

    # -------------------------------------------------------------------------
    # Meetings
    # -------------------------------------------------------------------------

    def get_meetings(
        self,
        date_from: str = "today",
        category: str = "T",
        country: str = "AUS",
    ) -> list[dict]:
        """
        Get racing meetings.

        Args:
            date_from: Date (YYYY-MM-DD, "today", "now", "week")
            category: T=Thoroughbred, H=Harness, G=Greyhound
            country: Country code (AUS, NZ, etc.)

        Returns:
            List of meeting dicts with races
        """
        data = self._request(
            "/racing/meetings",
            {
                "date_from": date_from,
                "date_to": date_from,
                "category": category,
                "country": country,
            },
        )
        return data.get("meetings", [])

    # -------------------------------------------------------------------------
    # Race Details
    # -------------------------------------------------------------------------

    def get_race(self, race_id: str) -> Optional[dict]:
        """
        Get race details with runners and odds.

        Args:
            race_id: Ladbrokes race UUID

        Returns:
            Dict with race info and runners, or None if not found
        """
        try:
            data = self._request(f"/racing/events/{race_id}")
            return {
                "race": data.get("race", {}),
                "runners": data.get("runners", []),
                "results": data.get("results", []),
            }
        except APIError:
            return None

    # -------------------------------------------------------------------------
    # Get Odds by Track/Race
    # -------------------------------------------------------------------------

    def get_odds_for_race(
        self,
        track_name: str,
        race_number: int,
        date: str = "today",
    ) -> tuple[dict[str, dict], Optional[str], Optional[str]]:
        """
        Get odds for a specific track and race.

        Args:
            track_name: Track name (partial match OK)
            race_number: Race number
            date: Date for the race (YYYY-MM-DD, "today", or "tomorrow")

        Returns:
            Tuple of (odds_dict, race_status, error_message)
            - odds_dict: Dict mapping normalized horse names to odds
            - race_status: "open", "closed", "final", "abandoned" or None
            - error_message: Human-readable error or None if success
        """
        logger.debug(f"Fetching odds for {track_name} R{race_number} on {date}")

        meetings = self.get_meetings(date_from=date)

        for meeting in meetings:
            meeting_name = meeting.get("name", "")

            if tracks_match(track_name, meeting_name) or tracks_equivalent(track_name, meeting_name):
                # Found the track, find the race
                for race in meeting.get("races", []):
                    if race.get("race_number") == race_number:
                        race_id = race.get("id")
                        race_status = race.get("status", "unknown")

                        race_data = self.get_race(race_id)

                        if not race_data:
                            logger.warning(f"No race data for {track_name} R{race_number}")
                            return {}, race_status, f"No race data available"

                        odds = self._build_odds_dict(race_data.get("runners", []))
                        logger.debug(f"Found odds for {len(odds)} runners at {track_name} R{race_number} (status: {race_status})")
                        return odds, race_status, None

        logger.info(f"Track not found in Ladbrokes: {track_name}")
        return {}, None, f"Track not found: {track_name}"

    def get_odds_for_pf_track(
        self,
        pf_track_name: str,
        race_number: int,
        date: str = "today",
    ) -> tuple[dict[str, dict], Optional[str]]:
        """
        Get odds for a PuntingForm track, handling name differences.

        This is the recommended method when starting from PuntingForm data.
        It handles track name mapping (e.g., Sandown-Lakeside -> Sandown).

        Args:
            pf_track_name: Track name from PuntingForm
            race_number: Race number
            date: Date for the race (YYYY-MM-DD, "today", or "tomorrow")

        Returns:
            Tuple of (odds_dict, error_message)
            - If successful: (odds_dict, None)
            - If failed: ({}, "Human-readable error message")

        Examples:
            odds, error = api.get_odds_for_pf_track("Sandown-Lakeside", 1)
            if error:
                print(f"Skipped: {error}")
            else:
                print(f"Got odds for {len(odds)} runners")
        """
        # Check if track is supported
        if is_pf_only_track(pf_track_name):
            reason = f"{pf_track_name} is not covered by Ladbrokes"
            log_prediction_skip(logger, pf_track_name, race_number, reason)
            return {}, reason

        # Get the Ladbrokes track name
        lb_track = get_lb_track_for_odds(pf_track_name)
        if not lb_track:
            reason = f"No Ladbrokes mapping for {pf_track_name}"
            log_prediction_skip(logger, pf_track_name, race_number, reason)
            return {}, reason

        # Fetch odds and check race status
        odds, race_status, error = self.get_odds_for_race(lb_track, race_number, date)

        logger.debug(f"Race status for {pf_track_name} R{race_number}: '{race_status}' (repr: {repr(race_status)})")

        # Check if race is still open for betting (case-insensitive, strip whitespace)
        # Use startswith to handle variations like "Open - Early Prices"
        race_status_clean = race_status.strip().lower() if race_status else None
        is_open = race_status_clean and race_status_clean.startswith("open")
        if race_status_clean and not is_open:
            status_messages = {
                "closed": "Race has started - betting is closed",
                "final": "Race has finished",
                "abandoned": "Race has been abandoned",
                "interim": "Race has finished - results pending",
            }
            reason = status_messages.get(race_status_clean, f"Race is not available (status: {race_status})")
            log_prediction_skip(logger, pf_track_name, race_number, reason)
            return {}, reason

        if error:
            log_prediction_skip(logger, pf_track_name, race_number, error)
            return {}, error

        if not odds:
            reason = f"No odds available for {pf_track_name} R{race_number} (searched: {lb_track})"
            log_prediction_skip(logger, pf_track_name, race_number, reason)
            return {}, reason

        return odds, None

    def _build_odds_dict(self, runners: list[dict]) -> dict[str, dict]:
        """
        Build odds dictionary from runners list.

        Args:
            runners: List of runner dicts from API

        Returns:
            Dict keyed by normalized horse name
        """
        odds_dict = {}

        for runner in runners:
            name = runner.get("name", "")
            normalized_name = normalize_horse_name(name)

            if not normalized_name:
                continue

            odds = runner.get("odds", {})

            # Check for scratching - API uses "is_scratched" field
            is_scratched = runner.get("is_scratched", False)

            # Also check for blank jockey as late scratching indicator
            jockey = runner.get("jockey", "")
            if not jockey or jockey.strip() == "":
                is_scratched = True

            odds_dict[normalized_name] = {
                "fixed_win": odds.get("fixed_win"),
                "fixed_place": odds.get("fixed_place"),
                "scratched": is_scratched,
                "barrier": runner.get("barrier"),
                "runner_number": runner.get("runner_number"),
                "jockey": jockey,
                "trainer": runner.get("trainer_name"),
                "original_name": name,  # Keep original for debugging
            }

        return odds_dict

    # -------------------------------------------------------------------------
    # Get Odds for Horse
    # -------------------------------------------------------------------------

    def get_horse_odds(
        self,
        track_name: str,
        race_number: int,
        horse_name: str,
    ) -> Optional[dict]:
        """
        Get odds for a specific horse.

        Args:
            track_name: Track name
            race_number: Race number
            horse_name: Horse name (will be normalized for matching)

        Returns:
            Odds dict or None if not found
        """
        race_odds, _, _ = self.get_odds_for_race(track_name, race_number)
        normalized_name = normalize_horse_name(horse_name)
        return race_odds.get(normalized_name)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_price(
        self,
        horse_name: str,
        pf_price: float,
        track_name: str,
        race_number: int,
        threshold: float = 2.0,
    ) -> tuple[bool, Optional[float]]:
        """
        Validate PuntingForm price against Ladbrokes.

        Args:
            horse_name: Horse name
            pf_price: PuntingForm bestPrice_Current
            track_name: Track name
            race_number: Race number
            threshold: Max acceptable ratio (default 2.0)

        Returns:
            Tuple of (is_valid, ladbrokes_price)
            is_valid is True if prices are within threshold
        """
        lb_odds = self.get_horse_odds(track_name, race_number, horse_name)

        if not lb_odds or not lb_odds.get("fixed_win"):
            return False, None

        lb_price = lb_odds["fixed_win"]

        if lb_odds.get("scratched"):
            return False, lb_price

        # Check ratio
        if pf_price > 0 and lb_price > 0:
            ratio = max(pf_price, lb_price) / min(pf_price, lb_price)
            is_valid = ratio <= threshold
            return is_valid, lb_price

        return False, lb_price
