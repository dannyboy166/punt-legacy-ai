"""
PuntingForm API Client.

Provides access to race data, form history, conditions, and scratchings.
Does NOT use bestPrice_Current (unreliable) - use Ladbrokes for live odds.

Usage:
    from api.puntingform import PuntingFormAPI

    api = PuntingFormAPI()
    meetings = api.get_meetings("08-Jan-2026")
"""

import os
from typing import Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.puntingform.com.au/v2"
DEFAULT_TIMEOUT = 15


@dataclass
class APIError(Exception):
    """PuntingForm API error."""
    status_code: int
    message: str


class PuntingFormAPI:
    """
    PuntingForm API client.

    Handles authentication, error handling, and provides typed methods
    for all endpoints.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client.

        Args:
            api_key: API key (defaults to PUNTINGFORM_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("PUNTINGFORM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set PUNTINGFORM_API_KEY env var or pass api_key."
            )

    def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Make API request.

        Args:
            endpoint: API endpoint (e.g., "/form/meetingslist")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response (payLoad field)

        Raises:
            APIError: If API returns error
        """
        url = f"{BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # PuntingForm returns data in "payLoad" field
            return data.get("payLoad", data)

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

    def get_meetings(self, date: str) -> list[dict]:
        """
        Get all meetings for a date.

        Args:
            date: Date in format "dd-MMM-yyyy" (e.g., "08-Jan-2026")

        Returns:
            List of meeting dicts with meetingId, track, etc.
        """
        return self._request("/form/meetingslist", {"meetingDate": date})

    # -------------------------------------------------------------------------
    # Fields (Runner Details)
    # -------------------------------------------------------------------------

    def get_fields(self, meeting_id: int, race_number: int = 0) -> dict:
        """
        Get runner details for a meeting.

        Args:
            meeting_id: Meeting ID from get_meetings()
            race_number: Race number (0 = all races)

        Returns:
            Dict with races containing runners with career stats, A/E data, etc.
        """
        return self._request(
            "/form/fields",
            {"meetingId": meeting_id, "raceNumber": race_number},
        )

    # -------------------------------------------------------------------------
    # Form (Past Runs) - CRITICAL FOR SPEED CALCULATIONS
    # -------------------------------------------------------------------------

    def get_form(
        self,
        meeting_id: int,
        race_number: int = 0,
        runs: int = 10,
    ) -> list[dict]:
        """
        Get past run history for runners.

        This is the most important endpoint for speed calculations.
        Contains officialRaceTime, margin, position, distance.

        Args:
            meeting_id: Meeting ID
            race_number: Race number (0 = all races)
            runs: Number of past runs per horse (max 10)

        Returns:
            List of runners with forms array containing past runs
        """
        return self._request(
            "/form/form",
            {"meetingId": meeting_id, "raceNumber": race_number, "runs": runs},
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    def get_results(self, meeting_id: int, race_number: int = 0) -> list[dict]:
        """
        Get race results after races finish.

        Args:
            meeting_id: Meeting ID
            race_number: Race number (0 = all races)

        Returns:
            List of race results with positions, margins, SPs
        """
        return self._request(
            "/form/results",
            {"meetingId": meeting_id, "raceNumber": race_number},
        )

    # -------------------------------------------------------------------------
    # Conditions
    # -------------------------------------------------------------------------

    def get_conditions(self, jurisdiction: int = 0) -> list[dict]:
        """
        Get current track conditions.

        Args:
            jurisdiction: 0 = all, 1 = NSW/ACT, 2 = VIC/TAS

        Returns:
            List of track conditions with conditionNumber (1-10)
        """
        return self._request(
            "/Updates/Conditions",
            {"jurisdiction": jurisdiction},
        )

    # -------------------------------------------------------------------------
    # Scratchings
    # -------------------------------------------------------------------------

    def get_scratchings(self, jurisdiction: int = 0) -> list[dict]:
        """
        Get scratched horses.

        Args:
            jurisdiction: 0 = all

        Returns:
            List of scratchings with meetingId, raceNo, runnerId, tabNo
        """
        return self._request(
            "/Updates/Scratchings",
            {"jurisdiction": jurisdiction},
        )

    # -------------------------------------------------------------------------
    # Speedmaps
    # -------------------------------------------------------------------------

    def get_speedmaps(self, meeting_id: int, race_number: int = 0) -> list[dict]:
        """
        Get pace/settling predictions.

        Args:
            meeting_id: Meeting ID
            race_number: Race number (0 = all races)

        Returns:
            List of runners with speed rank, settle position, runStyle
        """
        return self._request(
            "/User/Speedmaps",
            {"meetingId": meeting_id, "raceNo": race_number},
        )

    # -------------------------------------------------------------------------
    # PFAI Ratings
    # -------------------------------------------------------------------------

    def get_ratings(self, meeting_id: int, race_number: int = 0) -> dict[int, dict]:
        """
        Get PFAI ratings for a meeting, indexed by tabNo.

        Args:
            meeting_id: Meeting ID
            race_number: Race number (0 = all races)

        Returns:
            Dict mapping (race_number, tabNo) or just tabNo to rating data:
            {
                (1, 1): {"pfai_rank": 1, "is_reliable": True, "race_no": 1},
                (1, 2): {"pfai_rank": 3, "is_reliable": True, "race_no": 1},
                ...
            }
            Or if race_number specified:
            {
                1: {"pfai_rank": 1, "is_reliable": True},
                2: {"pfai_rank": 3, "is_reliable": True},
            }
        """
        data = self._request(
            "/Ratings/MeetingRatings",
            {"meetingId": meeting_id},
        )

        ratings_by_tab = {}
        if isinstance(data, list):
            for runner in data:
                # Data is flat - each item is a runner directly
                tab_no = runner.get("tabNo")
                race_no = runner.get("raceNo")
                if tab_no is not None:
                    rating_data = {
                        "pfai_rank": runner.get("pfaiRank"),
                        "is_reliable": runner.get("isReliable", False),
                        "pfai_score": runner.get("pfaiScore"),
                        "race_no": race_no,
                    }
                    if race_number == 0:
                        # Return keyed by (race_no, tab_no)
                        ratings_by_tab[(race_no, tab_no)] = rating_data
                    elif race_no == race_number:
                        # Return keyed by tab_no only for specific race
                        ratings_by_tab[tab_no] = rating_data
        return ratings_by_tab

    # -------------------------------------------------------------------------
    # Strikerate (Jockey/Trainer Stats)
    # -------------------------------------------------------------------------

    def get_strikerate(self, entity_type: int, jurisdiction: int = 0) -> list[dict]:
        """
        Get jockey or trainer career stats.

        Args:
            entity_type: 1 = Trainers, 2 = Jockeys
            jurisdiction: 0 = all

        Returns:
            List of entities with career and last100 stats, A/E data
        """
        if entity_type not in (1, 2):
            raise ValueError("entity_type must be 1 (Trainers) or 2 (Jockeys)")

        return self._request(
            "/form/strikerate",
            {"entityType": entity_type, "jurisdiction": jurisdiction},
        )

    def get_jockey_stats(self, jurisdiction: int = 0) -> list[dict]:
        """Get all jockey stats."""
        return self.get_strikerate(entity_type=2, jurisdiction=jurisdiction)

    def get_trainer_stats(self, jurisdiction: int = 0) -> list[dict]:
        """Get all trainer stats."""
        return self.get_strikerate(entity_type=1, jurisdiction=jurisdiction)

    # -------------------------------------------------------------------------
    # Worksheets (Market Prices - USE WITH CAUTION)
    # -------------------------------------------------------------------------

    def get_worksheets(self, meeting_id: int, race_number: int = 0) -> list[dict]:
        """
        Get market prices.

        WARNING: bestPrice_Current can be unreliable. Use Ladbrokes API
        for accurate live odds. This endpoint is useful for:
        - bestPrice_SinceOpen (opening price)
        - mapA2E (map-based A/E)

        Args:
            meeting_id: Meeting ID
            race_number: Race number (0 = all races)

        Returns:
            List of runners with price data
        """
        return self._request(
            "/User/Worksheets",
            {"meetingId": meeting_id, "raceNo": race_number},
        )

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def get_meeting_data(
        self,
        meeting_id: int,
        include_form: bool = True,
        include_speedmaps: bool = True,
    ) -> dict:
        """
        Get all data for a meeting in parallel.

        Args:
            meeting_id: Meeting ID
            include_form: Include form history
            include_speedmaps: Include speed maps

        Returns:
            Dict with fields, form, speedmaps, conditions, scratchings
        """
        result = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.get_fields, meeting_id, 0): "fields",
                executor.submit(self.get_conditions, 0): "conditions",
                executor.submit(self.get_scratchings, 0): "scratchings",
            }

            if include_form:
                futures[executor.submit(self.get_form, meeting_id, 0, 10)] = "form"

            if include_speedmaps:
                futures[executor.submit(self.get_speedmaps, meeting_id, 0)] = "speedmaps"

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result[key] = future.result()
                except Exception as e:
                    result[key] = None
                    print(f"Error fetching {key}: {e}")

        return result
