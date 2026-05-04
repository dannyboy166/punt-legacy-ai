"""
Betfair API Client for punt-legacy-ai.

Handles login, market discovery, and bet placement.
Reuses SSL certificate auth from betfair-signal-monitor.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()


class BetfairClient:
    """Simple Betfair API client for finding markets and placing bets."""

    def __init__(self):
        self.app_key = os.getenv('BETFAIR_APP_KEY')
        self.username = os.getenv('BETFAIR_USERNAME')
        self.password = os.getenv('BETFAIR_PASSWORD')
        self.session_token = None
        self.base_url = 'https://api.betfair.com/exchange/betting/json-rpc/v1'

        # Certificate paths - check both locations
        project_certs = Path(__file__).parent.parent / 'certs'
        monitor_certs = Path.home() / 'betfair-signal-monitor' / 'certs'

        if (project_certs / 'betfair.crt').exists():
            self.cert_path = str(project_certs / 'betfair.crt')
            self.key_path = str(project_certs / 'betfair.key')
        elif (monitor_certs / 'betfair.crt').exists():
            self.cert_path = str(monitor_certs / 'betfair.crt')
            self.key_path = str(monitor_certs / 'betfair.key')
        else:
            self.cert_path = None
            self.key_path = None

    def login(self) -> bool:
        """Login to Betfair using SSL certificate authentication."""
        if not self.app_key or not self.username or not self.password:
            print("Missing BETFAIR_APP_KEY, BETFAIR_USERNAME, or BETFAIR_PASSWORD in .env")
            return False

        if not self.cert_path:
            print("Betfair certificates not found. Place betfair.crt and betfair.key in certs/")
            return False

        headers = {
            'X-Application': self.app_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = f'username={quote(self.username)}&password={quote(self.password)}'

        try:
            response = requests.post(
                'https://identitysso-cert.betfair.com/api/certlogin',
                data=data,
                headers=headers,
                cert=(self.cert_path, self.key_path)
            )
            result = response.json()

            if result.get('loginStatus') == 'SUCCESS':
                self.session_token = result.get('sessionToken')
                print("Betfair login successful")
                return True
            else:
                print(f"Betfair login failed: {result.get('loginStatus')}")
                return False
        except Exception as e:
            print(f"Betfair login error: {e}")
            return False

    def _call(self, method: str, params: dict) -> Optional[dict]:
        """Call a Betfair API method."""
        if not self.session_token:
            print("Not logged in")
            return None

        headers = {
            'X-Application': self.app_key,
            'X-Authentication': self.session_token,
            'Content-Type': 'application/json'
        }
        payload = {
            'jsonrpc': '2.0',
            'method': f'SportsAPING/v1.0/{method}',
            'params': params,
            'id': 1
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            data = response.json()
            if 'result' in data:
                return data['result']
            elif 'error' in data:
                error = data['error']
                code = error.get('data', {}).get('APINGException', {}).get('errorCode', 'UNKNOWN')
                print(f"API error [{code}]: {error.get('message', error)}")
                return None
        except Exception as e:
            print(f"API call error: {e}")
            return None

    def find_race(self, track: str, date: str) -> list:
        """
        Find Betfair markets for a track on a date.

        Args:
            track: Track name (e.g., "Randwick")
            date: Date string (e.g., "04-May-2026")

        Returns:
            List of market dicts with id, name, start time
        """
        race_date = datetime.strptime(date, "%d-%b-%Y")
        from_time = race_date.strftime("%Y-%m-%dT00:00:00Z")
        to_time = race_date.strftime("%Y-%m-%dT23:59:59Z")

        result = self._call('listMarketCatalogue', {
            'filter': {
                'eventTypeIds': ['7'],  # Horse racing
                'marketTypeCodes': ['WIN'],
                'venues': [track],
                'marketStartTime': {
                    'from': from_time,
                    'to': to_time
                }
            },
            'marketProjection': ['RUNNER_DESCRIPTION', 'MARKET_START_TIME'],
            'maxResults': '20',
            'sort': 'FIRST_TO_START'
        })

        if not result:
            return []

        markets = []
        for m in result:
            runners = {}
            for r in m.get('runners', []):
                name = r.get('runnerName', '')
                # Strip barrier prefix (e.g., "1. Horse Name" -> "Horse Name")
                if '. ' in name and name.split('. ')[0].isdigit():
                    name = name.split('. ', 1)[1]
                runners[name] = r.get('selectionId')

            markets.append({
                'market_id': m['marketId'],
                'name': m.get('marketName', ''),
                'start_time': m.get('marketStartTime', ''),
                'runners': runners
            })

        return markets

    def get_market_odds(self, market_id: str) -> dict:
        """
        Get current back odds for all runners in a market.

        Returns:
            Dict of {selection_id: best_back_odds}
        """
        result = self._call('listMarketBook', {
            'marketIds': [market_id],
            'priceProjection': {'priceData': ['EX_BEST_OFFERS']}
        })

        if not result or not result[0].get('runners'):
            return {}

        odds = {}
        for runner in result[0]['runners']:
            sel_id = runner['selectionId']
            back_prices = runner.get('ex', {}).get('availableToBack', [])
            if back_prices:
                odds[sel_id] = back_prices[0]['price']

        return odds

    def place_bet(self, market_id: str, selection_id: int, odds: float, stake: float) -> dict:
        """
        Place a back bet on Betfair.

        Args:
            market_id: Betfair market ID
            selection_id: Runner selection ID
            odds: Minimum odds to accept (will get best available >= this)
            stake: Stake in AUD

        Returns:
            Dict with status, bet_id, odds_matched, etc.
        """
        result = self._call('placeOrders', {
            'marketId': market_id,
            'instructions': [{
                'selectionId': selection_id,
                'handicap': '0',
                'side': 'BACK',
                'orderType': 'LIMIT',
                'limitOrder': {
                    'size': round(stake, 2),
                    'price': odds,
                    'persistenceType': 'LAPSE'  # Cancel if not matched
                }
            }]
        })

        if not result:
            return {'status': 'ERROR', 'message': 'API call failed'}

        report = result.get('instructionReports', [{}])[0]
        status = report.get('status', 'UNKNOWN')

        return {
            'status': status,
            'bet_id': report.get('betId'),
            'odds_matched': report.get('averagePriceMatched'),
            'size_matched': report.get('sizeMatched'),
            'error': report.get('errorCode'),
            'message': result.get('errorCode', '')
        }
