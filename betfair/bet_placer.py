"""
Safe bet placement with limits and logging.

Usage:
    from betfair.bet_placer import BetPlacer

    placer = BetPlacer()
    placer.login()

    # Place a bet with safety checks
    result = placer.bet("Randwick", 5, "04-May-2026", "Horse Name", stake=10.0)
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from betfair.client import BetfairClient


# Safety limits
DEFAULT_MAX_STAKE = 20.0        # Max per bet
DEFAULT_MAX_DAILY_LOSS = 100.0  # Stop after this daily loss
DEFAULT_MIN_ODDS = 1.50         # Don't bet below this
DEFAULT_MAX_ODDS = 50.0         # Don't bet above this

LOG_FILE = Path(__file__).parent / 'bet_log.csv'


class BetPlacer:
    """Safe bet placement with limits and logging."""

    def __init__(
        self,
        max_stake: float = DEFAULT_MAX_STAKE,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        min_odds: float = DEFAULT_MIN_ODDS,
        max_odds: float = DEFAULT_MAX_ODDS,
        dry_run: bool = True  # Default to dry run (no real bets)
    ):
        self.client = BetfairClient()
        self.max_stake = max_stake
        self.max_daily_loss = max_daily_loss
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.dry_run = dry_run
        self.daily_pnl = 0.0
        self._init_log()

    def _init_log(self):
        """Create CSV log file if it doesn't exist."""
        if not LOG_FILE.exists():
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'date', 'track', 'race', 'horse',
                    'market_id', 'selection_id', 'odds_requested', 'odds_matched',
                    'stake', 'bet_id', 'status', 'dry_run', 'notes'
                ])

    def _log_bet(self, **kwargs):
        """Append a bet to the log."""
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                kwargs.get('date', ''),
                kwargs.get('track', ''),
                kwargs.get('race', ''),
                kwargs.get('horse', ''),
                kwargs.get('market_id', ''),
                kwargs.get('selection_id', ''),
                kwargs.get('odds_requested', ''),
                kwargs.get('odds_matched', ''),
                kwargs.get('stake', ''),
                kwargs.get('bet_id', ''),
                kwargs.get('status', ''),
                kwargs.get('dry_run', self.dry_run),
                kwargs.get('notes', ''),
            ])

    def login(self) -> bool:
        """Login to Betfair."""
        return self.client.login()

    def bet(
        self,
        track: str,
        race_number: int,
        date: str,
        horse_name: str,
        stake: float = 10.0,
        min_odds: Optional[float] = None
    ) -> dict:
        """
        Place a bet on a horse with safety checks.

        Args:
            track: Track name (e.g., "Randwick")
            race_number: Race number
            date: Date (e.g., "04-May-2026")
            horse_name: Horse name to bet on
            stake: Bet stake in AUD
            min_odds: Minimum acceptable odds (uses class default if None)

        Returns:
            Dict with status and details
        """
        min_odds = min_odds or self.min_odds

        # Safety check: stake
        if stake > self.max_stake:
            msg = f"Stake ${stake} exceeds max ${self.max_stake}"
            print(f"BLOCKED: {msg}")
            return {'status': 'BLOCKED', 'message': msg}

        # Safety check: daily loss
        if self.daily_pnl <= -self.max_daily_loss:
            msg = f"Daily loss limit reached (${abs(self.daily_pnl):.2f} lost)"
            print(f"BLOCKED: {msg}")
            return {'status': 'BLOCKED', 'message': msg}

        # Find the market
        markets = self.client.find_race(track, date)
        if not markets:
            msg = f"No markets found for {track} on {date}"
            print(f"ERROR: {msg}")
            return {'status': 'ERROR', 'message': msg}

        # Match race number from market name (e.g., "R5 1200m Mdn" or "5 1200m")
        market = None
        for m in markets:
            name = m['name']
            # Try to extract race number from market name
            if f'R{race_number} ' in name or name.startswith(f'{race_number} '):
                market = m
                break

        if not market:
            # Fallback: use position in sorted list
            if race_number <= len(markets):
                market = markets[race_number - 1]
            else:
                msg = f"Race {race_number} not found at {track}"
                print(f"ERROR: {msg}")
                return {'status': 'ERROR', 'message': msg}

        # Find horse in runners
        selection_id = None
        matched_name = None
        horse_lower = horse_name.lower().strip()

        for name, sel_id in market['runners'].items():
            if name.lower().strip() == horse_lower:
                selection_id = sel_id
                matched_name = name
                break
            # Fuzzy: check if horse name is contained
            if horse_lower in name.lower() or name.lower() in horse_lower:
                selection_id = sel_id
                matched_name = name
                break

        if not selection_id:
            msg = f"Horse '{horse_name}' not found in {track} R{race_number}. Runners: {list(market['runners'].keys())}"
            print(f"ERROR: {msg}")
            return {'status': 'ERROR', 'message': msg}

        # Get current odds
        odds_map = self.client.get_market_odds(market['market_id'])
        current_odds = odds_map.get(selection_id)

        if not current_odds:
            msg = f"No odds available for {matched_name}"
            print(f"ERROR: {msg}")
            return {'status': 'ERROR', 'message': msg}

        # Safety check: odds range
        if current_odds < min_odds:
            msg = f"Odds ${current_odds} below minimum ${min_odds}"
            print(f"BLOCKED: {msg}")
            return {'status': 'BLOCKED', 'message': msg}

        if current_odds > self.max_odds:
            msg = f"Odds ${current_odds} above maximum ${self.max_odds}"
            print(f"BLOCKED: {msg}")
            return {'status': 'BLOCKED', 'message': msg}

        print(f"\n{'='*50}")
        print(f"{'DRY RUN - ' if self.dry_run else ''}BET: {matched_name}")
        print(f"Track: {track} R{race_number} | Date: {date}")
        print(f"Odds: ${current_odds:.2f} | Stake: ${stake:.2f}")
        print(f"Potential return: ${current_odds * stake:.2f}")
        print(f"Market: {market['market_id']}")
        print(f"{'='*50}\n")

        if self.dry_run:
            self._log_bet(
                date=date, track=track, race=race_number, horse=matched_name,
                market_id=market['market_id'], selection_id=selection_id,
                odds_requested=current_odds, odds_matched=None,
                stake=stake, bet_id=None, status='DRY_RUN',
                dry_run=True, notes='Paper trade'
            )
            return {
                'status': 'DRY_RUN',
                'horse': matched_name,
                'odds': current_odds,
                'stake': stake,
                'market_id': market['market_id']
            }

        # REAL BET
        result = self.client.place_bet(
            market_id=market['market_id'],
            selection_id=selection_id,
            odds=current_odds,
            stake=stake
        )

        self._log_bet(
            date=date, track=track, race=race_number, horse=matched_name,
            market_id=market['market_id'], selection_id=selection_id,
            odds_requested=current_odds,
            odds_matched=result.get('odds_matched'),
            stake=stake, bet_id=result.get('bet_id'),
            status=result.get('status'), dry_run=False,
            notes=result.get('error', '')
        )

        if result['status'] == 'SUCCESS':
            print(f"BET PLACED: {matched_name} @ ${result.get('odds_matched', current_odds):.2f}")
        else:
            print(f"BET FAILED: {result.get('error', result.get('message', 'Unknown error'))}")

        return result
