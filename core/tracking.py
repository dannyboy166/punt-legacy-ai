"""
Prediction Tracking System.

Stores predictions and outcomes to enable:
1. Performance analysis by tag ("Value pick" vs "The one to beat")
2. Confidence calibration (does high confidence = more wins?)
3. ROI tracking
4. Future prompt injection with historical performance data

Usage:
    from core.tracking import PredictionTracker

    tracker = PredictionTracker()

    # Store a prediction (call after each prediction)
    tracker.store_prediction(prediction_output, race_data)

    # Record outcome after race finishes
    tracker.record_outcome(prediction_id, won=True, placed=True, finishing_position=1)

    # Get analytics
    stats = tracker.get_stats_by_tag()
    print(stats)
    # {"The one to beat": {"total": 50, "wins": 19, "places": 28, "win_rate": 0.38, ...}}
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from core.logging import get_logger

logger = get_logger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "predictions.db"


@dataclass
class PredictionRecord:
    """A stored prediction for tracking."""
    id: int
    timestamp: str
    track: str
    race_number: int
    race_date: str
    horse: str
    tab_no: int
    odds: float
    place_odds: Optional[float]
    tag: str
    confidence: int
    race_confidence: int
    mode: str  # "normal" or "promo_bonus"
    pick_type: str  # "contender", "bonus_bet", "promo_play"

    # Outcome (filled in after race)
    won: Optional[bool] = None
    placed: Optional[bool] = None
    finishing_position: Optional[int] = None
    outcome_recorded: bool = False


class PredictionTracker:
    """
    Tracks predictions and outcomes for performance analysis.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track TEXT NOT NULL,
                    race_number INTEGER NOT NULL,
                    race_date TEXT NOT NULL,
                    horse TEXT NOT NULL,
                    tab_no INTEGER NOT NULL,
                    odds REAL NOT NULL,
                    place_odds REAL,
                    tag TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    race_confidence INTEGER NOT NULL,
                    confidence_reason TEXT,
                    mode TEXT NOT NULL,
                    pick_type TEXT NOT NULL,
                    analysis TEXT,
                    tipsheet_pick INTEGER DEFAULT 0,  -- 1 if Claude would genuinely bet
                    race_class TEXT,  -- e.g. "Maiden", "BM78", "Group 1"

                    -- Outcome (filled in after race)
                    won INTEGER,  -- 0 or 1
                    placed INTEGER,  -- 0 or 1
                    finishing_position INTEGER,
                    outcome_recorded INTEGER DEFAULT 0,

                    -- Unique constraint to prevent duplicates
                    UNIQUE(track, race_number, race_date, horse, mode, pick_type)
                )
            """)

            # Index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_date
                ON predictions(race_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_tag
                ON predictions(tag)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_outcome
                ON predictions(outcome_recorded)
            """)

            # Migration: Add tipsheet_pick column if it doesn't exist
            cursor = conn.execute("PRAGMA table_info(predictions)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'tipsheet_pick' not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN tipsheet_pick INTEGER DEFAULT 0")
                logger.info("Added tipsheet_pick column to predictions table")

            # Migration: Add race_class column if it doesn't exist
            if 'race_class' not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN race_class TEXT")
                logger.info("Added race_class column to predictions table")

            # Migration: Add pfai_rank column if it doesn't exist
            if 'pfai_rank' not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN pfai_rank INTEGER")
                logger.info("Added pfai_rank column to predictions table")

            conn.commit()

    def store_prediction(
        self,
        prediction_output,  # PredictionOutput from predictor.py
        race_data,  # RaceData from race_data.py
        race_date: str,
    ) -> list[int]:
        """
        Store a prediction in the database.

        Args:
            prediction_output: PredictionOutput from predictor
            race_data: RaceData object
            race_date: Race date in format "dd-MMM-yyyy"

        Returns:
            List of prediction IDs that were inserted
        """
        inserted_ids = []

        with sqlite3.connect(self.db_path) as conn:
            if prediction_output.mode == "normal":
                # Store each contender
                for contender in prediction_output.contenders:
                    # Get place odds from race_data
                    place_odds = None
                    for runner in race_data.runners:
                        if runner.tab_no == contender.tab_no:
                            place_odds = runner.place_odds
                            break

                    try:
                        # Use INSERT OR IGNORE to avoid overwriting outcome data
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO predictions
                            (timestamp, track, race_number, race_date, horse, tab_no,
                             odds, place_odds, tag, confidence, race_confidence,
                             confidence_reason, mode, pick_type, analysis, tipsheet_pick, race_class, pfai_rank)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().isoformat(),
                            prediction_output.track,
                            prediction_output.race_number,
                            race_date,
                            contender.horse,
                            contender.tab_no,
                            contender.odds,
                            place_odds,
                            contender.tag,
                            contender.confidence,
                            prediction_output.race_confidence,
                            prediction_output.confidence_reason,
                            "normal",
                            "contender",
                            contender.analysis,
                            1 if getattr(contender, 'tipsheet_pick', False) else 0,
                            race_data.class_ if race_data else None,
                            getattr(contender, 'pfai_rank', None),
                        ))
                        if cursor.rowcount > 0:
                            inserted_ids.append(cursor.lastrowid)
                    except sqlite3.IntegrityError:
                        logger.debug(f"Prediction already exists: {contender.horse}")

            elif prediction_output.mode == "promo_bonus":
                # Store bonus pick
                if prediction_output.bonus_pick:
                    pick = prediction_output.bonus_pick
                    place_odds = None
                    for runner in race_data.runners:
                        if runner.tab_no == pick.tab_no:
                            place_odds = runner.place_odds
                            break

                    try:
                        # Use INSERT OR IGNORE to avoid overwriting outcome data
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO predictions
                            (timestamp, track, race_number, race_date, horse, tab_no,
                             odds, place_odds, tag, confidence, race_confidence,
                             confidence_reason, mode, pick_type, analysis, tipsheet_pick, race_class)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().isoformat(),
                            prediction_output.track,
                            prediction_output.race_number,
                            race_date,
                            pick.horse,
                            pick.tab_no,
                            pick.odds,
                            place_odds,
                            "Bonus Bet",
                            5,  # Default confidence for promo/bonus
                            prediction_output.race_confidence,
                            prediction_output.confidence_reason,
                            "promo_bonus",
                            "bonus_bet",
                            pick.analysis,
                            0,  # tipsheet_pick not applicable for promo/bonus
                            race_data.class_ if race_data else None,
                        ))
                        if cursor.rowcount > 0:
                            inserted_ids.append(cursor.lastrowid)
                    except sqlite3.IntegrityError:
                        pass

                # Store promo pick
                if prediction_output.promo_pick:
                    pick = prediction_output.promo_pick
                    place_odds = None
                    for runner in race_data.runners:
                        if runner.tab_no == pick.tab_no:
                            place_odds = runner.place_odds
                            break

                    try:
                        # Use INSERT OR IGNORE to avoid overwriting outcome data
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO predictions
                            (timestamp, track, race_number, race_date, horse, tab_no,
                             odds, place_odds, tag, confidence, race_confidence,
                             confidence_reason, mode, pick_type, analysis, tipsheet_pick, race_class)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().isoformat(),
                            prediction_output.track,
                            prediction_output.race_number,
                            race_date,
                            pick.horse,
                            pick.tab_no,
                            pick.odds,
                            place_odds,
                            "Promo Play",
                            5,
                            prediction_output.race_confidence,
                            prediction_output.confidence_reason,
                            "promo_bonus",
                            "promo_play",
                            pick.analysis,
                            0,  # tipsheet_pick not applicable for promo/bonus
                            race_data.class_ if race_data else None,
                        ))
                        if cursor.rowcount > 0:
                            inserted_ids.append(cursor.lastrowid)
                    except sqlite3.IntegrityError:
                        pass

            conn.commit()

        logger.info(f"Stored {len(inserted_ids)} predictions for {prediction_output.track} R{prediction_output.race_number}")
        return inserted_ids

    def record_outcome(
        self,
        track: str,
        race_number: int,
        race_date: str,
        horse: str,
        won: bool,
        placed: bool,
        finishing_position: int,
    ) -> bool:
        """
        Record the outcome of a prediction.

        Args:
            track: Track name
            race_number: Race number
            race_date: Race date
            horse: Horse name
            won: Did the horse win?
            placed: Did the horse place (top 3)?
            finishing_position: Final position

        Returns:
            True if outcome was recorded, False if prediction not found
        """
        from core.normalize import normalize_horse_name

        with sqlite3.connect(self.db_path) as conn:
            # First try exact match
            cursor = conn.execute("""
                UPDATE predictions
                SET won = ?, placed = ?, finishing_position = ?, outcome_recorded = 1
                WHERE track = ? AND race_number = ? AND race_date = ? AND horse = ?
            """, (
                1 if won else 0,
                1 if placed else 0,
                finishing_position,
                track,
                race_number,
                race_date,
                horse,
            ))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Recorded outcome for {horse}: pos={finishing_position}")
                return True

            # Try case-insensitive match using LOWER()
            cursor = conn.execute("""
                UPDATE predictions
                SET won = ?, placed = ?, finishing_position = ?, outcome_recorded = 1
                WHERE track = ? AND race_number = ? AND race_date = ?
                AND LOWER(REPLACE(REPLACE(horse, '''', ''), ' ', '')) = LOWER(REPLACE(REPLACE(?, '''', ''), ' ', ''))
            """, (
                1 if won else 0,
                1 if placed else 0,
                finishing_position,
                track,
                race_number,
                race_date,
                horse,
            ))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Recorded outcome for {horse} (fuzzy match): pos={finishing_position}")
                return True

            # Try matching without date (handles timezone issues where stored date is off)
            # Only match pending predictions (outcome_recorded = 0) to avoid updating old races
            cursor = conn.execute("""
                UPDATE predictions
                SET won = ?, placed = ?, finishing_position = ?, outcome_recorded = 1
                WHERE track = ? AND race_number = ? AND outcome_recorded = 0
                AND LOWER(REPLACE(REPLACE(horse, '''', ''), ' ', '')) = LOWER(REPLACE(REPLACE(?, '''', ''), ' ', ''))
            """, (
                1 if won else 0,
                1 if placed else 0,
                finishing_position,
                track,
                race_number,
                horse,
            ))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Recorded outcome for {horse} (date-flexible match): pos={finishing_position}")
                return True

            return False

    def record_outcomes_bulk(
        self,
        track: str,
        race_number: int,
        race_date: str,
        results: dict[str, int],  # {horse_name: finishing_position}
    ) -> int:
        """
        Record outcomes for all predictions in a race.

        Args:
            track: Track name
            race_number: Race number
            race_date: Race date
            results: Dict of horse name -> finishing position

        Returns:
            Number of outcomes recorded
        """
        count = 0
        # Australian rules: 8+ runners = 1st/2nd/3rd pay place
        #                   5-7 runners = 1st/2nd only
        field_size = len(results)

        for horse, position in results.items():
            won = position == 1
            # Place only pays 3rd if 8+ runners
            if field_size >= 8:
                placed = position <= 3
            else:
                placed = position <= 2  # Only 1st/2nd pay place
            if self.record_outcome(track, race_number, race_date, horse, won, placed, position):
                count += 1
        return count

    def get_pending_outcomes(self, race_date: Optional[str] = None) -> list[dict]:
        """
        Get predictions that haven't had outcomes recorded yet.

        Args:
            race_date: Optional filter by date

        Returns:
            List of prediction dicts awaiting outcomes
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if race_date:
                rows = conn.execute("""
                    SELECT DISTINCT track, race_number, race_date, horse, tab_no, odds, tag
                    FROM predictions
                    WHERE outcome_recorded = 0 AND race_date = ?
                    ORDER BY race_date, track, race_number
                """, (race_date,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT DISTINCT track, race_number, race_date, horse, tab_no, odds, tag
                    FROM predictions
                    WHERE outcome_recorded = 0
                    ORDER BY race_date, track, race_number
                """).fetchall()

            return [dict(row) for row in rows]

    def get_stats_by_tag(self, min_samples: int = 5) -> dict:
        """
        Get performance statistics grouped by tag.

        Args:
            min_samples: Minimum samples required to include tag

        Returns:
            Dict of tag -> stats
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT
                    tag,
                    COUNT(*) as total,
                    SUM(won) as wins,
                    SUM(placed) as places,
                    AVG(odds) as avg_odds,
                    SUM(CASE WHEN won = 1 THEN odds ELSE 0 END) as total_returns
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY tag
                HAVING COUNT(*) >= ?
            """, (min_samples,)).fetchall()

            stats = {}
            for row in rows:
                tag, total, wins, places, avg_odds, total_returns = row
                wins = wins or 0
                places = places or 0
                total_returns = total_returns or 0

                # Calculate ROI (assuming $1 bets)
                # ROI = (total_returns - total_staked) / total_staked
                roi = (total_returns - total) / total if total > 0 else 0

                stats[tag] = {
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": avg_odds,
                    "roi": roi,
                }

            return stats

    def get_stats_by_confidence(self) -> dict:
        """
        Get performance statistics grouped by confidence level.

        Returns:
            Dict of confidence_range -> stats
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 8 THEN 'high (8-10)'
                        WHEN confidence >= 5 THEN 'medium (5-7)'
                        ELSE 'low (1-4)'
                    END as confidence_range,
                    COUNT(*) as total,
                    SUM(won) as wins,
                    SUM(placed) as places,
                    AVG(odds) as avg_odds,
                    SUM(CASE WHEN won = 1 THEN odds ELSE 0 END) as total_returns
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY confidence_range
            """).fetchall()

            stats = {}
            for row in rows:
                conf_range, total, wins, places, avg_odds, total_returns = row
                wins = wins or 0
                places = places or 0
                total_returns = total_returns or 0

                roi = (total_returns - total) / total if total > 0 else 0

                stats[conf_range] = {
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": avg_odds,
                    "roi": roi,
                }

            return stats

    def get_stats_by_mode(self) -> dict:
        """
        Get performance statistics grouped by mode (normal vs promo_bonus).

        Returns:
            Dict of mode -> stats
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT
                    mode,
                    pick_type,
                    COUNT(*) as total,
                    SUM(won) as wins,
                    SUM(placed) as places,
                    AVG(odds) as avg_odds,
                    SUM(CASE WHEN won = 1 THEN odds ELSE 0 END) as total_returns
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY mode, pick_type
            """).fetchall()

            stats = {}
            for row in rows:
                mode, pick_type, total, wins, places, avg_odds, total_returns = row
                wins = wins or 0
                places = places or 0
                total_returns = total_returns or 0

                # Calculate ROI
                roi = (total_returns - total) / total if total > 0 else 0

                key = f"{mode}:{pick_type}"
                stats[key] = {
                    "mode": mode,
                    "pick_type": pick_type,
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": avg_odds,
                    "roi": roi,
                }

            return stats

    def get_stats_by_race_confidence(self) -> dict:
        """
        Get performance statistics grouped by race-level confidence.

        Returns:
            Dict of race_confidence_range -> stats
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT
                    CASE
                        WHEN race_confidence >= 8 THEN 'high (8-10)'
                        WHEN race_confidence >= 5 THEN 'medium (5-7)'
                        ELSE 'low (1-4)'
                    END as confidence_range,
                    COUNT(*) as total,
                    SUM(won) as wins,
                    SUM(placed) as places,
                    AVG(odds) as avg_odds,
                    SUM(CASE WHEN won = 1 THEN odds ELSE 0 END) as total_returns
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY confidence_range
            """).fetchall()

            stats = {}
            for row in rows:
                conf_range, total, wins, places, avg_odds, total_returns = row
                wins = wins or 0
                places = places or 0
                total_returns = total_returns or 0

                roi = (total_returns - total) / total if total > 0 else 0

                stats[conf_range] = {
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": avg_odds,
                    "roi": roi,
                }

            return stats

    def get_summary(self) -> dict:
        """Get overall summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    SUM(outcome_recorded) as outcomes_recorded,
                    SUM(won) as total_wins,
                    SUM(placed) as total_places,
                    AVG(odds) as avg_odds
                FROM predictions
            """).fetchone()

            total, recorded, wins, places, avg_odds = row
            wins = wins or 0
            places = places or 0

            return {
                "total_predictions": total,
                "outcomes_recorded": recorded or 0,
                "pending_outcomes": total - (recorded or 0),
                "total_wins": wins,
                "total_places": places,
                "win_rate": wins / recorded if recorded else 0,
                "place_rate": places / recorded if recorded else 0,
                "avg_odds": avg_odds or 0,
            }

    def generate_prompt_context(self, min_samples: int = 20) -> str:
        """
        Generate historical context to inject into Claude's prompt.

        This enables the "learning" feedback loop.

        Args:
            min_samples: Minimum samples required before generating context

        Returns:
            String to add to system prompt, or empty string if not enough data
        """
        summary = self.get_summary()
        if summary["outcomes_recorded"] < min_samples:
            return ""

        tag_stats = self.get_stats_by_tag(min_samples=5)
        conf_stats = self.get_stats_by_confidence()

        lines = [
            "",
            "## Historical Performance Notes",
            f"Based on {summary['outcomes_recorded']} tracked predictions:",
            "",
        ]

        # Tag performance
        if tag_stats:
            lines.append("**By tag:**")
            for tag, stats in sorted(tag_stats.items(), key=lambda x: -x[1]["win_rate"]):
                roi_str = f"+{stats['roi']:.1%}" if stats['roi'] > 0 else f"{stats['roi']:.1%}"
                lines.append(
                    f"- '{tag}': {stats['win_rate']:.0%} win rate "
                    f"(avg ${stats['avg_odds']:.2f}, ROI {roi_str})"
                )
            lines.append("")

        # Confidence performance
        if conf_stats:
            lines.append("**By confidence level:**")
            for level in ["high (8-10)", "medium (5-7)", "low (1-4)"]:
                if level in conf_stats:
                    stats = conf_stats[level]
                    lines.append(
                        f"- {level}: {stats['win_rate']:.0%} win rate"
                    )
            lines.append("")

        lines.append("Adjust selections based on this historical data.")

        return "\n".join(lines)

    def get_recent_predictions(self, limit: int = 50) -> list[dict]:
        """Get recent predictions for display."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT *
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [dict(row) for row in rows]

    def get_stats_by_tag_with_staking(self, min_samples: int = 1) -> dict:
        """
        Get performance statistics grouped by tag with staking calculations.

        Returns for each tag:
        - total, wins, places, win_rate, place_rate
        - flat_bet: 1u each horse
        - fixed_return: $100 target return per bet
        - each_way: 1u win + 2u place (for "Each-way chance" tag)

        Args:
            min_samples: Minimum samples required to include tag

        Returns:
            Dict of tag -> stats including staking ROI
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    tag,
                    odds,
                    place_odds,
                    won,
                    placed
                FROM predictions
                WHERE outcome_recorded = 1
            """).fetchall()

            # Group by tag
            by_tag = {}
            for row in rows:
                tag = row['tag']
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append({
                    'odds': row['odds'],
                    'place_odds': row['place_odds'],
                    'won': row['won'],
                    'placed': row['placed']
                })

            stats = {}
            for tag, predictions in by_tag.items():
                if len(predictions) < min_samples:
                    continue

                total = len(predictions)
                wins = sum(1 for p in predictions if p['won'])
                places = sum(1 for p in predictions if p['placed'])

                # ============================================
                # FLAT BET: 1 unit each horse
                # ============================================
                flat_profit = 0
                flat_staked = 0
                for p in predictions:
                    flat_staked += 1
                    if p['won']:
                        flat_profit += p['odds'] - 1
                    else:
                        flat_profit -= 1
                flat_roi = (flat_profit / flat_staked * 100) if flat_staked > 0 else 0

                # ============================================
                # FIXED RETURN: $100 target return per bet
                # ============================================
                fixed_profit = 0
                fixed_staked = 0
                for p in predictions:
                    # Stake to win $100
                    stake = 100 / (p['odds'] - 1) if p['odds'] > 1 else 100
                    fixed_staked += stake
                    if p['won']:
                        fixed_profit += 100  # Won $100
                    else:
                        fixed_profit -= stake  # Lost stake
                fixed_roi = (fixed_profit / fixed_staked * 100) if fixed_staked > 0 else 0

                # ============================================
                # EACH-WAY: 1u win + 2u place
                # (Only meaningful for "Each-way chance" tag)
                # ============================================
                ew_profit = 0
                ew_staked = 0
                for p in predictions:
                    ew_staked += 3  # 1u win + 2u place

                    # Win component: 1u @ win odds
                    if p['won']:
                        ew_profit += p['odds'] - 1
                    else:
                        ew_profit -= 1

                    # Place component: 2u @ place odds
                    place_odds = p['place_odds'] or (1 + (p['odds'] - 1) * 0.35)
                    if p['placed']:
                        ew_profit += 2 * (place_odds - 1)
                    else:
                        ew_profit -= 2

                ew_roi = (ew_profit / ew_staked * 100) if ew_staked > 0 else 0

                stats[tag] = {
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": sum(p['odds'] for p in predictions) / total,

                    # Staking results
                    "flat_bet": {
                        "staked": round(flat_staked, 2),
                        "profit": round(flat_profit, 2),
                        "roi": round(flat_roi, 1)
                    },
                    "fixed_return": {
                        "staked": round(fixed_staked, 2),
                        "profit": round(fixed_profit, 2),
                        "roi": round(fixed_roi, 1)
                    },
                    "each_way": {
                        "staked": round(ew_staked, 2),
                        "profit": round(ew_profit, 2),
                        "roi": round(ew_roi, 1)
                    }
                }

            return stats

    def get_stats_by_tag_and_tipsheet(self) -> dict:
        """
        Get performance statistics grouped by tag, split by tipsheet_pick.

        Returns dict of tag -> {starred: {...}, regular: {...}}
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    tag,
                    tipsheet_pick,
                    odds,
                    won,
                    placed
                FROM predictions
                WHERE outcome_recorded = 1
            """).fetchall()

            # Group by tag and tipsheet
            by_tag: dict[str, dict[str, list]] = {}
            for row in rows:
                tag = row['tag']
                is_star = row['tipsheet_pick'] == 1
                key = 'starred' if is_star else 'regular'

                if tag not in by_tag:
                    by_tag[tag] = {'starred': [], 'regular': []}
                by_tag[tag][key].append({
                    'odds': row['odds'],
                    'won': row['won'],
                    'placed': row['placed']
                })

            stats = {}
            for tag, groups in by_tag.items():
                tag_stats = {}
                for group_name, predictions in groups.items():
                    if not predictions:
                        continue

                    total = len(predictions)
                    wins = sum(1 for p in predictions if p['won'])
                    places = sum(1 for p in predictions if p['placed'])
                    avg_odds = sum(p['odds'] for p in predictions) / total

                    # Flat bet ROI
                    profit = sum(p['odds'] - 1 if p['won'] else -1 for p in predictions)
                    roi = (profit / total * 100) if total > 0 else 0

                    tag_stats[group_name] = {
                        'total': total,
                        'wins': wins,
                        'places': places,
                        'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                        'place_rate': round(places / total * 100, 1) if total > 0 else 0,
                        'avg_odds': round(avg_odds, 2),
                        'profit': round(profit, 2),
                        'roi': round(roi, 1)
                    }

                if tag_stats:
                    stats[tag] = tag_stats

            return stats

    def get_stats_by_metro(self, tag: Optional[str] = None) -> dict:
        """
        Get performance statistics split by metro vs non-metro tracks.

        Metro tracks: Randwick, Rosehill, Canterbury, Warwick Farm, Flemington,
        Caulfield, Moonee Valley, Sandown, Pakenham, Eagle Farm, Doomben,
        Gold Coast, Morphettville, Ascot, Belmont.
        """
        metro_tracks = {
            "randwick", "rosehill", "canterbury", "warwick farm", "royal randwick",
            "flemington", "caulfield", "moonee valley", "sandown", "sandown-hillside",
            "sandown-lakeside", "pakenham",
            "eagle farm", "doomben", "gold coast",
            "morphettville", "morphettville parks",
            "ascot", "belmont", "belmont park",
        }

        def is_metro(track_name: str) -> bool:
            track_lower = track_name.lower().strip()
            return any(metro in track_lower for metro in metro_tracks)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT track, tag, odds, won, placed, tipsheet_pick
                FROM predictions
                WHERE outcome_recorded = 1
            """
            if tag:
                query += f" AND tag = '{tag}'"

            rows = conn.execute(query).fetchall()

            # Group by metro/non-metro
            metro_preds = []
            non_metro_preds = []

            for row in rows:
                pred = {
                    'odds': row['odds'],
                    'won': row['won'],
                    'placed': row['placed'],
                    'tipsheet_pick': row['tipsheet_pick']
                }
                if is_metro(row['track']):
                    metro_preds.append(pred)
                else:
                    non_metro_preds.append(pred)

            def calc_stats(predictions: list) -> dict:
                if not predictions:
                    return None
                total = len(predictions)
                wins = sum(1 for p in predictions if p['won'])
                places = sum(1 for p in predictions if p['placed'])
                starred = sum(1 for p in predictions if p['tipsheet_pick'])
                avg_odds = sum(p['odds'] for p in predictions) / total
                profit = sum(p['odds'] - 1 if p['won'] else -1 for p in predictions)
                roi = (profit / total * 100) if total > 0 else 0

                return {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'starred': starred,
                    'win_rate': round(wins / total * 100, 1),
                    'place_rate': round(places / total * 100, 1),
                    'avg_odds': round(avg_odds, 2),
                    'profit': round(profit, 2),
                    'roi': round(roi, 1)
                }

            return {
                'metro': calc_stats(metro_preds),
                'non_metro': calc_stats(non_metro_preds)
            }

    def get_stats_by_meeting(self) -> list[dict]:
        """
        Get performance statistics grouped by meeting (track + date).

        Returns list of meetings with stats per tag and overall profit/loss.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    track,
                    race_date,
                    tag,
                    odds,
                    place_odds,
                    won,
                    placed
                FROM predictions
                WHERE outcome_recorded = 1
                ORDER BY race_date DESC, track
            """).fetchall()

            # Group by meeting (track + date)
            meetings = {}
            for row in rows:
                key = f"{row['track']}|{row['race_date']}"
                if key not in meetings:
                    meetings[key] = {
                        'track': row['track'],
                        'date': row['race_date'],
                        'picks': [],
                    }
                meetings[key]['picks'].append({
                    'tag': row['tag'],
                    'odds': row['odds'],
                    'place_odds': row['place_odds'],
                    'won': row['won'],
                    'placed': row['placed']
                })

            # Calculate stats for each meeting
            result = []
            for key, meeting in meetings.items():
                picks = meeting['picks']
                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])

                # Overall flat bet profit
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)

                # Stats by tag
                by_tag = {}
                for p in picks:
                    tag = p['tag']
                    if tag not in by_tag:
                        by_tag[tag] = {'total': 0, 'wins': 0, 'places': 0, 'profit': 0}
                    by_tag[tag]['total'] += 1
                    if p['won']:
                        by_tag[tag]['wins'] += 1
                        by_tag[tag]['profit'] += p['odds'] - 1
                    else:
                        by_tag[tag]['profit'] -= 1
                    if p['placed']:
                        by_tag[tag]['places'] += 1

                result.append({
                    'track': meeting['track'],
                    'date': meeting['date'],
                    'total_picks': total,
                    'wins': wins,
                    'places': places,
                    'win_rate': wins / total if total > 0 else 0,
                    'place_rate': places / total if total > 0 else 0,
                    'flat_profit': round(flat_profit, 2),
                    'by_tag': {
                        tag: {
                            'total': s['total'],
                            'wins': s['wins'],
                            'places': s['places'],
                            'profit': round(s['profit'], 2)
                        }
                        for tag, s in by_tag.items()
                    }
                })

            return result

    def get_stats_by_day(self) -> list[dict]:
        """
        Get performance statistics grouped by day (aggregated across all tracks).

        Similar to get_stats_by_meeting() but aggregates ALL tracks for each date,
        providing a per-day view of performance.

        Returns list of days with:
        - date
        - total_picks, wins, places, win_rate, place_rate
        - flat_profit (1u per pick)
        - tracks: list of track names that had picks
        - by_tag: breakdown by tag
        - starred_picks: count of tipsheet_pick=1
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    race_date,
                    track,
                    tag,
                    odds,
                    place_odds,
                    won,
                    placed,
                    tipsheet_pick
                FROM predictions
                WHERE outcome_recorded = 1
                ORDER BY race_date DESC
            """).fetchall()

            # Group by date
            days: dict[str, dict] = {}
            for row in rows:
                date = row['race_date']
                if date not in days:
                    days[date] = {
                        'date': date,
                        'tracks': set(),
                        'picks': [],
                    }
                days[date]['tracks'].add(row['track'])
                days[date]['picks'].append({
                    'track': row['track'],
                    'tag': row['tag'],
                    'odds': row['odds'],
                    'place_odds': row['place_odds'],
                    'won': row['won'],
                    'placed': row['placed'],
                    'tipsheet_pick': row['tipsheet_pick']
                })

            # Calculate stats for each day
            result = []
            for date, day_data in days.items():
                picks = day_data['picks']
                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])
                starred = sum(1 for p in picks if p['tipsheet_pick'] == 1)

                # Overall flat bet profit
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)

                # Stats by tag
                by_tag: dict[str, dict] = {}
                for p in picks:
                    tag = p['tag']
                    if tag not in by_tag:
                        by_tag[tag] = {'total': 0, 'wins': 0, 'places': 0, 'profit': 0}
                    by_tag[tag]['total'] += 1
                    if p['won']:
                        by_tag[tag]['wins'] += 1
                        by_tag[tag]['profit'] += p['odds'] - 1
                    else:
                        by_tag[tag]['profit'] -= 1
                    if p['placed']:
                        by_tag[tag]['places'] += 1

                result.append({
                    'date': date,
                    'tracks': list(day_data['tracks']),
                    'total_picks': total,
                    'wins': wins,
                    'places': places,
                    'win_rate': wins / total if total > 0 else 0,
                    'place_rate': places / total if total > 0 else 0,
                    'flat_profit': round(flat_profit, 2),
                    'starred_picks': starred,
                    'by_tag': {
                        tag: {
                            'total': s['total'],
                            'wins': s['wins'],
                            'places': s['places'],
                            'profit': round(s['profit'], 2)
                        }
                        for tag, s in by_tag.items()
                    }
                })

            # Sort by date descending
            result.sort(key=lambda x: x['date'], reverse=True)
            return result

    def get_picks_for_day(self, race_date: str) -> list[dict]:
        """
        Get all individual picks for a specific day.

        Used to expand a day row and see the actual predictions.

        Args:
            race_date: Date in dd-MMM-yyyy format

        Returns:
            List of pick details with track, race, horse, odds, result
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    track,
                    race_number,
                    horse,
                    tab_no,
                    odds,
                    place_odds,
                    tag,
                    tipsheet_pick,
                    won,
                    placed,
                    finishing_position
                FROM predictions
                WHERE race_date = ?
                ORDER BY track, race_number, tab_no
            """, (race_date,)).fetchall()

            return [dict(row) for row in rows]

    def clear_all(self) -> int:
        """
        Delete all prediction records from the database.
        WARNING: This permanently deletes all tracking data!

        Returns:
            Number of records deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM predictions")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM predictions")
            conn.commit()
            logger.warning(f"Cleared all {count} predictions from tracking database")
            return count

    def get_stats_by_tipsheet(self) -> dict:
        """
        Get performance statistics comparing tipsheet_pick=true vs tipsheet_pick=false.

        Returns dict with 'tipsheet' and 'non_tipsheet' stats.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    tipsheet_pick,
                    COUNT(*) as total,
                    SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN placed = 1 THEN 1 ELSE 0 END) as places,
                    AVG(odds) as avg_odds,
                    SUM(CASE WHEN won = 1 THEN odds - 1 ELSE -1 END) as flat_profit
                FROM predictions
                WHERE outcome_recorded = 1
                  AND mode = 'normal'
                GROUP BY tipsheet_pick
            """).fetchall()

            result = {}
            for row in rows:
                key = 'tipsheet' if row['tipsheet_pick'] == 1 else 'non_tipsheet'
                total = row['total']
                wins = row['wins'] or 0
                places = row['places'] or 0
                flat_profit = row['flat_profit'] or 0

                result[key] = {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'win_rate': wins / total if total > 0 else 0,
                    'place_rate': places / total if total > 0 else 0,
                    'avg_odds': row['avg_odds'] or 0,
                    'flat_profit': round(flat_profit, 2),
                    'roi': round((flat_profit / total) * 100, 1) if total > 0 else 0,
                }

            return result

    def get_stats_by_class(self, min_samples: int = 1, tag: Optional[str] = None) -> dict:
        """
        Get performance statistics grouped by race class.

        Args:
            min_samples: Minimum picks required to include a class
            tag: Optional tag filter (e.g., "The one to beat")

        Returns dict of race_class -> stats with win/place rates and ROI.
        Race classes are normalized to groups like "Maiden", "Class 1-3", "BM50-65", etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if tag:
                rows = conn.execute("""
                    SELECT
                        race_class,
                        odds,
                        won,
                        placed
                    FROM predictions
                    WHERE outcome_recorded = 1 AND race_class IS NOT NULL AND tag = ?
                """, (tag,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT
                        race_class,
                        odds,
                        won,
                        placed
                    FROM predictions
                    WHERE outcome_recorded = 1 AND race_class IS NOT NULL
                """).fetchall()

            # Group by normalized class
            by_class: dict[str, list] = {}
            for row in rows:
                raw_class = row['race_class'] or "Unknown"
                # Normalize class name
                normalized = self._normalize_race_class(raw_class)
                if normalized not in by_class:
                    by_class[normalized] = []
                by_class[normalized].append({
                    'odds': row['odds'],
                    'won': row['won'],
                    'placed': row['placed']
                })

            # Calculate stats for each class
            stats = {}
            for class_name, picks in by_class.items():
                if len(picks) < min_samples:
                    continue

                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])
                avg_odds = sum(p['odds'] for p in picks) / total

                # Flat bet profit/ROI
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)
                roi = (flat_profit / total * 100) if total > 0 else 0

                stats[class_name] = {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    'place_rate': round(places / total * 100, 1) if total > 0 else 0,
                    'avg_odds': round(avg_odds, 2),
                    'flat_profit': round(flat_profit, 2),
                    'roi': round(roi, 1),
                }

            return stats

    def _normalize_race_class(self, raw_class: str) -> str:
        """
        Normalize race class string to a grouping.

        Examples:
        - "Maiden;" -> "Maiden"
        - "Class 1;" -> "Class 1-3"
        - "Benchmark 65;" -> "BM58-72"
        - "Group 1;" -> "Group 1"
        """
        raw = raw_class.strip().rstrip(';').lower()

        # Group races
        if 'group 1' in raw:
            return "Group 1"
        if 'group 2' in raw:
            return "Group 2"
        if 'group 3' in raw:
            return "Group 3"
        if 'listed' in raw:
            return "Listed"

        # Maiden
        if 'maiden' in raw:
            return "Maiden"

        # Class races
        if 'class 1' in raw:
            return "Class 1-3"
        if 'class 2' in raw:
            return "Class 1-3"
        if 'class 3' in raw:
            return "Class 1-3"
        if 'class 4' in raw:
            return "Class 4-6"
        if 'class 5' in raw:
            return "Class 4-6"
        if 'class 6' in raw:
            return "Class 4-6"

        # Benchmark races - extract the number
        import re
        bm_match = re.search(r'benchmark\s*(\d+)', raw)
        if bm_match:
            bm_num = int(bm_match.group(1))
            if bm_num <= 58:
                return "BM45-58"
            elif bm_num <= 72:
                return "BM58-72"
            elif bm_num <= 85:
                return "BM72-85"
            else:
                return "BM85+"

        # BM shorthand
        bm_match = re.search(r'bm\s*(\d+)', raw)
        if bm_match:
            bm_num = int(bm_match.group(1))
            if bm_num <= 58:
                return "BM45-58"
            elif bm_num <= 72:
                return "BM58-72"
            elif bm_num <= 85:
                return "BM72-85"
            else:
                return "BM85+"

        # Handicap
        if 'handicap' in raw:
            return "Handicap"

        # Restricted/age races
        if '2yo' in raw or '2 yo' in raw or '2-yo' in raw:
            return "2YO"
        if '3yo' in raw or '3 yo' in raw or '3-yo' in raw:
            return "3YO"

        # Default
        return raw_class.strip().rstrip(';')

    def get_stats_by_pfai_rank(self, tag: Optional[str] = None) -> dict:
        """
        Get performance statistics grouped by PFAI rank.

        Args:
            tag: Optional tag filter (e.g., "The one to beat")

        Returns:
            Dict of pfai_rank -> stats with win/place rates and ROI.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if tag:
                rows = conn.execute("""
                    SELECT
                        pfai_rank,
                        odds,
                        won,
                        placed
                    FROM predictions
                    WHERE outcome_recorded = 1 AND pfai_rank IS NOT NULL AND tag = ?
                """, (tag,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT
                        pfai_rank,
                        odds,
                        won,
                        placed
                    FROM predictions
                    WHERE outcome_recorded = 1 AND pfai_rank IS NOT NULL
                """).fetchall()

            # Group by PFAI rank
            by_rank: dict[int, list] = {}
            for row in rows:
                rank = row['pfai_rank']
                if rank not in by_rank:
                    by_rank[rank] = []
                by_rank[rank].append({
                    'odds': row['odds'],
                    'won': row['won'],
                    'placed': row['placed']
                })

            # Calculate stats for each rank
            stats = {}
            for rank, picks in sorted(by_rank.items()):
                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])
                avg_odds = sum(p['odds'] for p in picks) / total if total > 0 else 0

                # Flat bet profit/ROI
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)
                roi = (flat_profit / total * 100) if total > 0 else 0

                stats[rank] = {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    'place_rate': round(places / total * 100, 1) if total > 0 else 0,
                    'avg_odds': round(avg_odds, 2),
                    'flat_profit': round(flat_profit, 2),
                    'roi': round(roi, 1),
                }

            return stats

    def get_stats_by_tag_and_pfai(self, pfai_rank: int = 1) -> dict:
        """
        Get performance of each tag filtered to a specific PFAI rank.

        This is useful to see how our "The one to beat" picks perform when
        they're also PFAI Rank 1 (consensus picks).

        Args:
            pfai_rank: PFAI rank to filter (default 1 = PFAI's top pick)

        Returns:
            Dict of tag -> stats for picks where pfai_rank matches.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    tag,
                    odds,
                    won,
                    placed,
                    tipsheet_pick
                FROM predictions
                WHERE outcome_recorded = 1 AND pfai_rank = ?
            """, (pfai_rank,)).fetchall()

            # Group by tag
            by_tag: dict[str, list] = {}
            for row in rows:
                tag = row['tag']
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append({
                    'odds': row['odds'],
                    'won': row['won'],
                    'placed': row['placed'],
                    'tipsheet_pick': row['tipsheet_pick']
                })

            # Calculate stats for each tag
            stats = {}
            for tag, picks in by_tag.items():
                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])
                starred = sum(1 for p in picks if p['tipsheet_pick'] == 1)
                avg_odds = sum(p['odds'] for p in picks) / total if total > 0 else 0

                # Flat bet profit/ROI
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)
                roi = (flat_profit / total * 100) if total > 0 else 0

                stats[tag] = {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'starred': starred,
                    'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    'place_rate': round(places / total * 100, 1) if total > 0 else 0,
                    'avg_odds': round(avg_odds, 2),
                    'flat_profit': round(flat_profit, 2),
                    'roi': round(roi, 1),
                }

            return stats

    def get_consensus_picks_stats(self) -> dict:
        """
        Get performance of "The one to beat" picks that are also PFAI Rank 1.

        These are "consensus" picks where both AIs agree - potential tipsheet picks.

        Returns:
            Dict with stats for:
            - consensus: "The one to beat" + PFAI Rank 1
            - our_ai_only: "The one to beat" but NOT PFAI Rank 1
            - pfai_only: PFAI Rank 1 but NOT "The one to beat"
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    tag,
                    pfai_rank,
                    odds,
                    won,
                    placed,
                    tipsheet_pick
                FROM predictions
                WHERE outcome_recorded = 1
            """).fetchall()

            # Group into categories
            categories = {
                'consensus': [],      # Both agree
                'our_ai_only': [],    # Our pick, not PFAI #1
                'pfai_only': [],      # PFAI #1, not our top pick
            }

            for row in rows:
                is_our_top = row['tag'] == "The one to beat"
                is_pfai_top = row['pfai_rank'] == 1

                if is_our_top and is_pfai_top:
                    categories['consensus'].append(row)
                elif is_our_top and not is_pfai_top:
                    categories['our_ai_only'].append(row)
                elif is_pfai_top and not is_our_top:
                    categories['pfai_only'].append(row)

            # Calculate stats for each category
            stats = {}
            for cat_name, picks in categories.items():
                if not picks:
                    stats[cat_name] = {'total': 0}
                    continue

                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])
                starred = sum(1 for p in picks if p['tipsheet_pick'] == 1)
                avg_odds = sum(p['odds'] for p in picks) / total

                # Flat bet profit/ROI
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)
                roi = (flat_profit / total * 100) if total > 0 else 0

                stats[cat_name] = {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'starred': starred,
                    'win_rate': round(wins / total * 100, 1),
                    'place_rate': round(places / total * 100, 1),
                    'avg_odds': round(avg_odds, 2),
                    'flat_profit': round(flat_profit, 2),
                    'roi': round(roi, 1),
                }

            return stats

    def get_stats_by_odds_range(self, tag: Optional[str] = None, starred_only: bool = False) -> dict:
        """
        Get performance statistics grouped by odds range.

        Args:
            tag: Optional filter by tag (e.g., "The one to beat")
            starred_only: If True, only include tipsheet_pick=1

        Returns dict of odds_range -> stats
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT
                    odds,
                    won,
                    placed,
                    tipsheet_pick
                FROM predictions
                WHERE outcome_recorded = 1
            """
            params = []

            if tag:
                query += " AND tag = ?"
                params.append(tag)

            if starred_only:
                query += " AND tipsheet_pick = 1"

            rows = conn.execute(query, params).fetchall()

            # Define odds ranges
            def get_odds_range(odds: float) -> str:
                if odds < 2.0:
                    return "$1.01-$1.99"
                elif odds < 3.0:
                    return "$2.00-$2.99"
                elif odds < 4.0:
                    return "$3.00-$3.99"
                elif odds < 5.0:
                    return "$4.00-$4.99"
                elif odds < 7.0:
                    return "$5.00-$6.99"
                elif odds < 10.0:
                    return "$7.00-$9.99"
                elif odds < 15.0:
                    return "$10.00-$14.99"
                else:
                    return "$15.00+"

            # Group by odds range
            by_range: dict[str, list] = {}
            for row in rows:
                odds_range = get_odds_range(row['odds'])
                if odds_range not in by_range:
                    by_range[odds_range] = []
                by_range[odds_range].append({
                    'odds': row['odds'],
                    'won': row['won'],
                    'placed': row['placed'],
                    'tipsheet_pick': row['tipsheet_pick']
                })

            # Calculate stats for each range
            stats = {}
            # Sort by odds range
            range_order = [
                "$1.01-$1.99", "$2.00-$2.99", "$3.00-$3.99", "$4.00-$4.99",
                "$5.00-$6.99", "$7.00-$9.99", "$10.00-$14.99", "$15.00+"
            ]

            for odds_range in range_order:
                if odds_range not in by_range:
                    continue

                picks = by_range[odds_range]
                total = len(picks)
                wins = sum(1 for p in picks if p['won'])
                places = sum(1 for p in picks if p['placed'])
                starred = sum(1 for p in picks if p['tipsheet_pick'] == 1)
                avg_odds = sum(p['odds'] for p in picks) / total

                # Flat bet profit/ROI
                flat_profit = sum(p['odds'] - 1 if p['won'] else -1 for p in picks)
                roi = (flat_profit / total * 100) if total > 0 else 0

                stats[odds_range] = {
                    'total': total,
                    'wins': wins,
                    'places': places,
                    'starred': starred,
                    'win_rate': round(wins / total * 100, 1),
                    'place_rate': round(places / total * 100, 1),
                    'avg_odds': round(avg_odds, 2),
                    'flat_profit': round(flat_profit, 2),
                    'roi': round(roi, 1),
                }

            return stats
