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
                        cursor = conn.execute("""
                            INSERT OR REPLACE INTO predictions
                            (timestamp, track, race_number, race_date, horse, tab_no,
                             odds, place_odds, tag, confidence, race_confidence,
                             confidence_reason, mode, pick_type, analysis)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ))
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
                        cursor = conn.execute("""
                            INSERT OR REPLACE INTO predictions
                            (timestamp, track, race_number, race_date, horse, tab_no,
                             odds, place_odds, tag, confidence, race_confidence,
                             confidence_reason, mode, pick_type, analysis)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ))
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
                        cursor = conn.execute("""
                            INSERT OR REPLACE INTO predictions
                            (timestamp, track, race_number, race_date, horse, tab_no,
                             odds, place_odds, tag, confidence, race_confidence,
                             confidence_reason, mode, pick_type, analysis)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ))
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
        with sqlite3.connect(self.db_path) as conn:
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
        for horse, position in results.items():
            won = position == 1
            placed = position <= 3
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
                    AVG(CASE WHEN won = 1 THEN odds ELSE 0 END) as avg_winning_odds
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY tag
                HAVING COUNT(*) >= ?
            """, (min_samples,)).fetchall()

            stats = {}
            for row in rows:
                tag, total, wins, places, avg_odds, avg_winning_odds = row
                wins = wins or 0
                places = places or 0

                # Calculate ROI (assuming $1 bets)
                # ROI = (total_returns - total_staked) / total_staked
                total_returns = wins * avg_winning_odds if avg_winning_odds else 0
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
                    AVG(odds) as avg_odds
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY confidence_range
            """).fetchall()

            stats = {}
            for row in rows:
                conf_range, total, wins, places, avg_odds = row
                wins = wins or 0
                places = places or 0

                stats[conf_range] = {
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": avg_odds,
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
                    AVG(odds) as avg_odds
                FROM predictions
                WHERE outcome_recorded = 1
                GROUP BY confidence_range
            """).fetchall()

            stats = {}
            for row in rows:
                conf_range, total, wins, places, avg_odds = row
                wins = wins or 0
                places = places or 0

                stats[conf_range] = {
                    "total": total,
                    "wins": wins,
                    "places": places,
                    "win_rate": wins / total if total > 0 else 0,
                    "place_rate": places / total if total > 0 else 0,
                    "avg_odds": avg_odds,
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
