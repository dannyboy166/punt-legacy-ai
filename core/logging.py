"""
Logging configuration for Punt Legacy AI.

Provides structured logging that can be:
- Written to console during development
- Written to files in production
- Sent to monitoring services

Usage:
    from core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing race", track="Randwick", race=1)
    logger.warning("Odds mismatch", pf_odds=7.90, lb_odds=1.30)
    logger.error("API failed", error=str(e))
"""

import logging
import sys
from typing import Optional
from datetime import datetime


# Custom formatter that includes extra fields
class StructuredFormatter(logging.Formatter):
    """Formatter that includes extra fields in log output."""

    def format(self, record: logging.LogRecord) -> str:
        # Get base message
        base = super().format(record)

        # Add any extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            }:
                extras.append(f"{key}={value}")

        if extras:
            return f"{base} | {' '.join(extras)}"
        return base


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name (usually __name__)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Format: timestamp - level - name - message
        formatter = StructuredFormatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


class LogContext:
    """
    Context manager for adding context to log messages.

    Usage:
        with LogContext(logger, track="Randwick", race=1):
            logger.info("Processing")  # Will include track=Randwick race=1
    """

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._old_factory = None

    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()

        def factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self._old_factory)
        return False


# Pre-configured loggers for common modules
api_logger = get_logger("punt_legacy.api")
core_logger = get_logger("punt_legacy.core")
predictor_logger = get_logger("punt_legacy.predictor")


def log_api_call(
    logger: logging.Logger,
    endpoint: str,
    params: dict,
    success: bool,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log an API call with standard format."""
    extra = {
        "endpoint": endpoint,
        "params": str(params),
        "success": success,
    }
    if duration_ms:
        extra["duration_ms"] = round(duration_ms, 2)
    if error:
        extra["error"] = error

    if success:
        logger.debug(f"API call: {endpoint}", extra=extra)
    else:
        logger.warning(f"API call failed: {endpoint}", extra=extra)


def log_prediction_skip(
    logger: logging.Logger,
    track: str,
    race_number: int,
    reason: str,
    **details,
) -> None:
    """Log when a prediction is skipped."""
    logger.info(
        f"Skipping {track} R{race_number}: {reason}",
        extra={"track": track, "race": race_number, **details},
    )


def log_odds_mismatch(
    logger: logging.Logger,
    horse: str,
    pf_odds: float,
    lb_odds: float,
    track: str,
    race_number: int,
) -> None:
    """Log when PuntingForm and Ladbrokes odds don't match."""
    ratio = pf_odds / lb_odds if lb_odds > 0 else float("inf")
    logger.warning(
        f"Odds mismatch for {horse}: PF ${pf_odds:.2f} vs LB ${lb_odds:.2f} ({ratio:.1f}x)",
        extra={
            "horse": horse,
            "pf_odds": pf_odds,
            "lb_odds": lb_odds,
            "ratio": round(ratio, 2),
            "track": track,
            "race": race_number,
        },
    )
