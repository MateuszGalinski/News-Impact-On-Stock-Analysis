from datetime import datetime
from dataclasses import dataclass

@dataclass
class NewsItem:
    """Represents a single news article."""
    created_at: str      # ISO format timestamp with timezone
    symbols: str          # Comma-separated stock symbols, e.g. "AMZN,MSFT"
    headline: str
    summary: str
    full_text: str
    source: str
    url: str

    def to_timestamp(self) -> int:
        """Convert created_at string to Unix timestamp (seconds)."""
        # Handle ISO format with space (e.g., "2026-02-26 08:49:31+00:00")
        dt = datetime.strptime(self.created_at, "%Y-%m-%d %H:%M:%S%z")
        return int(dt.timestamp())

    def combined_text(self) -> str:
        """Combine headline, summary and full text for embedding and BM25."""
        parts = [self.headline, self.summary, self.full_text]
        # Remove empty parts and join with space
        return " ".join(p for p in parts if p).strip()