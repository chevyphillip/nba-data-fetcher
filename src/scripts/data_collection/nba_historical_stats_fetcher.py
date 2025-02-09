"""
NBA Historical Stats Fetcher

This script fetches historical NBA statistics from basketball-reference.com
and saves them in a structured format.

The script collects various types of player statistics:
- Totals: Season totals for various statistics
- Per Game: Average statistics per game
- Advanced: Advanced analytics and metrics
- Per Minute: Statistics normalized by minutes played
- Per Possession: Statistics normalized by team possessions
- Shooting: Detailed shooting statistics and percentages

Features:
- Fetches data for multiple seasons (2010-2024 by default)
- Handles rate limiting and random user agents
- Processes and cleans data automatically
- Saves data in a structured CSV format with metadata

Usage:
    python nba_historical_stats_fetcher.py

The data will be saved in the src/data/raw directory with the filename format:
nba_player_stats_START-END_YYYYMMDD.csv

Dependencies:
    - beautifulsoup4: HTML parsing
    - pandas: Data processing
    - requests: HTTP requests
    - fake-useragent: Random user agent generation
    - lxml: HTML parsing backend for pandas

Author: [Your Name]
Date: February 2024
"""

import logging
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.basketball-reference.com/leagues/NBA_"
SEASONS = list(range(2010, 2025))
TABS = ["totals", "per_game", "advanced", "per_minute", "per_poss", "shooting"]

# Get the script's directory and set up data paths
SCRIPT_DIR = Path(__file__).parent.parent.parent  # src directory
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
)

# Create session with retry strategy
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def get_random_user_agent() -> str:
    """Generate a random user agent string."""
    try:
        ua = UserAgent()
        return ua.random
    except Exception as e:
        logger.warning(f"Failed to generate random user agent: {e}")
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def fetch_table(
    url: str, headers: dict, max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Fetch and parse a table from the given URL with retries.

    Args:
        url: The URL to fetch the table from
        headers: Request headers
        max_retries: Maximum number of retry attempts

    Returns:
        Optional[pd.DataFrame]: Parsed table or None if failed
    """
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Check if we've been rate limited
            if "Rate limit exceeded" in response.text:
                wait_time = min(60 * (attempt + 1), 300)  # Max 5 minute wait
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")

            if table is None:
                logger.warning(f"No table found at {url}")
                return None

            # Use StringIO to avoid FutureWarning
            html_io = StringIO(str(table))
            df = pd.read_html(html_io)[0]

            # Basic validation of the data
            if df.empty:
                logger.warning(f"Empty table found at {url}")
                return None

            return df

        except requests.RequestException as e:
            logger.error(
                f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            continue
        except Exception as e:
            logger.error(f"Error processing table from {url}: {e}")
            return None

    return None


def process_dataframe(df: pd.DataFrame, season: str, tab: str) -> pd.DataFrame:
    """
    Process the raw DataFrame by adding metadata columns and cleaning data.

    Args:
        df: Raw DataFrame
        season: Season year
        tab: Statistics type

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()

    # Handle MultiIndex columns (like in shooting stats)
    if isinstance(df.columns, pd.MultiIndex):
        # Join MultiIndex levels with underscore
        df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
    else:
        # Clean any special characters in column names
        df.columns = df.columns.str.strip().str.replace("\n", "_")

    # Add metadata columns
    df["Season"] = season
    df["Tab"] = tab
    df["FetchDate"] = datetime.now().strftime("%Y-%m-%d")

    return df


def fetch_nba_stats(
    seasons: List[int] = SEASONS, tabs: List[str] = TABS
) -> Optional[pd.DataFrame]:
    """
    Fetch NBA statistics for specified seasons and statistical categories.

    Args:
        seasons: List of seasons to fetch
        tabs: List of statistical categories

    Returns:
        Optional[pd.DataFrame]: Combined statistics or None if no data
    """
    dataframes = []

    for season in seasons:
        for tab in tabs:
            url = f"{BASE_URL}{season}_{tab}.html"
            headers = {"User-Agent": get_random_user_agent()}

            logger.info(f"Fetching data for {season} - {tab}...")

            df = fetch_table(url, headers)
            if df is not None:
                df = process_dataframe(df, str(season), tab)
                dataframes.append(df)
                logger.info(f"Successfully processed data for {season} - {tab}")

            # Rate limiting
            time.sleep(3)

    if not dataframes:
        logger.error("No data was collected")
        return None

    return pd.concat(dataframes, ignore_index=True)


def save_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save the DataFrame to a CSV file in the raw data directory.

    Args:
        df: DataFrame to save
        filename: Name of the file
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DIR / filename

    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")


def main():
    """Main execution function."""
    try:
        logger.info("Starting NBA stats collection...")

        master_df = fetch_nba_stats()
        if master_df is not None:
            filename = f"nba_player_stats_{min(SEASONS)}-{max(SEASONS)}_{datetime.now().strftime('%Y%m%d')}.csv"
            save_data(master_df, filename)
            logger.info("Data collection completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    main()
