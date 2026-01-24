"""Parser for betting app exports: matches bets to games and assigns game_ids."""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import pandas as pd


# NBA team code to full name mapping
TEAM_CODE_MAP = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}

# NCAAF team code to full name mapping (sample)
NCAAF_TEAM_CODE_MAP = {
    'PSU': 'Penn State',
    'CLEM': 'Clemson',
    'UCONN': 'UConn',
    'ARMY': 'Army',
    'GT': 'Georgia Tech',
    'BYU': 'Brigham Young',
    'GASO': 'Georgia Southern',
    'APP': 'Appalachian State',
}

# NHL team code to database abbreviation mapping (31 teams)
NHL_TEAM_CODE_MAP = {
    'ANA': 'ANA',
    'BOS': 'BOS',
    'BUF': 'BUF',
    'CAR': 'CAR',
    'CBJ': 'CBJ',
    'CGY': 'CGY',
    'CHI': 'CHI',
    'COL': 'COL',
    'DAL': 'DAL',
    'DET': 'DET',
    'EDM': 'EDM',
    'FLA': 'FLA',
    'LA': 'LAK',  # LA Kings stored as LAK in DB
    'MIN': 'MIN',
    'MTL': 'MTL',
    'NJ': 'NJD',  # New Jersey Devils stored as NJD in DB
    'NYI': 'NYI',
    'NYR': 'NYR',
    'OTT': 'OTT',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SJ': 'SJS',  # San Jose Sharks stored as SJS in DB
    'STL': 'STL',
    'TB': 'TBL',  # Tampa Bay Lightning stored as TBL in DB
    'TOR': 'TOR',
    'UTA': 'UTA',
    'VAN': 'VAN',
    'VGK': 'VGK',
    'WPG': 'WPG',
    'WSH': 'WSH',
}


def parse_betting_app_export(
    csv_path: str,
    db_path: str,
    output_path: str | None = None,
    sport: str = 'nba',
    season: str = '2025-26',
) -> tuple[int, int, list[str]]:
    """Parse betting app export CSV and assign game_ids from database.
    
    Matches bets by:
    1. Parsing team codes from game string (e.g., "DAL @ SAC")
    2. Extracting game date from start_time
    3. Looking up game in database by sport, season, home/away teams, and date
    4. Adding game_id column to the CSV
    5. Saving enriched CSV to output_path (or with "-with-ids" suffix if not specified)
    
    Args:
        csv_path: Path to betting app export CSV
        db_path: Path to the database
        output_path: Optional output path; defaults to csv_path with "-with-ids" suffix
        sport: Sport code (nba, ncaaf, nhl, etc.)
        season: Season code (e.g., 2025-26)
    
    Returns:
        Tuple of (matched_count, total_count, unmatched_games)
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_file}")
    
    # Read CSV, skip first row only if it contains the data: URI prefix
    df = pd.read_csv(csv_path)
    
    # Check if first data row looks like a URI (data: prefix)
    if len(df) > 0 and isinstance(df.iloc[0, 0], str) and str(df.iloc[0, 0]).startswith('data:'):
        # This came from browser export - skip the URI row
        df = df.iloc[1:].reset_index(drop=True)
    
    # Normalize column names (case-insensitive, handle special characters)
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(r'/', '_', regex=False).str.replace(r'\s+', '_', regex=True)
    
    # Select team code map based on sport
    if sport.lower() == 'ncaaf':
        team_code_map = NCAAF_TEAM_CODE_MAP
    elif sport.lower() == 'nhl':
        team_code_map = NHL_TEAM_CODE_MAP
    else:
        team_code_map = TEAM_CODE_MAP
    
    # Get games from database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        'SELECT game_id, home_team, away_team, date FROM games WHERE sport = ? AND season = ?',
        (sport, season)
    )
    games = cur.fetchall()
    conn.close()
    
    # Build lookup dict: {(date, home_team_lower, away_team_lower): game_id}
    game_lookup = {}
    for game_id, home_team, away_team, date in games:
        key = (date, home_team.lower(), away_team.lower())
        game_lookup[key] = game_id
    
    # Function to resolve game_id from game string and timestamp
    def find_game_id(game_str: str, start_time: str) -> str | None:
        """Match game from team codes and date, with timezone tolerance."""
        try:
            # Parse game string (e.g., "DAL @ SAC" → away="DAL", home="SAC")
            parts = game_str.split('@')
            if len(parts) != 2:
                return None
            
            away_code = parts[0].strip()
            home_code = parts[1].strip()
            
            # Convert codes to full team names
            away_team = team_code_map.get(away_code)
            home_team = team_code_map.get(home_code)
            
            if not away_team or not home_team:
                return None
            
            # Extract date from ISO timestamp
            dt = pd.to_datetime(start_time)
            date = dt.date().isoformat()
            
            # Try exact match first
            key = (date, home_team.lower(), away_team.lower())
            if key in game_lookup:
                return game_lookup[key]
            
            # Try nearby dates (±1 day) to handle timezone differences
            # (e.g., bet at 00:00 UTC on Dec 30 might be for a game on Dec 29 local time)
            for days_offset in [-1, 1]:
                nearby_date = (dt + timedelta(days=days_offset)).date().isoformat()
                nearby_key = (nearby_date, home_team.lower(), away_team.lower())
                if nearby_key in game_lookup:
                    return game_lookup[nearby_key]
            
            return None
        except Exception:
            return None
    
    # Add game_id column
    df['game_id'] = df.apply(
        lambda row: find_game_id(row.get('game', ''), row.get('start_time', '')),
        axis=1
    )
    
    # Count matches
    matched = df['game_id'].notna().sum()
    total = len(df)
    
    # Get unmatched games with normalized column names
    unmatched_mask = df['game_id'].isna()
    unmatched_games = df[unmatched_mask][['game', 'start_time']].drop_duplicates()
    unmatched_list = [f"{row['game']} @ {row['start_time']}" for _, row in unmatched_games.iterrows()]
    
    # Determine output path
    if output_path is None:
        output_file = csv_file.parent / f"{csv_file.stem}_with_ids.csv"
    else:
        output_file = Path(output_path)
    
    # Save enriched CSV
    df.to_csv(output_file, index=False)
    
    return matched, total, unmatched_list


def parse_betting_app_exports_by_sport(
    csv_path: str,
    db_path: str,
    output_dir: str | None = None,
) -> dict:
    """Parse betting app export with mixed sports and split into separate files with game_ids.
    
    Automatically:
    1. Detects sports in CSV
    2. Splits by sport
    3. Matches each sport's bets to the correct database
    4. Writes _with_ids CSV for each sport
    
    Returns:
        Dict with keys: {sport: {matched: int, total: int, unmatched: [str]}}
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    
    # Read CSV
    df = pd.read_csv(csv_path, skiprows=1)
    
    # Detect sports
    sports = df['League'].str.lower().unique()
    sports = [s for s in sports if s and pd.notna(s)]
    
    results = {}
    
    for sport in sports:
        # Split by sport
        sport_df = df[df['League'].str.lower() == sport]
        
        # Write temporary sport-specific CSV (without extra header since we're just splitting)
        temp_csv = csv_file.parent / f"temp_{sport}.csv"
        sport_df.to_csv(temp_csv, index=False)
        
        # Parse with game_ids
        db_path_sport = Path(db_path.replace('<sport>', sport))
        
        # Skip sports that don't have a database
        if not db_path_sport.exists():
            print(f"[SKIP] {sport.upper()}: Database not found at {db_path_sport}")
            results[sport] = {'matched': 0, 'total': len(sport_df), 'unmatched': list(sport_df['Game'].unique())}
            temp_csv.unlink(missing_ok=True)
            continue
        
        try:
            matched, total, unmatched = parse_betting_app_export(
                csv_path=str(temp_csv),
                db_path=str(db_path_sport),
                output_path=csv_file.parent / f"{csv_file.stem}_{sport}_with_ids.csv",
                sport=sport,
                season="2025-26",  # Add default season
            )
            results[sport] = {'matched': matched, 'total': total, 'unmatched': unmatched}
        finally:
            # Clean up temp file
            temp_csv.unlink(missing_ok=True)
    
    return results
