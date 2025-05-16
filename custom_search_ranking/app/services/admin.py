import pandas as pd
from datetime import datetime
import sqlite3

# from constants import DB_PATH

def score_boost_admin(product_id: int, date: datetime, store_id: str, df_boosts: pd.DataFrame) -> float:
    # Ensure date is timezone-naive for comparison
    if hasattr(date, 'tzinfo') and date.tzinfo is not None:
        date = date.astimezone(tz=None).replace(tzinfo=None)

    df_boosts['start_date'] = pd.to_datetime(df_boosts['start_date'], errors='coerce')
    df_boosts['end_date'] = pd.to_datetime(df_boosts['end_date'], errors='coerce')

    # Remove timezone if it exists
    if isinstance(df_boosts['start_date'].dtype, pd.DatetimeTZDtype):    
        df_boosts['start_date'] = df_boosts['start_date'].dt.tz_convert(None)

    if isinstance(df_boosts['end_date'].dtype, pd.DatetimeTZDtype):
        df_boosts['end_date'] = df_boosts['end_date'].dt.tz_convert(None)

    filtered_boosts = df_boosts[
        (df_boosts['target_type'] == 'Produit') &
        (df_boosts['target_id'] == product_id) &
        (df_boosts['store_id'].astype(str) == store_id) &
        (df_boosts['start_date'] <= date) &
        (df_boosts['end_date'] >= date)
    ]

    if not filtered_boosts.empty:
        return filtered_boosts['boost_score'].max()
    return 0.0


