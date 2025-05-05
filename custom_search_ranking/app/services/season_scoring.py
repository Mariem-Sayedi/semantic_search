import pandas as pd
from datetime import datetime, timedelta, timezone
import sqlite3
from custom_search_ranking.app.services.constants import DB_PATH



pd.set_option("display.float_format", "{:.3f}".format)

def calculate_week_store(product_id: str, df: pd.DataFrame, current_date: datetime) -> float:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_week = current_date - timedelta(days=7)
    df_week = df[df['timestamp'] >= start_week]

    total = df_week['product_id'].value_counts().sum()
    count = df_week['product_id'].value_counts().get(product_id, 0)
    return count / total if total > 0 else 0.0


def calculate_season_score(product_id: str, df: pd.DataFrame, current_date: datetime) -> float:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_season = current_date - timedelta(days=90)
    df_season = df[df['timestamp'] >= start_season]

    total = df_season['product_id'].value_counts().sum()
    count = df_season['product_id'].value_counts().get(product_id, 0)
    return count / total if total > 0 else 0.0


def total_season_score(product_id: str, df: pd.DataFrame, current_date: datetime) -> float:
    week_score = calculate_week_store(product_id, df, current_date)
    three_months_score = calculate_season_score(product_id, df, current_date)
    return 0.7 * week_score + 0.3 * three_months_score


conn = sqlite3.connect("custom_search_ranking/app/data/LFF.db")
current_date = datetime.now(timezone.utc)

df_cart = pd.read_sql_query("SELECT product_id, timestamp FROM cart_purchases", conn)
conn.close()

score = total_season_score("10000259220", df_cart, current_date)
