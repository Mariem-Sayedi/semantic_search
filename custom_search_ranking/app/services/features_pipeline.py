# features_pipeline.py
import pandas as pd
from datetime import datetime, timezone
import sqlite3
from user_product_matrix import (
    load_views_from_db, build_user_product_matrix_from_df,
    compute_user_similarity_matrix, compute_svd_scores,
    score_collaboratif, score_svd
)
from season import total_saison_score
from store_trends import compute_local_trend_score
from LFF_trends import compute_global_trend_score
from promo_scoring import extract_promotions_from_api, compute_score_promotion


pd.set_option("display.float_format", "{:.3f}".format)



DB_PATH = "custom_search_ranking/app/data/LFF.db"

df_all = load_views_from_db(DB_PATH)
matrix = build_user_product_matrix_from_df(df_all)
similarity_matrix = compute_user_similarity_matrix(matrix)
svd_matrix = compute_svd_scores(matrix)

conn = sqlite3.connect(DB_PATH)
df_cart = pd.read_sql_query("SELECT product_id, timestamp FROM cart_purchases", conn)
conn.close()


def get_promo_score_dict(query: str) -> dict:
    df_promo = extract_promotions_from_api(query)
    return dict(zip(df_promo['product_id'], df_promo['score_promotion']))


def compute_all_scores(product_id: str, user_guid: str, store_id: str, timestamp: datetime,
                       df_local: pd.DataFrame, df_global: pd.DataFrame, promo_scores: dict) -> dict:
    score_local = df_local[df_local['product_id'] == product_id]['score_trend']
    score_local = float(score_local.values[0]) if not score_local.empty else 0.0

    score_global = df_global[df_global['product_id'] == product_id]['score_trend']
    score_global = float(score_global.values[0]) if not score_global.empty else 0.0

    score_trend = 0.7 * score_local + 0.3 * score_global

    return {
        "product_id": product_id,
        "score_svd": round(score_svd(user_guid, product_id, svd_matrix), 3),
        "score_collaboratif": round(score_collaboratif(product_id, user_guid, matrix, similarity_matrix), 3),
        "score_trend": round(score_trend, 3),
        "score_promotion": round(promo_scores.get(product_id, 0.0), 3),
        "score_saison": round(total_saison_score(product_id, df_cart, timestamp), 3)
    }


def build_items_dataframe(product_ids: list, user_guid: str, store_id: str,
                          timestamp: datetime, query: str) -> pd.DataFrame:
    df_local = compute_local_trend_score(store_id)
    df_global = compute_global_trend_score()
    promo_scores = get_promo_score_dict(query)

    rows = [
        compute_all_scores(pid, user_guid, store_id, timestamp, df_local, df_global, promo_scores)
        for pid in product_ids
    ]

    df = pd.DataFrame(rows)

    df["score_total"] = (
        0.3 * df["score_svd"] +
        0.2 * df["score_collaboratif"] +
        0.2 * df["score_trend"] +
        0.2 * df["score_promotion"] +
        0.1 * df["score_saison"]
    )
    return df.sort_values(by="score_total", ascending=False)


if __name__ == "__main__":
    user_guid = "d005cb3992d9b3471f77b2eb2b854afa0d716cd0c793c5fe4d00f293a3855a8b"
    store_id = "0414"
    product_ids = ["10000106050", "10000504317", "10000556466"]
    timestamp = datetime.now(timezone.utc)
    query = "chaise"

    df_scores = build_items_dataframe(product_ids, user_guid, store_id, timestamp, query)
    print("\n résultat du pipeline personnalisé :\n")
    print(df_scores)
