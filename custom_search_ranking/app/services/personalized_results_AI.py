
import sqlite3
from datetime import datetime, timezone
import pandas as pd

from custom_search_ranking.app.services.user_interaction import compute_normalized_user_interaction_scores
from custom_search_ranking.app.services.promo_scoring import compute_score_promotion
from custom_search_ranking.app.services.user_product_matrix import (
    load_views_from_db,
    build_user_product_matrix_from_df,
    compute_user_similarity_matrix,
    score_collaboratif,
    compute_svd_scores,
    score_svd,
)
from custom_search_ranking.app.services.season_scoring import total_season_score
from custom_search_ranking.app.services.store_trends import compute_local_trend_score
from custom_search_ranking.app.services.LFF_trends import compute_global_trend_score
from custom_search_ranking.app.services.predict_ranker import predict_with_model
from custom_search_ranking.app.services.constants import RANKING_MODEL_PATH_JSON, DB_PATH

def personalize_ranking(user_guid: str, df_products: pd.DataFrame, store_id: str) -> pd.DataFrame:
    if df_products.empty:
        print("Aucun produit à personnaliser.")
        return pd.DataFrame()

    df_products.rename(columns={"code": "product_id"}, inplace=True)

    # 1. Score de promotion
    df_products['score_promotion'] = df_products['promo_rate'].apply(compute_score_promotion)

    # 2. Collaborative filtering / SVD
    df_all = load_views_from_db(DB_PATH)
    matrix = build_user_product_matrix_from_df(df_all)
    sim_matrix = compute_user_similarity_matrix(matrix)
    svd_matrix = compute_svd_scores(matrix)

    df_products['score_collaboratif'] = df_products['product_id'].apply(
        lambda pid: score_collaboratif(pid, user_guid, matrix, sim_matrix)
    )
    df_products['score_svd'] = df_products['product_id'].apply(
        lambda pid: score_svd(user_guid, pid, svd_matrix)
    )

    # 3. Score saisonnalité
    conn = sqlite3.connect(DB_PATH)
    df_cart = pd.read_sql_query("SELECT product_id, timestamp FROM cart_purchases", conn)
    conn.close()
    now = datetime.now(timezone.utc)
    df_products['score_season'] = df_products['product_id'].apply(
        lambda pid: total_season_score(pid, df_cart.copy(), now)
    )

    # 4. Tendance locale
    df_local_trend = compute_local_trend_score(store_id)
    df_local_trend.rename(columns={'score_trend': 'score_local_trend'}, inplace=True)
    df_products = df_products.merge(df_local_trend[['product_id', 'score_local_trend']], on='product_id', how='left')
    df_products['score_local_trend'] = df_products['score_local_trend'].fillna(0)

    # 5. Tendance globale
    df_global_trend = compute_global_trend_score()
    df_global_trend.rename(columns={'score_trend': 'score_global_trend'}, inplace=True)
    df_products = df_products.merge(df_global_trend[['product_id', 'score_global_trend']], on='product_id', how='left')
    df_products['score_global_trend'] = df_products['score_global_trend'].fillna(0)

    # 6. Navigation client
    df_user_inter = compute_normalized_user_interaction_scores()
    df_user_inter = df_user_inter[df_user_inter['user_guid'] == user_guid]
    df_products = df_products.merge(
        df_user_inter[['product_id', 'score_navigation_client']],
        on='product_id', how='left'
    )
    df_products['score_navigation_client'] = df_products['score_navigation_client'].fillna(0)

    # 7. Score final prédictif
    df_products['predicted_score'] = predict_with_model(df_products, RANKING_MODEL_PATH_JSON)

    # 8. Tri final
    df_products = df_products.sort_values(by="predicted_score", ascending=False)
    print(df_products)
    return df_products
