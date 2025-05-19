
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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
from custom_search_ranking.app.services.admin import score_boost_admin





def personalize_ranking(user_guid: str, df_products: pd.DataFrame, store_id: str) -> pd.DataFrame:
    if df_products.empty:
        print("Aucun produit à personnaliser.")
        return pd.DataFrame()

    df_products.rename(columns={"code": "product_id"}, inplace=True)


    # Pre-fetch data
    df_all = load_views_from_db(DB_PATH)
    matrix = build_user_product_matrix_from_df(df_all)
    sim_matrix = compute_user_similarity_matrix(matrix)
    svd_matrix = compute_svd_scores(matrix)

    conn = sqlite3.connect(DB_PATH)
    try: 
        df_cart = pd.read_sql_query("SELECT product_id, timestamp FROM cart_purchases", conn)
        df_boosts = pd.read_sql_query("SELECT * FROM admin_boosts", conn)
        df_boosts['target_id'] = df_boosts['target_id'].astype(str)

        # Ensure start_date and end_date are timezone-naive
        df_boosts['start_date'] = pd.to_datetime(df_boosts['start_date'], errors='coerce').dt.tz_localize(None)
        df_boosts['end_date'] = pd.to_datetime(df_boosts['end_date'], errors='coerce').dt.tz_localize(None)
    finally:
        conn.close()

    # Ensure now is timezone-naive
    now_boost = datetime.now().replace(tzinfo=None)
    now_cart = datetime.now(timezone.utc)
    # Apply score_boost_admin
    df_products["boost_score"] = df_products["product_id"].apply(
        lambda pid: score_boost_admin(pid, now_boost, store_id, df_boosts)
    )
    df_products["boost_score"] = df_products["boost_score"].fillna(0)


    # Define scoring tasks
    def compute_promotion_scores():
        return df_products['promo_rate'].apply(compute_score_promotion)

    def compute_collaborative_scores():
        return df_products['product_id'].apply(
            lambda pid: score_collaboratif(pid, user_guid, matrix, sim_matrix)
        )

    def compute_svd_scores_local():
        return df_products['product_id'].apply(
            lambda pid: score_svd(user_guid, pid, svd_matrix)
        )

    def compute_season_scores():
        return df_products['product_id'].apply(
            lambda pid: total_season_score(pid, df_cart.copy(), now_cart)
        )
    

   # Execute tasks in parallel
    with ThreadPoolExecutor() as executor:
        future_promotion = executor.submit(compute_promotion_scores)
        future_collaborative = executor.submit(compute_collaborative_scores)
        future_svd = executor.submit(compute_svd_scores_local)
        future_season = executor.submit(compute_season_scores)

        df_products['score_promotion'] = future_promotion.result()
        df_products['score_collaboratif'] = future_collaborative.result()
        df_products['score_svd'] = future_svd.result()
        df_products['score_season'] = future_season.result()

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
    df_products['predicted_score'] = predict_with_model(df_products, RANKING_MODEL_PATH_JSON) + df_products["boost_score"]

    #8. Attribuer des badges
    def assign_badges(row):
        badges = []
        if row['score_global_trend'] >= 0.01:
            badges.append("🔥 Tendance")
        if row['score_svd'] >= 0.3:
            badges.append("💡 Recommandé pour vous")
        if row['score_navigation_client'] >= 0.3:
            badges.append("⏱️ Visité récemment")
        return badges
    
    df_products['badges'] = df_products.apply(assign_badges, axis=1)
    # 8. Tri final
    df_products = df_products.sort_values(by="predicted_score", ascending=False)
    print(df_products[['product_id', 'product_name', 'predicted_score', 'score_svd', 'score_promotion', 'promo_rate', 'score_collaboratif', 'score_local_trend', 'score_global_trend', 'score_season', 'score_navigation_client', 'boost_score']])

    return df_products