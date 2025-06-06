import requests
import sqlite3
from datetime import datetime, timezone
from user_interaction import compute_normalized_user_interaction_scores
import pandas as pd
from promo_scoring import compute_score_promotion
from user_product_matrix import (
    load_views_from_db, 
    build_user_product_matrix_from_df,
    compute_user_similarity_matrix, 
    score_collaboratif, 
    compute_svd_scores, 
    score_svd,
    # print_user_history
)
from season_scoring import total_season_score
from store_trends import compute_local_trend_score
from LFF_trends import compute_global_trend_score
from constants import DB_PATH
from search_products_api import fetch_and_display_products



def personalized_ranking(user_guid: str, query: str, store_id: str) -> pd.DataFrame:
    # 1. Récupérer les produits de l'API
    df_products = fetch_and_display_products(query, store_id)
    if df_products.empty:
        print("Aucun produit trouvé.")
        return pd.DataFrame()

    product_ids = df_products['product_id'].tolist()

    # 2. Ajout score de promotion
    df_products['score_promotion'] = df_products['promo_rate'].apply(compute_score_promotion)

    # 3. Chargement données user pour score collaboratif / SVD
    df_all = load_views_from_db(DB_PATH)
    matrix = build_user_product_matrix_from_df(df_all)
    # print_user_history(user_guid, matrix)
    sim_matrix = compute_user_similarity_matrix(matrix)
    svd_matrix = compute_svd_scores(matrix)

    df_products['score_collaboratif'] = df_products['product_id'].apply(
        lambda pid: score_collaboratif(pid, user_guid, matrix, sim_matrix)
    )
    df_products['score_svd'] = df_products['product_id'].apply(
        lambda pid: score_svd(user_guid, pid, svd_matrix)
    )

    # 4. Score de seasonnalité
    conn = sqlite3.connect(DB_PATH)
    df_cart = pd.read_sql_query("SELECT product_id, timestamp FROM cart_purchases", conn)
    conn.close()
    now = datetime.now(timezone.utc)
    df_products['score_season'] = df_products['product_id'].apply(
        lambda pid: total_season_score(pid, df_cart.copy(), now)
    )

    # 5. Score de tendance locale
    df_local_trend = compute_local_trend_score(store_id)
    df_local_trend.rename(columns={'score_trend': 'score_local_trend'}, inplace=True)
    df_products = df_products.merge(df_local_trend[['product_id', 'score_local_trend']], on='product_id', how='left')
    df_products['score_local_trend'] = df_products['score_local_trend'].fillna(0)


    # 6. Score de tendance globale
    df_global_trend = compute_global_trend_score()
    df_global_trend.rename(columns={'score_trend': 'score_global_trend'}, inplace=True)
    df_products = df_products.merge(df_global_trend[['product_id', 'score_global_trend']], on='product_id', how='left')
    df_products['score_global_trend'] = df_products['score_global_trend'].fillna(0)



    # 7. Score navigation client
    df_user_inter = compute_normalized_user_interaction_scores()
    df_user_inter = df_user_inter[df_user_inter['user_guid'] == user_guid]

    df_products = df_products.merge(df_user_inter[['product_id', 'score_navigation_client']], 
                                    on='product_id', how='left')
    df_products['score_navigation_client'] = df_products['score_navigation_client'].fillna(0)

     # 8. Calcul score final pondéré
    df_products['final_score'] = (
        0.9 * df_products['score_promotion'] +
        0.4 * df_products['score_collaboratif'] +
        0.3 * df_products['score_svd'] +
        0.2 * df_products['score_season'] +
        0.2 * df_products['score_local_trend'] + 
        0.4 * df_products['score_global_trend'] + 
        0.8 * df_products['score_navigation_client']
    )

    return df_products.sort_values(by="final_score", ascending=False)


if __name__ == "__main__":
    ranking = personalized_ranking(
        user_guid="b52b4e3cc61d597266d3156a1948406265df0713af92f7cd46a88bc69c0ae143",
        query="table",
        store_id="0008"
    )
    ranking_list = ranking['product_id'].tolist()
    print("*************calcul du score de ranking avec la somme pondérée******************")
    print(ranking[['product_id', 'final_score', 'score_svd', 'score_promotion', 'score_collaboratif', 'score_local_trend', 'score_global_trend', 'score_season', 'score_navigation_client']])


#3a8ea2b9a7be61cb3633dbea6059fc1bb90f0328d006b9201455198fc8eaae40    