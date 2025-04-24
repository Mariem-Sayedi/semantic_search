from user_product_matrix import load_views_from_db
import pandas as pd

db_path = 'custom_search_ranking/app/data/LFF.db'


def compute_normalized_user_interaction_scores() -> pd.DataFrame:
    df_all = load_views_from_db(db_path)

    # 1. Agrégation des interactions client/produit
    df_user_product = df_all.groupby(['user_guid', 'product_id'])['interaction'].sum().reset_index()

    # 2. Trouver le max des interactions pour chaque client
    df_max = df_user_product.groupby('user_guid')['interaction'].max().reset_index()
    df_max.rename(columns={'interaction': 'max_interaction'}, inplace=True)

    # 3. Merge pour normaliser
    df_user_product = df_user_product.merge(df_max, on='user_guid')
    df_user_product['score_navigation_client'] = df_user_product['interaction'] / df_user_product['max_interaction']

    # 4. Optionnel : ne retourner que ce qui est utile
    return df_user_product[['user_guid', 'product_id', 'score_navigation_client']]


if __name__ == "__main__":
    df = compute_normalized_user_interaction_scores()
    print(df.head(20))
