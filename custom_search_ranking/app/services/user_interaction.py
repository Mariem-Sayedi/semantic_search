from user_product_matrix import load_views_from_db
import pandas as pd
from constants import DB_PATH




def compute_normalized_user_interaction_scores() -> pd.DataFrame:
    df_all = load_views_from_db(DB_PATH)

    # 1 agrégation des interactions client/produit
    df_user_product = df_all.groupby(['user_guid', 'product_id'])['interaction'].sum().reset_index()

    # 2 trouver le max des interactions pour chaque client
    df_max = df_user_product.groupby('user_guid')['interaction'].max().reset_index()
    df_max.rename(columns={'interaction': 'max_interaction'}, inplace=True)

    # 3 merge pour normaliser
    df_user_product = df_user_product.merge(df_max, on='user_guid')
    df_user_product['score_navigation_client'] = df_user_product['interaction'] / df_user_product['max_interaction']

    return df_user_product[['user_guid', 'product_id', 'score_navigation_client']]


if __name__ == "__main__":
    df = compute_normalized_user_interaction_scores()
    print(df.head(20))
