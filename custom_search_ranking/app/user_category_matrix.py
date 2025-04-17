import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity


db_path = 'custom_search_ranking/app/data/LFF.db'

def load_viewed_products_from_db(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = "SELECT user_guid, category FROM category_views"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def build_user_product_matrix_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="user_guid", columns="category", aggfunc=lambda x: 1, fill_value=0)


def build_user_product_matrix_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="user_guid", columns="category", aggfunc=lambda x: 1, fill_value=0)


def compute_user_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(matrix.values)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

def score_collaboratif(category: str, user_guid: str, matrix: pd.DataFrame, similarity_df: pd.DataFrame) -> float:
    if user_guid not in similarity_df.index or category not in matrix.columns:
        return 0.0

    similar_users = similarity_df[user_guid].drop(index=user_guid)
    score = 0.0
    total_sim = 0.0

    for other_user, sim in similar_users.items():
        if matrix.at[other_user, category] == 1:
            score += sim
            total_sim += sim

    return score / total_sim if total_sim > 0 else 0.0



df = load_viewed_products_from_db("custom_search_ranking/app/data/LFF.db")
matrix = build_user_product_matrix_from_df(df)
similarity_matrix = compute_user_similarity_matrix(matrix)
print("Matrix \n")
print(matrix)
print("similarity_matrix")
print(similarity_matrix)


score = score_collaboratif("Abattants WC et accessoires", "d005cb3992d9b3471f77b2eb2b854afa0d716cd0c793c5fe4d00f293a3855a8b", matrix, similarity_matrix)
print(f"Score collaboratif de d005cb3992d9b3471f77b2eb2b854afa0d716cd0c793c5fe4d00f293a3855a8b pour Abattants WC et accessoires : {score}")
