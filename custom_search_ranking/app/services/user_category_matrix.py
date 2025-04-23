import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.float_format", "{:.3f}".format)


db_path = 'custom_search_ranking/app/data/LFF.db'

def load_viewed_categories_from_db(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = "SELECT user_guid, category FROM category_views"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def build_user_category_matrix_from_df(df: pd.DataFrame) -> pd.DataFrame:
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



def compute_svd_scores(matrix: pd.DataFrame, n_components: int = 20) -> pd.DataFrame:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_

    raw_scores = np.dot(user_factors, item_factors)
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(raw_scores)

    return pd.DataFrame(normalized_scores, index=matrix.index, columns=matrix.columns)

def score_svd(user_guid: str, category: str, svd_matrix: pd.DataFrame) -> float:
    if user_guid not in svd_matrix.index or category not in svd_matrix.columns:
        return 0.0
    return svd_matrix.at[user_guid, category]





df = load_viewed_categories_from_db("custom_search_ranking/app/data/LFF.db")
matrix = build_user_category_matrix_from_df(df)
similarity_matrix = compute_user_similarity_matrix(matrix)
print("Matrix \n")
print(matrix)
print("similarity_matrix")
print(similarity_matrix)

user_guid = "d005cb3992d9b3471f77b2eb2b854afa0d716cd0c793c5fe4d00f293a3855a8b"
category = "Consoles"

score = score_collaboratif(category, user_guid, matrix, similarity_matrix)
print(f"Score collaboratif de {user_guid} pour Abattants WC et accessoires : {score}")


svd_matrix = compute_svd_scores(matrix)

    
svd_score = score_svd(user_guid, category, svd_matrix)
print(f"\n Score SVD de {user_guid} pour {category} : {svd_score:.4f}")
