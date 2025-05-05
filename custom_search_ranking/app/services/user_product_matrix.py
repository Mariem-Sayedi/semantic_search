import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from custom_search_ranking.app.services.constants import DB_PATH



pd.set_option("display.float_format", "{:.3f}".format)



def load_views_from_db(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    product_query = "SELECT user_guid, product_id FROM product_views"
    cart_purchase_query = "SELECT user_guid, product_id, event_type FROM cart_purchases"

    df_product = pd.read_sql_query(product_query, conn)
    df_cart_purchase = pd.read_sql_query(cart_purchase_query, conn)
    conn.close()

    df_product['interaction'] = 3

    df_cart_purchase['interaction'] = df_cart_purchase['event_type'].map({
        'add_to_cart': 5,
        'purchase': 10
    })
    df_cart_purchase.drop(columns=['event_type'], inplace=True)

    df_all = pd.concat([df_product, df_cart_purchase])
    df_all = df_all.groupby(['user_guid', 'product_id']).sum().reset_index()

    return df_all


def build_user_product_matrix_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index='user_guid', columns='product_id', values='interaction').fillna(0)


def compute_user_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(matrix.values)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)


def score_collaboratif(product_id: str, user_guid: str, matrix: pd.DataFrame, similarity_df: pd.DataFrame) -> float:
    if user_guid not in similarity_df.index or product_id not in matrix.columns:
        return 0.0

    similar_users = similarity_df[user_guid].drop(index=user_guid)
    score = 0.0
    total_sim = 0.0

    for other_user, sim in similar_users.items():
        interaction = matrix.at[other_user, product_id] if product_id in matrix.columns else 0
        if interaction > 0:
            score += sim * interaction
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


def score_svd(user_guid: str, product_id: str, svd_matrix: pd.DataFrame) -> float:
    if user_guid not in svd_matrix.index or product_id not in svd_matrix.columns:
        return 0.0
    return svd_matrix.at[user_guid, product_id]


def plot_score_comparison(scores_cos_scaled, scores_svd, save_path="score_comparison.png"):
    sorted_cos = np.sort(scores_cos_scaled)
    sorted_svd = np.sort(scores_svd)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_cos, label="Collaboratif (normalisé)")
    plt.plot(sorted_svd, label="SVD")
    plt.title("Scores normalisés - Comparaison utilisateur")
    plt.xlabel("Index produit trié")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"graphe sauvegardé ici : {save_path}")


# def print_user_history(user_guid: str, matrix: pd.DataFrame):
#     if user_guid not in matrix.index:
#         print(f"Aucune interaction trouvée pour l'utilisateur {user_guid}")
#     else:
#         # print(f"\nProduits vus/achetés par {user_guid} :")
#         print(matrix.loc[user_guid][matrix.loc[user_guid] > 0])
        



if __name__ == "__main__":
    df_all = load_views_from_db(DB_PATH)
    matrix = build_user_product_matrix_from_df(df_all)
    similarity_matrix = compute_user_similarity_matrix(matrix)

    print("Matrice d'interaction :\n", matrix)
    print("\n Matrice de similarité :\n", similarity_matrix)

    user_guid = "9f2bace64e9acd27ae936110a3738c137a6cfa5073b637f1819554e8ab8a5aa1"
    product_id = "10000269636"

    score = score_collaboratif(product_id, user_guid, matrix, similarity_matrix)
    print(f"\nScore collaboratif de {user_guid} pour {product_id} : {score:.4f}")

    svd_matrix = compute_svd_scores(matrix)
    svd_score = score_svd(user_guid, product_id, svd_matrix)
    print(f"\nScore SVD de {user_guid} pour {product_id} : {svd_score:.4f}")

    scores_cos = []
    scores_svd = []
    products = matrix.columns

    for product in products:
        scores_cos.append(score_collaboratif(product, user_guid, matrix, similarity_matrix))
        scores_svd.append(score_svd(user_guid, product, svd_matrix))

    scores_cos_scaled = MinMaxScaler().fit_transform(np.array(scores_cos).reshape(-1, 1)).flatten()

    try:
        idx = list(products).index(product_id)
        print(f"\n score collaboratif après normalisation de {user_guid} pour {product_id} : {scores_cos_scaled[idx]:.4f}")
    except ValueError:
        print("\n produit non trouvé dans les colonnes pour affichage du score normalisé.")

    plot_score_comparison(scores_cos_scaled, scores_svd)

