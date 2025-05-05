import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from custom_search_ranking.app.services.predict_ranker import predict_with_model
from custom_search_ranking.app.services.constants import RANKING_MODEL_PATH, TRAINING_DATA_PATH



def evaluate_per_user(df, model_path):

    features = [
        'score_promotion', 
        'score_collaboratif', 
        'score_svd', 
        'score_season', 
        'score_local_trend', 
        'score_global_trend', 
        'score_navigation_client'
    ]

    # Prédiction sur tout
    X = df[features]
    df['predicted_score'] = predict_with_model(X, model_path)

    print(" prédictions obtenues, début évaluation par utilisateur...")

    ndcg_scores = []
    users_queries = []

    grouped = df.groupby(['user_guid', 'search_query'])

    for (user_guid, search_query), group in grouped:
        if len(group) < 2:
            continue  # besoin d'au moins 2 produits pour un classement

        y_true = group['final_score'].values
        y_pred = group['predicted_score'].values

        score = ndcg_score([y_true], [y_pred], k=10)
        ndcg_scores.append(score)
        users_queries.append((user_guid, search_query))

    if not ndcg_scores:
        print(" pas assez de données")
        return

    mean_ndcg = np.mean(ndcg_scores)

    print("\n résultats globaux :")
    print(f"nombre de requêtes évaluées : {len(ndcg_scores)}")
    print(f"NDCG@10 moyen : {mean_ndcg:.4f}")

    #  Graphe 
    plt.figure(figsize=(10, 6))
    plt.hist(ndcg_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('distribution des scores NDCG@10 par requête utilisateur')
    plt.xlabel('NDCG@10')
    plt.ylabel('Nombre de requêtes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("custom_search_ranking/app/data/ndcg_distribution.png")
    print(" graphe sauvegardé ")

    plt.show()

    return mean_ndcg

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA_PATH)
    evaluate_per_user(df, RANKING_MODEL_PATH)
