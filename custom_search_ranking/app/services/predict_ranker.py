import pandas as pd
import joblib
import xgboost as xgb
from pipeline import personalized_ranking
import sqlite3
from user_product_matrix import load_views_from_db, build_user_product_matrix_from_df
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


DB_PATH = "custom_search_ranking/app/data/LFF.db"
MODEL_PATH = "custom_search_ranking/app/model/ranker_model.xgb"


def predict_ranking(user_guid: str, query: str, store_id: str, model_path: str = MODEL_PATH):
    # 1. Charger modèle
    model = xgb.Booster()
    model.load_model(model_path)

    # 2. Récupérer les features comme dans personalized_ranking
    df_products = personalized_ranking(user_guid, query, store_id)
    
    if df_products.empty:
        print("Aucun produit trouvé pour cette recherche.")
        return pd.DataFrame()

    # 3. Sélectionner les mêmes features que durant l'entraînement
    feature_cols = ["score_promotion", "score_collaboratif", "score_svd", 
                    'score_saison', 'score_local_trend', 'score_global_trend', 'score_navigation_client']

    X = df_products[feature_cols]

    # 4. Transformer en DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(X)

    # 5. Faire la prédiction
    preds = model.predict(dmatrix)

    # 6. Ajouter les prédictions au DataFrame
    df_products["predicted_score"] = preds

    # 7. Trier par score décroissant
    ranked_products = df_products.sort_values(by="predicted_score", ascending=False)

    return ranked_products


if __name__ == "__main__":
    # Test
    user_guid = "06f5c30584b25c685b3d04d4de85fba6fbe087f0534d64bb5af6876f155c92f2"
    query = "chaise"
    store_id = "0414"

    ranking = predict_ranking(user_guid, query, store_id)

    print("\n=== Résultats prédits ===")
    print(ranking[["product_id", "predicted_score"]].head(20))
 