import pandas as pd
import sqlite3
from pipeline import personalized_ranking
from tqdm import tqdm
import time
import os
from custom_search_ranking.app.services.constants import DB_PATH, PROCESSED_USERS_FILE



def load_search_queries(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    search_query = "SELECT user_guid, search_query, store_id FROM searches"
    df_search = pd.read_sql_query(search_query, conn)
    conn.close()
    return df_search

def load_processed_users(file_path: str) -> set:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            users = {line.strip() for line in f.readlines()}
        return users
    else:
        return set()

def save_processed_user(user_guid: str, file_path: str):
    with open(file_path, "a") as f:
        f.write(user_guid + "\n")

def generate_training_data(df_search_queries: pd.DataFrame):
    full_data = []
    summary_data = []

    processed_users = load_processed_users(PROCESSED_USERS_FILE)

    for idx, row in tqdm(df_search_queries.iterrows(), total=len(df_search_queries)):
        user_guid = row['user_guid']
        search_query = row['search_query']
        store_id = row['store_id']

        if user_guid in processed_users:
            continue

        print(f"Traitement de {user_guid} - '{search_query}'...")

        try:
            df = personalized_ranking(user_guid, search_query, store_id)
        except Exception as e:
            print(f"Erreur pour {user_guid}, on continue : {e}")
            continue

        if df.empty:
            continue

        # partie full training data
        df_partial = df[[
            'product_id', 'score_promotion', 'score_collaboratif', 'score_svd',
            'score_season', 'score_local_trend', 'score_global_trend', 'score_navigation_client', 'final_score'
        ]].copy()
        df_partial['user_guid'] = user_guid
        df_partial['search_query'] = search_query
        df_partial['store_id'] = store_id

        full_data.append(df_partial)

        # partie résumé : 1 ligne par recherche
        product_ids = df_partial['product_id'].tolist()
        summary_data.append({
            "user_guid": user_guid,
            "search_query": search_query,
            "store_id": store_id,
            "product_ids": product_ids
        })

        #  traité
        save_processed_user(user_guid, PROCESSED_USERS_FILE)

        time.sleep(1)  # éviter de trop appeler l'API

    # concaténer tout
    if full_data:
        training_data = pd.concat(full_data, axis=0)
        training_data.to_csv("custom_search_ranking/app/data/training_data_full.csv", index=False)
        print(" training_data_full.csv sauvegardé !")
    else:
        print(" Aucun training_data collecté")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("custom_search_ranking/app/data/search_results_summary.csv", index=False)
        print(" search_results_summary.csv sauvegardé !")
    else:
        print(" Aucun résumé généré")

if __name__ == "__main__":
    df_search_queries = load_search_queries(DB_PATH)
    print(f"Nombre total de requêtes utilisateurs chargées : {len(df_search_queries)}")

    generate_training_data(df_search_queries)
