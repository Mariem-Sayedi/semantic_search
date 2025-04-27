import pandas as pd
import sqlite3
from pipeline import personalized_ranking
from tqdm import tqdm  # barre de progression
import time

DB_PATH = "custom_search_ranking/app/data/LFF.db"


def load_search_queries(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    search_query = "SELECT user_guid, search_query, store_id FROM searches"
    df_search = pd.read_sql_query(search_query, conn)
    conn.close()
    df_search_counts = df_search.groupby(['user_guid', 'search_query', 'store_id']).size().reset_index(name='search_count')
    return df_search_counts


def generate_training_data(df_search_queries: pd.DataFrame):
    X = []
    y = []
    group = []

    for idx, row in tqdm(df_search_queries.iterrows(), total=len(df_search_queries)):
        user_guid = row['user_guid']
        search_query = row['search_query']
        store_id = row['store_id']
        
        print(f"Traitement de {user_guid}, - {search_query}")

        try:
            df = personalized_ranking(user_guid, search_query, store_id)

            if df.empty:
                continue

            features = df[[
                "score_promotion",
                "score_collaboratif",
                "score_svd",
                "score_saison",
                "score_local_trend",
                "score_global_trend",
                "score_navigation_client"
            ]]
            labels = df['final_score']

            X.append(features)
            y.append(labels)
            group.append(len(df))

            time.sleep(1)  # petite pause pour ne pas surcharger

        except Exception as e:
            print(f"Erreur pour {user_guid}, {search_query}: {e}")
            time.sleep(2)  # On attend un peu plus si erreur
            continue

    if X:
        X = pd.concat(X, axis=0)
    else:
        X = pd.DataFrame()

    if y:
        y = pd.concat(y, axis=0)
    else:
        y = pd.Series()

    return X, y, group


def chunk_data(df: pd.DataFrame, chunk_size: int = 300):
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]


if __name__ == "__main__":
    df_search_queries = load_search_queries(DB_PATH)

    print(f"Nombre total de requêtes utilisateurs chargées : {len(df_search_queries)}")

    # Variables pour tout accumuler
    X_all = []
    y_all = []
    group_all = []

    batch_size = 300
    total_batches = (len(df_search_queries) + batch_size - 1) // batch_size
    print(f"Nombre de lots (batchs) à traiter : {total_batches}")

    for i, chunk in enumerate(chunk_data(df_search_queries, chunk_size=batch_size)):
        print(f"\n=== Traitement du batch {i+1}/{total_batches} ===")
        
        X_chunk, y_chunk, group_chunk = generate_training_data(chunk)

        if not X_chunk.empty:
            X_all.append(X_chunk)
            y_all.append(y_chunk)
            group_all.extend(group_chunk)  # ajouter les longueurs de chaque utilisateur

    # Fusionner tout
    if X_all:
        X_final = pd.concat(X_all, axis=0)
    else:
        X_final = pd.DataFrame()

    if y_all:
        y_final = pd.concat(y_all, axis=0)
    else:
        y_final = pd.Series()

    print("\n--- Résultat final ---")
    print("X_final shape: ", X_final.shape)
    print("y_final shape: ", y_final.shape)
    print("Nombre total de groupes: ", len(group_all))

    # Sauvegarder
    X_final.to_csv("custom_search_ranking/app/data/X.csv", index=False)
    y_final.to_csv("custom_search_ranking/app/data/y.csv", index=False)
    with open("custom_search_ranking/app/data/group.txt", "w") as f:
        f.write(",".join(map(str, group_all)))

    print("\nFichiers X.csv, y.csv et group.txt sauvegardés avec succès ")
