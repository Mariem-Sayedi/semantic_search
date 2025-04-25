import json
import uuid
import time
from pipeline import personalized_ranking
import sqlite3
import pandas as pd

DB_PATH = "custom_search_ranking/app/data/LFF.db"
store_id = "0414"
session_id = str(uuid.uuid4())


def load_search_queries(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    search_query = "SELECT user_guid, search_query FROM searches"
    df_search = pd.read_sql_query(search_query, conn)
    conn.close()

    df_search_counts = df_search.groupby(['user_guid', 'search_query']).size().reset_index(name='search_count')
    return df_search_counts


# Récupérer les données de recherche
df_search_data = load_search_queries(DB_PATH)

# Ouvrir le fichier une seule fois en mode 'a' (append) pour ajouter les événements
with open("events.jsonl", "a") as f:
    # Itérer sur chaque combinaison unique d'utilisateur et de requête
    for index, row in df_search_data.iterrows():
        user_guid = row['user_guid']
        query = row['search_query']

        # Appel au pipeline de ranking personnalisé
        ranking_df = personalized_ranking(user_guid=user_guid, query=query, store_id=store_id)

        # Génération de l'objet JSON au format MetaRank
        ranking_event = {
            "event": "ranking",
            "id": str(uuid.uuid4()),
            "timestamp": str(int(time.time() * 1000)),
            "user": user_guid,
            "session": session_id,
            "fields": [
                {"name": "query", "value": query},
                {"name": "source", "value": "search"}
            ],
            "items": []
        }

        for _, row in ranking_df.iterrows():
            ranking_event["items"].append({
                "id": row["product_id"],
                "fields": [
                    {"name": "relevancy_score", "value": round(row["final_score"], 3)}
                ]
            })

        # Écrire l'événement au fichier JSONL, suivi d'une nouvelle ligne
        f.write(json.dumps(ranking_event) + "\n")

        print(f"Événement de ranking pour l'utilisateur '{user_guid}' et la requête '{query}' ajouté au fichier.")

print("Fichier 'events.jsonl' généré/mis à jour.")