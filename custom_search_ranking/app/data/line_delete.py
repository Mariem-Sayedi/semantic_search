import sqlite3


db_path = 'custom_search_ranking/app/data/LFF.db'

# Timestamp cible
target_timestamp = '2025-05-14 09:11:43.714159'

try:
    # Connexion à la base de données
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Suppression de la ligne avec le timestamp spécifié
    cursor.execute("DELETE FROM cart_purchases WHERE timestamp = ?", (target_timestamp,))
    conn.commit()

    print("Ligne supprimée avec succès.")
except sqlite3.Error as e:
    print(f"Erreur lors de l'accès à la base de données : {e}")
finally:
    # Fermeture de la connexion
    if conn:
        conn.close()