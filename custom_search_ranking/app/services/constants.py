# modèle entraîné
RANKING_MODEL_PATH = "custom_search_ranking/app/models/xgboost_ranking_model.pkl"
RANKING_MODEL_PATH_JSON = "custom_search_ranking/app/models/xgboost_ranking_model.json"


# base de données SQLite
DB_PATH = "custom_search_ranking/app/data/LFF.db"

PROCESSED_USERS_FILE = "custom_search_ranking/app/data/processed_users.txt"

# données d'entrainement
TRAINING_DATA_PATH = "custom_search_ranking/app/data/training_data_full.csv"
SUMMARY_DATA_PATH = "custom_search_ranking/app/data/search_results_summary.csv"


TOKEN_API_URL = "https://preprod-api.lafoirfouille.fr/occ/v2/token"
SEARCH_API_URL = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"