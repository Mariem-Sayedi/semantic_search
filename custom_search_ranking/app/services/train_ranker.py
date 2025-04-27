import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# 0. Charger X, y, group
X = pd.read_csv("custom_search_ranking/app/data/X.csv")
y = pd.read_csv("custom_search_ranking/app/data/y.csv").squeeze()
with open("custom_search_ranking/app/data/group.txt", "r") as f:
    group = list(map(int, f.read().split(",")))

# 1. Reconstituer les groupes
# On construit une colonne 'group_id' pour chaque requête
group_id = []
current_group = 0
for count in group:
    group_id += [current_group] * count
    current_group += 1
group_id = pd.Series(group_id)

# 2. Séparer train/val par groupes
X_train, X_val, y_train, y_val, group_id_train, group_id_val = train_test_split(
    X, y, group_id, test_size=0.2, random_state=42, stratify=group_id
)

# 3. Recalculer group counts pour train/val
group_train = group_id_train.value_counts().sort_index().tolist()
group_val = group_id_val.value_counts().sort_index().tolist()

# 4. Préparer le DMatrix pour XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

dtrain.set_group(group_train)
dval.set_group(group_val)

# 5. Définir les hyperparamètres du modèle
params = {
    'objective': 'rank:pairwise',
    'eta': 0.1,
    'gamma': 1.0,
    'min_child_weight': 0.1,
    'max_depth': 6,
    'eval_metric': 'ndcg',
    'tree_method': 'hist',
    'verbosity': 1
}

# 6. Entraîner
evallist = [(dtrain, 'train'), (dval, 'eval')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evallist,
    early_stopping_rounds=10
)

# 7. Sauvegarder le modèle
bst.save_model('custom_search_ranking/app/models/xgboost_ranking_model.json')

# Optionnel : sauvegarder aussi avec pickle
with open('custom_search_ranking/app/models/xgboost_ranking_model.pkl', 'wb') as f:
    pickle.dump(bst, f)

print(" Modèle entraîné et sauvegardé !")
