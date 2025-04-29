import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
from constants import TRAINING_DATA_PATH

# 1 charger les données 
df = pd.read_csv(TRAINING_DATA_PATH)

# 2 séparer les features et la target (X et y)
features = [
    'score_promotion', 
    'score_collaboratif', 
    'score_svd',
    'score_saison', 
    'score_local_trend', 
    'score_global_trend', 
    'score_navigation_client'
]
X = df[features]
y = df['final_score'].apply(lambda x: max(0, int(round(x))))

# 3 générer la colonne group_id
df['group_id'] = df['user_guid'].astype(str) + '_' + df['search_query'].astype(str)

# 4 calculer la taille de chaque groupe
group_sizes = df['group_id'].value_counts().sort_index()

# 5 split train/val en gardant les groupes ensemble
unique_groups = group_sizes.index.tolist()
train_groups, val_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

# filtrer les données
train_idx = df['group_id'].isin(train_groups)
val_idx = df['group_id'].isin(val_groups)

X_train = X[train_idx]
y_train = y[train_idx].apply(lambda x: max(0, int(round(x))))
X_val = X[val_idx]
y_val = y[val_idx].apply(lambda x: max(0, int(round(x))))

# recalculer group_sizes pour train et val
group_train = df[train_idx]['group_id'].value_counts().sort_index().tolist()
group_val = df[val_idx]['group_id'].value_counts().sort_index().tolist()

# 6. créer les DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

dtrain.set_group(group_train)
dval.set_group(group_val)

# 7. paramètres XGBoost
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

# 8. entraîner
evallist = [(dtrain, 'train'), (dval, 'eval')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evallist,
    early_stopping_rounds=10
)

# 9. sauvegarder le modèle
bst.save_model('custom_search_ranking/app/models/xgboost_ranking_model.json')

with open('custom_search_ranking/app/models/xgboost_ranking_model.pkl', 'wb') as f:
    pickle.dump(bst, f)

print("Modèle XGBoost entraîné et sauvegardé avec succès !")
