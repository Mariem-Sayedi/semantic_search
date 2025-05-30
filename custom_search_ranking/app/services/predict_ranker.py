import pandas as pd
import xgboost as xgb
import pickle

model_path = "custom_search_ranking/app/test/test1/xgboost_ranking_model1.json"
bst = xgb.Booster()
bst.load_model(model_path)




def predict_with_model(df_features: pd.DataFrame, model_path: str) -> pd.Series:
    features = [
        'score_promotion', 
        'score_collaboratif', 
        'score_svd',
        'score_season', 
        'score_local_trend', 
        'score_global_trend', 
        'score_navigation_client'
    ]
    
    dmatrix = xgb.DMatrix(df_features[features])
    
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    preds = bst.predict(dmatrix)
    
    return pd.Series(preds, index=df_features.index)
