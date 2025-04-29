import pandas as pd
from app.services.search_products_api import fetch_products_from_api

pd.set_option("display.float_format", "{:.3f}".format)

def compute_score_promotion(promoRate: float) -> float:
    return round(min(promoRate, 100) / 100, 3)

def apply_promotion_score(df: pd.DataFrame) -> pd.DataFrame:
    df['score_promotion'] = df['promo_rate'].apply(compute_score_promotion)
    return df

if __name__ == "__main__":
    df_promos = fetch_products_from_api("chaise", store_id='0414')
    
    if df_promos.empty:
        print("Aucun produit trouvé.")
    else:
        df_promos = apply_promotion_score(df_promos)
        print(df_promos.sort_values(by="score_promotion", ascending=False))
