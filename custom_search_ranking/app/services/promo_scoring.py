import pandas as pd
import requests

session = requests.Session()


def extract_promotions_from_api(query: str) -> pd.DataFrame:
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    params = {"text": query}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print("Erreur lors de l'appel API")
        return pd.DataFrame()

    products_data = response.json()
    if 'searchPageData' not in products_data or 'results' not in products_data['searchPageData']:
        print("Aucun produit trouvé.")
        return pd.DataFrame()

    products = products_data['searchPageData']['results']

    promo_list = []
    for product in products:
        product_id = product.get("code")
        promo_info = product.get("storeStockPrice", {})
        promo_rate = promo_info.get("promoRate", 0)

        promo_list.append({
            "product_id": product_id,
            "promo_rate": promo_rate,
            "score_promotion": compute_score_promotion(promo_rate)
        })

    return pd.DataFrame(promo_list)

def extract_promotions_from_api(products: list) -> pd.DataFrame:
    promo_list = []

    for product in products:
        product_id = product.get("code")
        promo_info = product.get("storeStockPrice", {})
        promo_rate = promo_info.get("promoRate", 0)

        promo_list.append({
            "product_id": product_id,
            "promo_rate": promo_rate
        })

    return pd.DataFrame(promo_list)




def compute_score_promotion(discount: float) -> float:
    return min(discount, 99) / 99



def apply_promotion_score(promotions_df: pd.DataFrame) -> pd.DataFrame:
   
    promotions_df['score_promotion'] = promotions_df['promo_rate'].apply(lambda d: compute_score_promotion(d))
    return promotions_df


if __name__ == "__main__":
    df_promos = extract_promotions_from_api("chaise")
    print(df_promos.sort_values(by="score_promotion", ascending=False).head(10))

