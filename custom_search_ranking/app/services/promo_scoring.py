import pandas as pd
import requests
pd.set_option("display.float_format", "{:.3f}".format)

session = requests.Session()


def get_access_token():
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/token"

    response = session.get(url)

    print("Status:", response.status_code)
    print("Body:", response.text)

    if response.status_code == 200:
        data = response.json()
        return data.get("access_token")
    return None

def extract_promotions_from_api(query: str) -> pd.DataFrame:
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    cookies = {
    "preferredStoreCode": "0414"
    }
    params = {"text": query}

    response = requests.get(url, headers=headers, params=params, cookies=cookies)

    if response.status_code != 200:
        print("Erreur lors de l'appel API")
        return pd.DataFrame()

    
    products_data = response.json()
    if products_data and 'searchPageData' in products_data and 'results' in products_data['searchPageData']:
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


def compute_score_promotion(promoRate: float) -> float:
    return round(min(promoRate, 99) / 99, 3)



def apply_promotion_score(promotions_df: pd.DataFrame) -> pd.DataFrame:
   
    promotions_df['score_promotion'] = promotions_df['promo_rate'].apply(lambda d: compute_score_promotion(d))
    return promotions_df


if __name__ == "__main__":
    df_promos = extract_promotions_from_api("chaise")
    print(df_promos.sort_values(by="score_promotion", ascending=False))

