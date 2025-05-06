import requests
import pandas as pd
session = requests.Session()


def get_access_token():
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/token"

    response = session.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get("access_token")
    return None



def fetch_products_from_api(query: str, store_id: str) -> pd.DataFrame:
    url = "https://preprod-api.lafoirfouille.fr/occ/v2/products/search/"
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    cookies = {"preferredStoreCode": store_id}
    params = {"text": query}

    response = requests.get(url, headers=headers, params=params, cookies=cookies)
    if response.status_code != 200:
        return pd.DataFrame()

    results = response.json().get("searchPageData", {}).get("results", [])
    products = []
    for prod in results:
        product_id = prod.get("code")
        product_name = prod.get("name")
        promo_rate = prod.get("storeStockPrice", {}).get("promoRate", 0)
        products.append({
            "product_id": product_id,
            "product_name": product_name,
            "promo_rate": promo_rate
        })
    return pd.DataFrame(products)

