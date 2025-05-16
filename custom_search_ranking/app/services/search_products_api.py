import requests
import pandas as pd
from custom_search_ranking.app.services.constants import TOKEN_API_URL, SEARCH_API_URL


session = requests.Session()



def get_access_token():

    response = session.get(TOKEN_API_URL)

    if response.status_code == 200:
        data = response.json()
        return data.get("access_token")
    return None

def fetch_and_display_products(query, store_id):
    access_token = get_access_token()

    
    headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
    }
    cookies = {"preferredStoreCode": str(store_id)}

    params = {"text": query}
    response = requests.get(SEARCH_API_URL, headers=headers, params=params, cookies=cookies)
    if response.status_code != 200:
        return pd.DataFrame()

    raw_results = response.json().get("searchPageData", {}).get("results", [])

    products = []
    for item in raw_results:

        # Extraction des infos nécessaires
        product_id = item.get("code", "")
        product_name = item.get("name", "")
        product_url = item.get("url", "")
        formatted_price = item.get("price", {}).get("formattedValue", "")
        image_urls = [img["url"] for img in item.get("images", []) if img.get("format") == "product"]
        promo_rate = item.get("storeStockPrice", {}).get("promoRate", 0)
        product_brand = item.get("brand", "")
        price = item.get("storeStockPrice", {}).get("price", 0.0)
        gross_price = item.get("storeStockPrice", {}).get("grossPrice", 0.0)
        store_stock_price = item.get("storeStockPrice", {})
        products.append({
            "product_id": product_id,
            "product_name": product_name,
            "product_url": product_url,
            "image_url": image_urls,
            "promo_rate": promo_rate,
            "product_brand": product_brand,
            "price": price,
            "gross_price": gross_price,
            "store_stock_price": store_stock_price
        })

    return pd.DataFrame(products)