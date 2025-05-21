import requests
import pandas as pd
from custom_search_ranking.app.services.constants import TOKEN_API_URL, SEARCH_API_URL
from concurrent.futures import ThreadPoolExecutor, as_completed

session = requests.Session()



def get_access_token():

    response = session.get(TOKEN_API_URL)

    if response.status_code == 200:
        data = response.json()
        return data.get("access_token")
    return None





def fetch_single_product_page(query, store_id, page=0):
    access_token = get_access_token()
    if not access_token:
        return pd.DataFrame(), {}
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }

    cookies = {"preferredStoreCode": str(store_id)}
    params = {"q": query, "page": page}

    response = requests.get(SEARCH_API_URL, headers=headers, params=params, cookies=cookies)
    if response.status_code != 200:
        return pd.DataFrame(), {}
    
    json_data = response.json().get("searchPageData", {})
    raw_results = json_data.get("results", [])
    pagination = json_data.get("pagination", {})

    products = []
    for item in raw_results:
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

    return pd.DataFrame(products), pagination




def fetch_all_products(query, store_id):
    #recuperer page 0 pour obtenir les infos pagination
    first_df, pagination = fetch_single_product_page(query, store_id, page=0)
    if first_df.empty or not pagination:
        return pd.DataFrame(), pagination
    

    total_pages = pagination.get("numberOfPages", 1)
    all_dfs = [first_df]

    #charger page 1 à n en //
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(fetch_single_product_page, query, store_id, page)
            for page in range(1, total_pages)
        ]

        for future in as_completed(futures):
            df, _ = future.result()
            if not df.empty:
                all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df, pagination





def fetch_and_display_products(query, store_id, page=0):
    access_token = get_access_token()

    
    headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
    }
    cookies = {"preferredStoreCode": str(store_id)}

    params = {"q": query, "page": page}
    response = requests.get(SEARCH_API_URL, headers=headers, params=params, cookies=cookies)
    if response.status_code != 200:
        return pd.DataFrame()

    raw_results = response.json().get("searchPageData", {}).get("results", [])
    pagination = response.json().get("searchPageData", {}).get("pagination", [])


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

    return pd.DataFrame(products), pagination