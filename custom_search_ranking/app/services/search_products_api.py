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

    if response.status_code == 200:
        products = response.json()
        if products and 'searchPageData' in products and 'results' in products['searchPageData']:
            product_list = products['searchPageData']['results']
            table_data = [[p.get('name', 'code')] for p in product_list]
            if table_data:
              return product_list
    print("erreur ou aucun produit trouvé.")
    return []