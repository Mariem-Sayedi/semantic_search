import sqlite3
import pandas as pd
pd.set_option("display.float_format", "{:.3f}".format)


DB_PATH = "custom_search_ranking/app/data/LFF.db"


def compute_product_views_score(store_id: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT product_id, store_id FROM product_views", conn)
    df = df[df['store_id'] == store_id]
    counts = df['product_id'].value_counts().reset_index()
    counts.columns = ['product_id', 'views']
    local_nb_views = counts['views'].sum()
    # print("local_store_nb_views", local_nb_views)
    counts['views_score'] = counts['views'] / local_nb_views
    return counts[['product_id', 'views_score']]



def get_top_viewed_products(top_n):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT product_id FROM product_views", conn)
    conn.close()
    top_products = df['product_id'].value_counts().reset_index()
    top_products.columns = ['product_id', 'views']
    return top_products.head(top_n)




def compute_category_views_score(store_id: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT category, store_id FROM category_views", conn)
    df = df[df['store_id'] == store_id]
    counts = df['category'].value_counts().reset_index()
    counts.columns = ['category', 'views']
    nb_views = counts['views'].sum()
    counts['views_score'] = counts['views'] / nb_views
    return counts[['category', 'views_score']]




def compute_cart_score(store_id: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT product_id, store_id, event_type FROM cart_purchases", conn)
    df = df[(df['store_id'] == store_id) & (df['event_type'] == 'add_to_cart')]
    counts = df['product_id'].value_counts().reset_index()
    counts.columns = ['product_id', 'cart_adds']
    nb_cart = counts['cart_adds'].sum()
    counts['cart_score'] = counts['cart_adds'] / nb_cart
    return counts[['product_id', 'cart_score']]





def compute_purchase_score(store_id: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT product_id, store_id, event_type FROM cart_purchases", conn)
    df = df[(df['store_id'] == store_id) & (df['event_type'] == 'purchase')]
    counts = df['product_id'].value_counts().reset_index()
    counts.columns = ['product_id', 'purchases']
    nb_cart = counts['purchases'].sum()
    counts['purchase_score'] = counts['purchases'] / nb_cart
    return counts[['product_id', 'purchase_score']]





def compute_local_trend_score(store_id: str) -> pd.DataFrame:
    df_views = compute_product_views_score(store_id)
    df_cart = compute_cart_score(store_id)
    df_purchase = compute_purchase_score(store_id)

    
    df = df_views.merge(df_cart, on='product_id', how='outer')
    df = df.merge(df_purchase, on='product_id', how='outer')

   
    df.fillna(0, inplace=True)

    
    df['score_trend'] = (
        0.3 * df['views_score'] +
        0.5 * df['cart_score'] +
        0.8 * df['purchase_score']
    )
    return df.sort_values(by='score_trend', ascending=False)





if __name__ == "__main__":

    store_id = "0414"

    trend_df = compute_local_trend_score(store_id)
    print("\n produits tendance par type d'interaction (store", store_id, ")")
    print(trend_df)
