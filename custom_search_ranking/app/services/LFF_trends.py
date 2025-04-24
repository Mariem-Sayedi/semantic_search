import sqlite3
import pandas as pd

DB_PATH = "custom_search_ranking/app/data/LFF.db"
pd.set_option("display.float_format", "{:.3f}".format)



def get_top_viewed_products(top_n):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT product_id FROM product_views", conn)
    conn.close()
    top_products = df['product_id'].value_counts().reset_index()
    top_products.columns = ['product_id', 'views']
    return top_products.head(top_n)


def get_top_added_to_cart_products(top_n):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT product_id, event_type FROM cart_purchases", conn)
    conn.close()
    added = df[df['event_type'] == 'add_to_cart']['product_id'].value_counts().reset_index()
    added.columns = ['product_id', 'added_to_cart']
    return added.head(top_n)


def get_top_purchased_products(top_n):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT product_id, event_type FROM cart_purchases", conn)
    conn.close()
    purchased = df[df['event_type'] == 'purchase']['product_id'].value_counts().reset_index()
    purchased.columns = ['product_id', 'purchased']
    return purchased.head(top_n)



def compute_product_views_score() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT product_id FROM product_views", conn)
    counts = df['product_id'].value_counts().reset_index()
    counts.columns = ['product_id', 'views']
    global_nb_views = counts['views'].sum()
    print("LFF nb_views", global_nb_views)
    counts['views_score'] = counts['views'] / global_nb_views
    return counts[['product_id', 'views_score']]


def compute_category_views_score() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT category FROM category_views", conn)
    counts = df['category'].value_counts().reset_index()
    counts.columns = ['category', 'views']
    nb_views = counts['views'].sum()
    counts['views_score'] = counts['views'] / nb_views
    return counts[['category', 'views_score']]




def compute_cart_score() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT product_id, event_type FROM cart_purchases", conn)
    df = df[(df['event_type'] == 'add_to_cart')]
    counts = df['product_id'].value_counts().reset_index()
    counts.columns = ['product_id', 'cart_adds']
    nb_cart = counts['cart_adds'].sum()
    counts['cart_score'] = counts['cart_adds'] / nb_cart
    return counts[['product_id', 'cart_score']]





def compute_purchase_score() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT product_id, event_type FROM cart_purchases", conn)
    df = df[(df['event_type'] == 'purchase')]
    counts = df['product_id'].value_counts().reset_index()
    counts.columns = ['product_id', 'purchases']
    nb_cart = counts['purchases'].sum()
    counts['purchase_score'] = counts['purchases'] / nb_cart
    return counts[['product_id', 'purchase_score']]





def compute_global_trend_score() -> pd.DataFrame:
    df_views = compute_product_views_score()
    df_cart = compute_cart_score()
    df_purchase = compute_purchase_score()

    
    df = df_views.merge(df_cart, on='product_id', how='outer')
    df = df.merge(df_purchase, on='product_id', how='outer')

   
    df.fillna(0, inplace=True)

    
    df['score_trend'] = (
        3 * df['views_score'] +
        5 * df['cart_score'] +
        8 * df['purchase_score']
    )
    return df.sort_values(by='score_trend', ascending=False)

if __name__ == "__main__":
    print("\n Top Produits vus :")
    print(get_top_viewed_products(5))

    print("\n Top Produits ajoutés au panier :")
    print(get_top_added_to_cart_products(5))


    trend_df = compute_global_trend_score()
    print("\n produits tendance par type d'interaction")
    print(trend_df.head(10))
