import sqlite3
import pandas as pd

db_path = 'custom_search_ranking/app/data/LFF.db'



def get_top_viewed_categories(top_n):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT category FROM category_views", conn)
    conn.close()
    top_categories = df['category'].value_counts().reset_index()
    top_categories.columns = ['category', 'views']
    return top_categories.head(top_n)


def get_top_viewed_products(top_n):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT product_id FROM product_views", conn)
    conn.close()
    top_products = df['product_id'].value_counts().reset_index()
    top_products.columns = ['product_id', 'views']
    return top_products.head(top_n)


def get_top_added_to_cart_products(top_n):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT product_id, event_type FROM cart_purchases", conn)
    conn.close()
    added = df[df['event_type'] == 'add_to_cart']['product_id'].value_counts().reset_index()
    added.columns = ['product_id', 'added_to_cart']
    return added.head(top_n)


def get_top_purchased_products(top_n):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT product_id, event_type FROM cart_purchases", conn)
    conn.close()
    purchased = df[df['event_type'] == 'purchase']['product_id'].value_counts().reset_index()
    purchased.columns = ['product_id', 'purchased']
    return purchased.head(top_n)


if __name__ == "__main__":
    print("\n Top Catégories vues :")
    print(get_top_viewed_categories(5))

    print("\n Top Produits vus :")
    print(get_top_viewed_products(5))

    print("\n Top Produits ajoutés au panier :")
    print(get_top_added_to_cart_products(5))

    print("\n Top Produits achetés :")
    print(get_top_purchased_products(5))
