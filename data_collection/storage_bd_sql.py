import sqlite3
from fastapi import HTTPException
from custom_search_ranking.app.services.constants import DB_PATH

DB_PATH = "custom_search_ranking/app/data/LFF.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_views (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_guid TEXT,
        user_id TEXT,
        product_id TEXT,
        summary TEXT,
        webCategories TEXT,
        store_id TEXT,    
        timestamp TEXT
    )
""")


cursor.execute("""
    CREATE TABLE IF NOT EXISTS category_views (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_guid TEXT,
        user_id TEXT,
        category TEXT,
        store_id TEXT,
        timestamp TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS cart_purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_guid TEXT,
        user_id TEXT,
        product_id TEXT,
        product_name TEXT,
        quantity INTEGER,
        store_id TEXT,
        event_type TEXT,
        timestamp TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_guid TEXT,
        user_id TEXT,
        search_query TEXT,
        store_id TEXT,
        timestamp TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS admin_boosts (
        boost_id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type TEXT,
        target_id TEXT,
        store_id TEXT,
        boost_score INTEGER,
        start_date DATE,
        end_date DATE,
        timestamp TEXT
    )
""")




conn.commit()

import json


def save_admin_boost(boost: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
    INSERT INTO admin_boosts (
        target_type, target_id, store_id, boost_score, start_date, end_date, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    boost["target_type"],
    boost["target_id"],
    boost["store_id"],
    boost["boost_score"],
    boost.get("start_date"),
    boost.get("end_date"),
    boost["timestamp"]
))
    conn.commit()

def save_update_admin_boost(boost_id: int, boost: dict):

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_boosts WHERE boost_id = ?", (boost_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Boost non trouvé")

        cursor.execute("""
            UPDATE admin_boosts
            SET  target_type = ?, target_id = ?, store_id = ?, boost_score = ?, timestamp = ?
            WHERE boost_id = ?
        """, (
            boost["target_type"],
            boost["target_id"],
            boost["store_id"],
            boost["boost_score"],
            boost["timestamp"],
            boost_id
        ))
        conn.commit()



def save_delete_admin_boost(boost_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_boosts WHERE boost_id = ?", (boost_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Boost non trouvé")
        
        cursor.execute("DELETE FROM admin_boosts WHERE boost_id = ?", (boost_id,))
        conn.commit()

def get_admin_boost_by_id(boost_id: int) -> dict:
    """Récupérer un boost depuis la base de données par son ID"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_boosts WHERE boost_id = ?", (boost_id,))
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Boost non trouvé")
        
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))
    

def get_all_admin_boosts() -> list[dict]:
    """Récupérer tous les boosts depuis la base de données"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_boosts")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    



def save_event(event: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO cart_purchases (user_guid, user_id, product_id, product_name, quantity, store_id, event_type, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                       (event["user_guid"], event["user_id"], event["product_id"], event["product_name"], event["quantity"], event["store_id"],  event["event_type"],event["timestamp"]))
        conn.commit()

def save_viewed_product(viewedProduct: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO product_views (user_guid, user_id, product_id, summary, webCategories, store_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                       (viewedProduct["user_guid"], viewedProduct["user_id"], viewedProduct["product_id"], viewedProduct["summary"], viewedProduct["webCategories"], viewedProduct["store_id"], viewedProduct["timestamp"]))
        conn.commit()

def save_viewed_category(viewedCategory: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO category_views (user_guid, user_id, category, store_id, timestamp) VALUES (?, ?, ?, ?, ?)", 
                       (viewedCategory["user_guid"], viewedCategory["user_id"], viewedCategory["category"],  viewedCategory["store_id"], viewedCategory["timestamp"]))
        conn.commit()

def get_all_events():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cart_purchases")
        return cursor.fetchall()



def save_search_query(search_query: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO searches (user_guid, user_id, search_query, store_id, timestamp) VALUES (?, ?, ?, ?, ?)", 
                       (search_query["user_guid"], search_query["user_id"], search_query["search_query"],  search_query["store_id"], search_query["timestamp"]))
        conn.commit()

# import os

# DB_PATH = "data/LFF.db"

# if os.path.exists(DB_PATH):
#     os.remove(DB_PATH)
#     print("Ancienne base supprimée.")
# else:
#     print("Aucune base existante trouvée.")


def show_table_data(table_name):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()

        if rows:
            for row in rows:
                print(row)
        else:
            print(f"La table '{table_name}' est vide.")

# show_table_data("product_views")
# show_table_data("searches")
# show_table_data("cart_purchases")
# show_table_data("category_views")

# import sqlite3

# DB_PATH = "custom_search_ranking/app/data/LFF.db"

# with sqlite3.connect(DB_PATH) as conn:
#     cursor = conn.cursor()
#     cursor.execute("DROP TABLE IF EXISTS admin_boosts")
#     conn.commit()
#     print("Table categories_views supprimée.")

