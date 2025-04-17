import pandas as pd
import sqlite3
from rich.console import Console
from rich.table import Table

console = Console()
db_path = 'custom_search_ranking/app/data/LFF.db'


def load_interactions_from_db(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    views_query = "SELECT user_guid, product_id, 'view' as interaction_type FROM product_views"
   

    views = pd.read_sql_query(views_query, conn)
    conn.close()

    interactions = pd.concat([views])

    score_map = {"view": 3}
    interactions["interaction_score"] = interactions["interaction_type"].map(score_map)

    return interactions


def recommend_similar_category_products(user_guid, interactions, product_categories, top_n=5):
    user_interactions = interactions[interactions['user_guid'] == user_guid]
    seen_products = user_interactions['product_id'].unique()

    user_products_with_cat = pd.merge(user_interactions, product_categories, on='product_id')

    top_categories = (
        user_products_with_cat.groupby('webCategories')['interaction_score']
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    candidate_products = product_categories[
        product_categories['webCategories'].isin(top_categories)
    ]

    unseen_products = candidate_products[~candidate_products['product_id'].isin(seen_products)]

    return unseen_products['product_id'].head(top_n).tolist()


def display_category_recommendations(user_guid, recommendations):
    table = Table(title=f"recommandations Catégorielles pour l'utilisateur : [green]{user_guid}[/green]")
    table.add_column("Produit", style="cyan")

    for prod in recommendations:
        table.add_row(str(prod))

    console.print(table)


def main():
    conn = sqlite3.connect(db_path)
    interactions = load_interactions_from_db(db_path)
    product_categories = pd.read_sql_query("SELECT DISTINCT product_id, webCategories FROM product_views", conn)
    conn.close()

    user_guid = interactions['user_guid'].iloc[0]

    recos_cat = recommend_similar_category_products(user_guid, interactions, product_categories)
    display_category_recommendations(user_guid, recos_cat)


if __name__ == "__main__":
    main()
