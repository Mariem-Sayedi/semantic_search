import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rich import print
from rich.console import Console
from rich.table import Table



db_path = 'custom_search_ranking/app/data/LFF.db'
console = Console()


def display_matrix(title: str, df: pd.DataFrame, limit: int = 5):
    console.rule(f"[bold blue]{title}")
    console.print(df.head(limit))

def display_recommendations(user_guid, recommendations):
    table = Table(title=f"Top Recommandations pour l'utilisateur : [green]{user_guid}[/green]")
    table.add_column("Produit", style="cyan")
    table.add_column("Score", style="magenta")

    for prod, score in recommendations:
        table.add_row(str(prod), f"{score:.4f}")

    console.print(table)



def load_interactions_from_db(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    views_query = "SELECT user_guid, product_id, 'view' as interaction_type FROM product_views"
    cart_purchases_query = """
        SELECT user_guid, product_id, 
               CASE 
                   WHEN event_type = 'add_to_cart' THEN 'add_to_cart'
                   WHEN event_type = 'purchase' THEN 'purchase'
                   ELSE 'unknown'
               END as interaction_type
        FROM cart_purchases
    """

    views = pd.read_sql_query(views_query, conn)
    cart_purchases = pd.read_sql_query(cart_purchases_query, conn)
    conn.close()

    interactions = pd.concat([views, cart_purchases])
    
    #score selon le type d'interaction
    score_map = {"view": 3, "add_to_cart": 5, "purchase": 10}
    interactions["interaction_score"] = interactions["interaction_type"].map(score_map)

    return interactions



def build_weighted_user_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="user_guid", 
        columns="product_id", 
        values="interaction_score", 
        aggfunc='sum', 
        fill_value=0
    )


def score_collaboratif(product_id: str, user_guid: str, matrix: pd.DataFrame, similarity_df: pd.DataFrame) -> float:
    if user_guid not in similarity_df.index or product_id not in matrix.columns:
        return 0.0

    similar_users = similarity_df[user_guid].drop(index=user_guid)
    score = 0.0
    total_sim = 0.0

    for other_user, sim in similar_users.items():
        interaction_score = matrix.at[other_user, product_id]
        if interaction_score > 0:
            score += sim * interaction_score
            total_sim += sim

    return score / total_sim if total_sim > 0 else 0.0




def get_top_n_recommendations(user_guid, matrix, similarity_df, n=5):
    user_products = matrix.loc[user_guid]
    unseen_products = user_products[user_products == 0].index

    scores = {
        prod: score_collaboratif(prod, user_guid, matrix, similarity_df)
        for prod in unseen_products
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:n]


def compute_user_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(matrix.values)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)




def main():

   interactions = load_interactions_from_db(db_path)
   print("interactions :")
   display_matrix("Interactions", interactions)


   
   weighted_matrix = build_weighted_user_product_matrix(interactions)
   display_matrix("Matrice User-Product Pondérée", weighted_matrix)


   print("\nColonnes (product_id):")
   print(weighted_matrix.columns.tolist()[:5])

   similarity_df = compute_user_similarity_matrix(weighted_matrix)
   display_matrix("Matrice de Similarité", similarity_df)

   user_guid = weighted_matrix.index[50]
   product_id = weighted_matrix.columns[10]

   score = score_collaboratif(product_id, user_guid, weighted_matrix, similarity_df)
   print(f"\nScore collaboratif pour l'utilisateur {user_guid} sur le produit {product_id} : {score:.4f}")



   top_score = get_top_n_recommendations(user_guid, weighted_matrix, similarity_df, n=5)
   print(f"\nTop recommandations pour {user_guid} :")
   for prod, score in top_score:
     print(f"Produit : {prod}, Score : {score:.4f}")




if __name__ == "__main__":
    main()