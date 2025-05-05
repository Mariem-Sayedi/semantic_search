from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import time

from custom_search_ranking.app.services.personalized_results_AI import personalize_ranking
from semantic_search.pipeline_semantic_search import traiter_requete


app = FastAPI()




@app.get("/search")
def search(query: str = Query(...), user_guid: str = Query(...), store_id: str = "0414"):
    start_time = time.time()
    response = full_search_pipeline(query, user_guid, store_id)
    end_time = time.time()
    response_time = round(end_time - start_time, 3)
    response["response_time_seconds"] = response_time
    return response


def full_search_pipeline(query: str, user_guid: str, store_id: str = "0414") -> dict:
   # étape 1 : Recherche sémantique
    semantic_output = traiter_requete(query)
    corrected_query = semantic_output["corrected_query"]
    expanded_terms = semantic_output["expanded_terms"]
    semantic_df_products = semantic_output["ranked_products"]
    print("df_products", semantic_df_products)
    # étape 2 : Personnalisation du ranking
    personalized_results_df = personalize_ranking(user_guid, semantic_df_products, store_id)

    personalized_results = personalized_results_df[["product_id", "predicted_score"]].to_dict(orient="records")

    return {
        "query_corrected": corrected_query,
        "expanded_terms": expanded_terms,
        "results": personalized_results
    }
