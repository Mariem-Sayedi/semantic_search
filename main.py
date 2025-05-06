from fastapi import FastAPI, Query
import time
from fastapi.responses import JSONResponse

from custom_search_ranking.app.services.personalized_results_AI import personalize_ranking
from custom_search_ranking.app.services.pipelineAI import personalized_ranking

from semantic_search.pipeline_semantic_search import traiter_requete


app = FastAPI()




@app.get("/customized-ranking")
def customized_search(query: str = Query(...), user_guid: str = Query(...), store_id: str = "0414"):
    start_time = time.time()
    df = personalized_ranking(user_guid, query, store_id)
    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    print(df)
    if df.empty:
        return JSONResponse(content={
            "results": [],
            "response_time_seconds": response_time
        })

    results = [
        {
            "product_name": row["product_name"],
            "product_id": row["product_id"],
            "predicted_score": round(row["predicted_score"], 3)
        }
        for _, row in df.iterrows()
    ]

    return JSONResponse(content={
        "results": results,
        "response_time_seconds": response_time
    })



@app.get("/semantic-search")
def semantic_search(query: str = Query(...), store_id: str = "0414"):
    start_time = time.time()
    semantic_output = traiter_requete(query, store_id)
    corrected_query = semantic_output["query_corrected"]
    expanded_terms = semantic_output["expanded_terms"]
    semantic_df_products = semantic_output["results"]
    end_time = time.time()
    response_time = round(end_time - start_time, 3)
    
    return {
        "query_corrected": corrected_query,
        "expanded_terms": expanded_terms,
        "results": semantic_df_products,
        "response_time": response_time
    }


# def full_search_pipeline(query: str, user_guid: str, store_id: str = "0414") -> dict:
#    # étape 1 : Recherche sémantique
#     semantic_output = traiter_requete(query)
#     corrected_query = semantic_output["corrected_query"]
#     expanded_terms = semantic_output["expanded_terms"]
#     semantic_df_products = semantic_output["products"]
#     # étape 2 : Personnalisation du ranking
#     personalized_results_df = personalized_ranking(user_guid, corrected_query, store_id)

#     personalized_results = personalized_results_df[["product_id", "predicted_score"]].to_dict(orient="records")

#     return {
#         "query_corrected": corrected_query,
#         "expanded_terms": expanded_terms,
#         "results": personalized_results
#     }
