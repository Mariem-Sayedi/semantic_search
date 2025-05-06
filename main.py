from fastapi import FastAPI, Query
import time
from fastapi.responses import JSONResponse
import pandas as pd
from custom_search_ranking.app.services.personalized_results_AI import personalize_ranking
from custom_search_ranking.app.services.pipelineAI import personalized_ranking

from semantic_search.pipeline_semantic_search import traiter_requete


app = FastAPI()




@app.get("/customized-ranking")
def customized_search(store_id, query: str = Query(...), user_guid: str = Query(...)):
    start_time = time.time()
    df = personalized_ranking(user_guid, query, store_id)
    end_time = time.time()
    response_time = round(end_time - start_time, 3)

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
def semantic_search(store_id, query: str = Query(...)):
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



@app.get("/semantic-customized-ranking")
def semantic_customized_ranking(store_id, query: str = Query(...), user_guid: str = Query(...)):
    start_time = time.time()

    # Step 1: Perform semantic search
    semantic_output = traiter_requete(query, store_id)
    corrected_query = semantic_output["query_corrected"]
    expanded_terms = semantic_output["expanded_terms"]
    semantic_df_products = semantic_output["results"]
    # Step 2: Perform personalized ranking on semantic search results
    if not semantic_df_products:
        return JSONResponse(content={
            "query_corrected": corrected_query,
            "expanded_terms": expanded_terms,
            "results": [],
            "response_time_seconds": round(time.time() - start_time, 3)
        })

    semantic_df_products = pd.DataFrame(semantic_df_products)
    ranked_products = personalize_ranking(user_guid, semantic_df_products, store_id)

    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    if ranked_products.empty:
        return JSONResponse(content={
            "query_corrected": corrected_query,
            "expanded_terms": expanded_terms,
            "results": [],
            "response_time_seconds": response_time
        })

    results = [
        {
            "product_name": row["product_name"],
            "product_id": row["product_id"],
            "predicted_score": round(row["predicted_score"], 3)
        }
        for _, row in ranked_products.iterrows()
    ]

    return JSONResponse(content={
        "query_corrected": corrected_query,
        "expanded_terms": expanded_terms,
        "results": results,
        "response_time_seconds": response_time
    })