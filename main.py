from fastapi import FastAPI, Query
import time
from fastapi.responses import JSONResponse
import pandas as pd
from custom_search_ranking.app.services.personalized_results_AI import personalize_ranking
from custom_search_ranking.app.services.pipelineAI import personalized_ranking

from semantic_search.pipeline_semantic_search import traiter_requete
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import HTTPException
from data_collection.models import Event, SearchEvent, ViewedProduct, ViewedCategory, Boost
from data_collection.storage_bd_sql import save_event, get_all_events, save_search_query, save_viewed_product, save_viewed_category, save_admin_boost, save_update_admin_boost, save_delete_admin_boost, get_admin_boost_by_id, get_all_admin_boosts
import math


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # autoriser ttes les méthodes
    allow_headers=["*"],
)





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
    pagination = semantic_output["pagination"]
    return {
        "query_corrected": corrected_query,
        "expanded_terms": expanded_terms,
        "results": semantic_df_products,
        "response_time": response_time,
        "pagination": pagination
    }



@app.get("/semantic-customized-ranking")
def semantic_customized_ranking(store_id, page, query: str = Query(...), user_guid: str = Query(...)):
    start_time = time.time()

    # Step 1: Perform semantic search
    semantic_output = traiter_requete(query, store_id)
    corrected_query = semantic_output["query_corrected"]
    expanded_terms = semantic_output["expanded_terms"]
    semantic_df_products = semantic_output["results"]
    pagination = semantic_output["pagination"]
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
    


    page = int(page)
    page_size = pagination.get("pageSize", 42)
    total_results = len(ranked_products)
    nb_pages = math.ceil(total_results / page_size)

    start = (page) * page_size
    end = start + page_size


    paginated_products = ranked_products.iloc[start:end]


    results = [
        {
            "product_name": row["product_name"],
            "product_id": row["product_id"],
            "predicted_score": round(row["predicted_score"], 3),
            "product_url": row["product_url"],
            "promo_rate": row["promo_rate"],
            "product_brand": row["product_brand"],
            "image_url": row["image_url"],
            "price": row["price"],
            "gross_price": row["gross_price"],
            "store_stock_price": row["store_stock_price"],
            "badges": row["badges"],
        }
        for _, row in paginated_products.iterrows()
    ]

    pagination.update({
        "currentPage": page,
        "numberOfPages": nb_pages,
        "totalNumberOfResults": total_results
    })

    return JSONResponse(content={
        "query_corrected": corrected_query,
        "expanded_terms": expanded_terms,
        "results": results,
        "response_time_seconds": response_time,
        "pagination": pagination
    })





@app.post("/track-event", status_code=201)
async def track_event(event: Event):
    """Suivre un événement utilisateur (ajout au panier, achat...)"""
    try:
        save_event(event.model_dump())

        return {"status": "success", "message": "Event tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking event: {str(e)}")
    

@app.post("/track-product", status_code=201)
async def track_product(viewedProduct: ViewedProduct):
    """Suivre un événement utilisateur: vue produit"""
    try:
        save_viewed_product(viewedProduct.model_dump()) 
        return {"status": "success", "message": "Viewed product tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking viewed product: {str(e)}")
    
@app.post("/track-category", status_code=201)
async def track_category(viewedCategory: ViewedCategory):
    """Suivre un événement utilisateur: vue categorie"""
    try:
        save_viewed_category(viewedCategory.model_dump()) 
        return {"status": "success", "message": "Viewed categorie tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking viewed categorie: {str(e)}")


@app.get("/events")
async def get_events():
    """Obtenir tous les événements enregistrés"""
    try:
        events = get_all_events()
        return {"status": "success", "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving events: {str(e)}")



@app.post("/search", status_code=201)
async def track_search(search_event: SearchEvent):
    """Suivre les recherches effectuées par un utilisateur"""
    try:
        # Sauvegarde de l'événement de recherche
        save_search_query(search_event.model_dump()) 
        return {"status": "success", "message": "Search tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking search: {str(e)}")

@app.post("/admin/boosts/", status_code=201)
def create_admin_boost(boost: Boost):
    """créer un boost de score """
    try:
        # Sauvegarde du boost admin
       save_admin_boost(boost.model_dump()) 
       return {"status": "success", "message": "Score boost created successfully", "boost": boost}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating boost: {str(e)}")
    

@app.put("/admin/boosts/{boost_id}")
def update_admin_boost(boost_id: int, boost: Boost):
    """Mettre à jour un boost de score"""
    try:
        save_update_admin_boost(boost_id, boost.model_dump()) 
        updated_boost = get_admin_boost_by_id(boost_id)
        return {"status": "success", "message": "Score boost updated successfully", "boost": updated_boost}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating boost: {str(e)}")


@app.delete("/admin/boosts/{boost_id}")
def delete_admin_boost(boost_id: int):
    """Supprimer un boost de score"""
    try:
        save_delete_admin_boost(boost_id)
        return {"status": "success", "message": "Score boost deleted successfully"}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting boost: {str(e)}")

@app.get("/admin/boosts/{boost_id}")
def read_admin_boost(boost_id: int):
    """Route pour récupérer un boost par ID"""
    try:
        boost = get_admin_boost_by_id(boost_id)
        return {"status": "success", "boosts": boost}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")



@app.get("/admin/boosts")
def read_all_admin_boosts():
    """Route pour récupérer tous les boosts"""
    try:
        boosts = get_all_admin_boosts()
        return {"status": "success", "boosts": boosts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")