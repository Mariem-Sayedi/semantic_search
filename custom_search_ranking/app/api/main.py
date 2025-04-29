from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import sys
import os
from app.services.pipelineAI import personalized_ranking


app = FastAPI()

class RankingRequest(BaseModel):
    user_guid: str
    query: str
    store_id: str

@app.post("/rank")
def rank_products(request: RankingRequest):
    results_df = personalized_ranking(
        user_guid=request.user_guid,
        query=request.query,
        store_id=request.store_id
    )

    if results_df.empty:
        return {"message": "Aucun produit trouvé pour cette requête."}

    return {
        "ranking": results_df[['product_id', 'predicted_score']].to_dict(orient="records")
    }
