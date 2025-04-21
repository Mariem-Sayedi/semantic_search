from fastapi import APIRouter
from app.services.scoring import compute_product_score

router = APIRouter()

@router.get("/score")
def get_product_score(user_id: str, product_id: str):
    score = compute_product_score(user_id, product_id)
    return {"product_id": product_id, "score": score}
