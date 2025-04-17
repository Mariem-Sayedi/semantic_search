from fastapi import FastAPI
from app.models import Product, User
from app.scoring import calculate_score


app = FastAPI()

@app.post("/ranking/")
def rank_products(user: User, products: list[Product]):
    # Appliquer le ranking
    scored = [
        {"product": p, "score": calculate_score(p, user, context={}, date=today, weights=default_weights)}
        for p in products
    ]
    # Trier par score
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    return ranked
