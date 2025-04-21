from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(
    title="LFF Recommendation API",
    description="API de personnalisation de recherche",
    version="1.0"
)

app.include_router(router)
