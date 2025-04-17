from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    id: str
    name: str
    category: str
    is_promo: bool
    seasonal_tags: List[str]
    available_stores: List[str]
    popularity: float
    boost_value: Optional[float] = 0.0

class User(BaseModel):
    id: str
    cart: List[str]
    viewed_products: List[str]
    purchased_products: List[str]
    browsed_categories: List[str]
    store: str
    country: str
