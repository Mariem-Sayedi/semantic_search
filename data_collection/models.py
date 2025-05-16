
from pydantic import BaseModel
from datetime import datetime
from typing import Optional



class Event(BaseModel):
    user_guid: str
    user_id: str
    product_id: str
    product_name: str
    quantity: Optional[int] = 1
    store_id: str
    event_type: str  
    timestamp: datetime = datetime.utcnow()

class ViewedProduct(BaseModel):
    user_guid: str
    user_id: str
    product_id: str
    summary: str
    webCategories: str
    store_id: str
    timestamp: datetime = datetime.utcnow()

class ViewedCategory(BaseModel):
    user_guid: str
    user_id: str
    category: str
    store_id: str
    timestamp: datetime = datetime.utcnow()

class SearchEvent(BaseModel):
    user_guid: str
    user_id: str        
    search_query: str
    store_id: str 
    timestamp: datetime = datetime.utcnow()

class Boost(BaseModel):
    target_type: str
    target_id: str
    store_id: str
    boost_score: int
    start_date: datetime
    end_date: datetime
    timestamp: datetime = datetime.utcnow()

     