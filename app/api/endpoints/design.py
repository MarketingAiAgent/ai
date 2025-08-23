from fastapi import APIRouter
from app.database.plans import save_design

router = APIRouter(tags=["Design"])

@router.post("/design")
def new_design(request): 
    
    url = "https://drive.google.com/uc?export=download&id=1j7ttxTtW5FCcKSwSQVCqwbp4_dFIGYt6"
    #--- 
    #--- 
    save_design(request.plan_id, url)
    return {
        "plan_id": request.plan_id,
        "url": url
    }