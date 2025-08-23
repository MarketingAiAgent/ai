from app.schema.chat import CreateBrandPlanResponse, CreateCategoryPlanResponse
from app.database.plans import create_plan
import uuid

def mock_create_plan(return_type, company):
    plan_id = uuid.uuid4().hex
    if return_type == "brand":
        plan_content = {
            "planId": plan_id, 
            "title":"Brand Plan",
            "mainBanner": "",
            "couponSection": "",
            "productSection": "",
            "eventNotes": ""
        }

        response = CreateBrandPlanResponse(**plan_content)
    else: 
        plan_content = {
            "planId": plan_id,
            "title": "Category Plan",
            "mainBanner": "",
            "section1": "",
            "section2": "",
            "section3": ""
        }
        response = CreateCategoryPlanResponse(**plan_content)

    create_plan(plan_id, company, return_type, plan_content)
    
    return response