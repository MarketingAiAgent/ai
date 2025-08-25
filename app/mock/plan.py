from app.schema.chat import CreateBrandPlanResponse, CreateCategoryPlanResponse
from app.database.plans import create_plan
import uuid

def mock_create_plan(return_type, company, user_id):
    plan_id = uuid.uuid4().hex
    if return_type == "brand":
        plan_content = {
            "planId": plan_id, 
            "title":"Brand Plan",
            "mainBanner": "메인 배너입니다.",
            "couponSection": "쿠폰 섹션입니다.",
            "productSection": "상품 섹션입니다.",
            "eventNotes": "이벤트 노트입니다."
        }

        response = CreateBrandPlanResponse(**plan_content)
    else: 
        plan_content = {
            "planId": plan_id,
            "title": "Category Plan",
            "mainBanner": "메인 배너입니다.",
            "section1": "섹션 1입니다.",
            "section2": "섹션 2입니다.",
            "section3": "섹션 3입니다."
        }
        response = CreateCategoryPlanResponse(**plan_content)

    create_plan(plan_id, user_id, company, return_type, plan_content)
    
    return response