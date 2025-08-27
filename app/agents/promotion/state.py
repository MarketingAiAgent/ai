from __future__ import annotations
from typing import Any, Dict, List, Optional
from app.agents.orchestrator.state import PromotionSlots

ASK_PROMPT_MAP = {
    "target_type": "프로모션 종류를 선택해 주세요. 브랜드 대상 프로모션과 카테고리 대상 프로모션 기능을 지원합니다.",
    "focus_brand": "어떤 브랜드로 프로모션을 진행하고 싶으신가요? (예: 나이키, 아디다스, 삼성 등)",
    "focus_category": "어떤 카테고리로 프로모션을 진행하고 싶으신가요? (예: 스포츠웨어, 화장품, 전자제품 등)",
    "duration": "프로모션 기간을 알려주실 수 있을까요? (예: 2025-09-01 ~ 2025-09-14)",
    "selected_product": "어떤 제품으로 프로모션을 진행할까요? 아래 추천 목록에서 선택하시거나 직접 입력해주세요.",
    # target과 objective는 필수가 아니므로 ASK_PROMPT_MAP에서 제거
    # 이제 이 필드들에 대한 재질문이 발생하지 않음
}

def _is_filled(v) -> bool:
    return v not in (None, "", [])

def get_action_state(
    *,
    slots: Optional[PromotionSlots], 
) -> Dict[str, Any]:
    
    if slots is None:
        return {"intent_type": "none", "status": "skip",
                "missing_slots": [], "ask_prompts": [], "payload": {}}

    if not _is_filled(slots.target_type):
        # target_type와 duration을 함께 물어봄
        ask_prompts = [ASK_PROMPT_MAP["target_type"]]
        missing_slots = ["target_type"]
                
        return {
            "intent_type": "promotion", "status": "ask_for_slots",
            "missing_slots": missing_slots, "ask_prompts": ask_prompts,
            "payload": {},
        }

    # 필수 슬롯 목록 확인 (target은 선택사항으로 변경)
    ordered_missing: List[str] = []
    focus_key = None
    
    if not _is_filled(slots.focus):
        if slots.target_type == "brand":
            focus_key = "focus_brand"
        else:
            focus_key = "focus_category"
        ordered_missing.append("focus")

        return {
            "intent_type": "promotion", "status": "ask_for_slots",
            "missing_slots": ordered_missing, "ask_prompts": ASK_PROMPT_MAP[focus_key],
            "payload": slots.model_dump(),
        }

    # target은 필수가 아님 - 제거
    # if not _is_filled(slots.target):
    #     ordered_missing.append("target")

    # focus가 선택되었지만 해당 focus의 상품이 선택되지 않은 경우
    if _is_filled(slots.focus) and not _is_filled(slots.selected_product):
        focus_label = "브랜드" if slots.target_type == "brand" else "카테고리"
        return {
            "intent_type": "promotion",
            "status": "ask_for_product",
            "missing_slots": ["selected_product"],
            "ask_prompts": [f"{slots.focus} {focus_label}의 어떤 제품으로 프로모션을 진행할까요? 아래 추천 목록에서 선택하시거나 직접 입력해주세요."],
            "payload": slots.model_dump(),
        }
    
    if not _is_filled(slots.duration):
        ordered_missing.append("duration")

    # if ordered_missing:
    #     # 아직 채워야 할 기본 정보가 남은 경우
    #     ask_prompts = []
    #     for k in ordered_missing[:2]:
    #         if k == "focus" and focus_key:
    #             ask_prompts.append(ASK_PROMPT_MAP[focus_key])
    #         else:
    #             ask_prompts.append(ASK_PROMPT_MAP[k])
        
    #     return {
    #         "intent_type": "promotion", "status": "ask_for_slots",
    #         "missing_slots": ordered_missing, "ask_prompts": ask_prompts,
    #         "payload": {},
    #     }


    # 트렌드 반영을 원한다면 외부 데이터 수집 후 최종 기획서 생성
    if slots.wants_trend is True:
        return {
            "intent_type": "promotion", "status": "apply_trends",
            "missing_slots": [], "ask_prompts": [],
            "payload": slots.model_dump(),
        }
    
    # 트렌드 반영을 원하지 않으면 바로 최종 기획서 생성 (wants_trend가 False인 경우)
    if slots.wants_trend is False:
        return {
            "intent_type": "promotion", "status": "create_final_plan",
            "missing_slots": [], "ask_prompts": [],
            "payload": slots.model_dump(),
        }
    
    # 기본 정보는 모두 채워졌지만 트렌드 반영 여부가 결정되지 않은 경우
    if slots.wants_trend is None:
        return {
            "intent_type": "promotion", "status": "start_promotion",
            "missing_slots": [], "ask_prompts": [],
            "payload": slots.model_dump(),
        }
    
    # 기본값: 트렌드 반영 없이 최종 기획서 생성
    return {
        "intent_type": "promotion", "status": "create_final_plan",
        "missing_slots": [], "ask_prompts": [],
        "payload": slots.model_dump(),
    }
