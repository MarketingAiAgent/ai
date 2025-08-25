from __future__ import annotations
from typing import Any, Dict, List, Optional
from app.agents.orchestrator.state import PromotionSlots

ASK_PROMPT_MAP = {
    "target_type": "타겟 종류를 선택해 주세요. (brand | category)",
    "brand": "타겟 브랜드를 알려주실 수 있을까요?",
    "target": "타겟 카테고리/고객군을 알려주실 수 있을까요?",
    "objective": "이번 프로모션의 목표(예: 매출 증대, 신규 고객 유입)를 알려주실 수 있을까요?",
    "duration": "프로모션 기간을 알려주실 수 있을까요? (예: 2025-09-01 ~ 2025-09-14)",
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
        return {
            "intent_type": "promotion", "status": "ask_for_slots",
            "missing_slots": ["target_type"], "ask_prompts": [ASK_PROMPT_MAP["target_type"]],
            "payload": {},
        }

    # 필수 슬롯 목록 확인
    ordered_missing: List[str] = []
    if slots.target_type == "brand":
        if not _is_filled(slots.brand):
            ordered_missing.append("brand")
    elif slots.target_type == "category":
        if not _is_filled(slots.target):
            ordered_missing.append("target")

    if not _is_filled(slots.objective):
        ordered_missing.append("objective")
    if not _is_filled(slots.duration):
        ordered_missing.append("duration")

    if ordered_missing:
        # 아직 채워야 할 기본 정보가 남은 경우
        return {
            "intent_type": "promotion", "status": "ask_for_slots",
            "missing_slots": ordered_missing, "ask_prompts": [ASK_PROMPT_MAP[k] for k in ordered_missing[:2]],
            "payload": {},
        }

    # --- 👇 여기가 핵심적인 변경 부분입니다 ---
    # 기본 정보는 다 채워졌지만, '어떤 제품'으로 할지가 빠진 경우
    if not _is_filled(slots.selected_product):
        # options_generator를 호출해야 한다는 신호로 'ask_for_product' 상태를 반환
        return {
            "intent_type": "promotion",
            "status": "ask_for_product", # 새로운 상태
            "missing_slots": ["selected_product"],
            "ask_prompts": ["어떤 제품으로 프로모션을 진행할까요? 아래 추천 목록에서 선택하시거나 직접 입력해주세요."],
            "payload": slots.model_dump(),
        }
    # ------------------------------------

    # 제품까지 모든 정보가 완벽하게 채워졌을 때만 'start_promotion' 상태가 됨
    return {
        "intent_type": "promotion", "status": "start_promotion",
        "missing_slots": [], "ask_prompts": [],
        "payload": slots.model_dump(),
    }
