from __future__ import annotations
from typing import Any, Dict, List, Optional
from app.agents.orchestrator.state import PromotionSlots

ASK_PROMPT_MAP = {
    "target_type": "타겟 종류를 선택해 주세요. (brand_target | category_target)",
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
            "intent_type": "promotion",
            "status": "ask_for_slots",
            "missing_slots": ["target_type"],
            "ask_prompts": [ASK_PROMPT_MAP["target_type"]],
            "payload": {},
        }

    ordered_missing: List[str] = []
    if slots.target_type == "brand_target":
        if not _is_filled(slots.brand):
            ordered_missing.append("brand")
    elif slots.target_type == "category_target":
        if not _is_filled(slots.target):
            ordered_missing.append("target")
    else:
        return {
            "intent_type": "promotion",
            "status": "ask_for_slots",
            "missing_slots": ["target_type"],
            "ask_prompts": [ASK_PROMPT_MAP["target_type"]],
            "payload": {},
        }

    if not _is_filled(slots.objective):
        ordered_missing.append("objective")
    if not _is_filled(slots.duration):
        ordered_missing.append("duration")

    if ordered_missing:
        asks = [ASK_PROMPT_MAP[k] for k in ordered_missing[:2]]
        return {
            "intent_type": "promotion",
            "status": "ask_for_slots",
            "missing_slots": ordered_missing,
            "ask_prompts": asks,
            "payload": {},
        }

    payload = {
        "objective": slots.objective,
        "target_type": slots.target_type,
        "target": slots.target,
        "brand": slots.brand,
        "selected_product": slots.selected_product,
        "duration": slots.duration,
        "product_options": slots.product_options,
    }
    return {
        "intent_type": "promotion",
        "status": "start_promotion",
        "missing_slots": [],
        "ask_prompts": [],
        "payload": payload,
    }
