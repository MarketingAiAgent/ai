"""
데이터 변환 및 정규화를 위한 공통 유틸리티 함수들
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from pandas.api.types import is_datetime64_any_dtype
import logging

logger = logging.getLogger(__name__)

def normalize_table_data(df: pd.DataFrame, max_rows: int = 20) -> Dict[str, Any]:
    """
    pandas DataFrame을 표준 테이블 JSON으로 변환
    
    Args:
        df: 변환할 DataFrame
        max_rows: 미리보기로 포함할 최대 행 수
        
    Returns:
        표준화된 테이블 데이터 구조
    """
    try:
        # 멀티컬럼 방어
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["__".join(map(str, c)).strip() for c in df.columns.values]

        # 날짜 컬럼 ISO 문자열화
        for col in df.columns:
            try:
                if is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                logger.warning(f"날짜 변환 실패 (컬럼: {col}): {e}")

        # NaN/NaT -> None (JSON null)
        df = df.where(pd.notnull(df), None)

        # 미리보기만 담고 전체 행수는 별도 기입
        preview = df.head(max_rows)

        return {
            "rows": preview.to_dict(orient="records"),     # [{col:val}, ...]
            "columns": [str(c) for c in preview.columns],  # 열 이름
            "row_count": int(df.shape[0]),                 # 전체 행 수
        }
        
    except Exception as e:
        logger.error(f"테이블 데이터 정규화 실패: {e}")
        return create_empty_table(str(e))

def create_empty_table(error: Optional[str] = None) -> Dict[str, Any]:
    """
    빈 테이블 생성
    
    Args:
        error: 에러 메시지 (선택사항)
        
    Returns:
        빈 테이블 데이터 구조
    """
    result = {
        "rows": [], 
        "columns": [], 
        "row_count": 0
    }
    if error:
        result["error"] = error
    return result

def ensure_table_payload(table: Any) -> Dict[str, Any]:
    """
    임의의 테이블 형태 데이터를 표준 {rows, columns, row_count} 형태로 정규화
    
    Args:
        table: 정규화할 테이블 데이터
        
    Returns:
        정규화된 테이블 데이터
    """
    if not isinstance(table, dict):
        return create_empty_table("Invalid table format")
    
    rows = table.get("rows", [])
    cols = table.get("columns", [])
    
    # 컬럼명이 없으면 첫 번째 행에서 추출
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and not cols:
        cols = list(rows[0].keys())
    
    return {
        "rows": rows if isinstance(rows, list) else [],
        "columns": cols if isinstance(cols, list) else [],
        "row_count": int(table.get("row_count", len(rows) or 0)),
        "error": table.get("error")  # 에러 정보 유지
    }

def normalize_web_data(web_results: List[Dict[str, Any]], max_items: int = 5) -> Dict[str, Any]:
    """
    Web 검색 결과를 표준화된 형태로 변환
    
    Args:
        web_results: Web 검색 결과 리스트
        max_items: 포함할 최대 아이템 수
        
    Returns:
        표준화된 Web 데이터
    """
    try:
        items = []
        for result in web_results[:max_items]:
            title = result.get("title", "").strip()
            content = result.get("content", "").strip()
            url = result.get("url", "").strip()
            
            # 제목이 없으면 내용에서 추출
            name = title or (content[:40] + "…" if content else "untitled")
            # 요약 내용
            signal = content[:180] + ("…" if len(content) > 180 else "")
            
            items.append({
                "name": name,
                "signal": signal,
                "source": url
            })
        
        return {
            "items": items,
            "total_count": len(web_results),
            "display_count": len(items)
        }
        
    except Exception as e:
        logger.error(f"Web 데이터 정규화 실패: {e}")
        return {
            "items": [],
            "total_count": 0,
            "display_count": 0,
            "error": str(e)
        }
