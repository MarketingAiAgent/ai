import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from io import StringIO

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from app.core.config import settings

logger = logging.getLogger(__name__)

def upload_dataframe_to_blob(df: pd.DataFrame, filename: Optional[str] = None) -> Optional[str]:
    """
    DataFrame을 CSV 파일로 변환하여 Azure Blob Storage에 업로드하고 다운로드 URL을 반환합니다.
    
    Args:
        df: 업로드할 DataFrame
        filename: 파일명 (None이면 자동 생성)
    
    Returns:
        다운로드 URL 또는 None (실패 시)
    """
    try:
        # Azure Storage 연결
        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            logger.error("Azure Storage 연결 문자열이 설정되지 않았습니다.")
            return None
            
        blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        
        # 컨테이너 클라이언트
        container_client = blob_service_client.get_container_client(
            settings.AZURE_STORAGE_CONTAINER_NAME
        )
        
        # 파일명 생성
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"export_{timestamp}_{unique_id}.csv"
        
        # DataFrame을 CSV로 변환
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')  # UTF-8 BOM for Excel compatibility
        csv_content = csv_buffer.getvalue()
        
        # Blob에 업로드
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(csv_content, overwrite=True)
        
        logger.info(f"✅ DataFrame이 Blob Storage에 업로드되었습니다: {filename}")
        
        # SAS 토큰 생성 (24시간 유효)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=settings.AZURE_STORAGE_CONTAINER_NAME,
            blob_name=filename,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=24)
        )
        
        # 다운로드 URL 생성
        download_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{settings.AZURE_STORAGE_CONTAINER_NAME}/{filename}?{sas_token}"
        
        logger.info(f"✅ 다운로드 URL 생성 완료: {download_url[:100]}...")
        return download_url
        
    except Exception as e:
        logger.error(f"❌ Blob Storage 업로드 실패: {e}")
        return None

def upload_json_to_blob(data: dict, filename: Optional[str] = None) -> Optional[str]:
    """
    JSON 데이터를 Azure Blob Storage에 업로드하고 다운로드 URL을 반환합니다.
    
    Args:
        data: 업로드할 JSON 데이터
        filename: 파일명 (None이면 자동 생성)
    
    Returns:
        다운로드 URL 또는 None (실패 시)
    """
    try:
        # Azure Storage 연결
        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            logger.error("Azure Storage 연결 문자열이 설정되지 않았습니다.")
            return None
            
        blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        
        # 컨테이너 클라이언트
        container_client = blob_service_client.get_container_client(
            settings.AZURE_STORAGE_CONTAINER_NAME
        )
        
        # 파일명 생성
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"export_{timestamp}_{unique_id}.json"
        
        # JSON 데이터를 문자열로 변환
        json_content = json.dumps(data, ensure_ascii=False, indent=2)
        
        # Blob에 업로드
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(json_content, overwrite=True)
        
        logger.info(f"✅ JSON 데이터가 Blob Storage에 업로드되었습니다: {filename}")
        
        # SAS 토큰 생성 (24시간 유효)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=settings.AZURE_STORAGE_CONTAINER_NAME,
            blob_name=filename,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=24)
        )
        
        # 다운로드 URL 생성
        download_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{settings.AZURE_STORAGE_CONTAINER_NAME}/{filename}?{sas_token}"
        
        logger.info(f"✅ 다운로드 URL 생성 완료: {download_url[:100]}...")
        return download_url
        
    except Exception as e:
        logger.error(f"❌ Blob Storage 업로드 실패: {e}")
        return None
