import logging
from app.core.config import settings
from .graph import t2s_app
from .state import SQLState

logger = logging.getLogger(__name__)

def call_sql_generator(message, conn_str, schema_info):
    state = SQLState(question=message, conn_str=conn_str, schema_info=schema_info)
    response = t2s_app.invoke(state)
    
    return response