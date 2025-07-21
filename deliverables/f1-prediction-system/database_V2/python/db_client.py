import asyncpg
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class FastF1TimescaleDB:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def init_pool(self):
        self.pool = await asyncpg.create_pool(self.connection_string)
    
    async def close_pool(self):
        if self.pool:
            await self.pool.close()
    
    async def create_session(self, session_id: str, race_name: str, 
                           session_type: str, track_name: str = None,
                           start_time: datetime = None, metadata: Dict = None) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT create_session($1, $2, $3, $4, $5, $6)",
                session_id, race_name, session_type, track_name,
                start_time or datetime.utcnow(), json.dumps(metadata or {})
            )
            return result
    
    async def insert_telemetry_batch(self, session_id: str, driver_id: str, 
                                   telemetry_data: List[Dict]) -> int:
        async with self.pool.acquire() as conn:
            telemetry_json = json.dumps(telemetry_data)
            result = await conn.fetchval(
                "SELECT insert_telemetry_batch($1, $2, $3::jsonb)",
                session_id, driver_id, telemetry_json
            )
            return result
    
    async def store_predictions(self, session_id: str, predictions: List[Dict],
                              model_version: str = "v1.0") -> int:
        async with self.pool.acquire() as conn:
            predictions_json = json.dumps(predictions)
            result = await conn.fetchval(
                "SELECT store_predictions($1, $2::jsonb, $3)",
                session_id, predictions_json, model_version
            )
            return result
