import asyncio
import pandas as pd
from db_client import FastF1TimescaleDB

class ETLPipelineAdapter:
    def __init__(self, db_connection_string: str):
        self.db_client = FastF1TimescaleDB(db_connection_string)
    
    async def init(self):
        await self.db_client.init_pool()
    
    async def close(self):
        await self.db_client.close_pool()
    
    async def ingest_from_your_etl(self, session_data: dict, telemetry_df: pd.DataFrame):
        session_id = f"{session_data['year']}_{session_data['gp'].replace(' ', '_')}_{session_data['session_type']}"
        await self.db_client.create_session(
            session_id, session_data['gp'], session_data['session_type']
        )
        
        for driver_id in telemetry_df['driver_id'].unique():
            driver_data = telemetry_df[telemetry_df['driver_id'] == driver_id]
            records = driver_data.to_dict('records')
            await self.db_client.insert_telemetry_batch(session_id, str(driver_id), records)
        
        return True

# Example
async def main():
    telemetry_df = pd.DataFrame({
        'driver_id': ['VER', 'HAM'],
        'Speed': [250.0, 245.0],
        'RPM': [11000, 10800]
    })
    
    session_data = {
        'year': 2023,
        'gp': 'Spanish Grand Prix', 
        'session_type': 'Q'
    }
    
    # Integrate with TimescaleDB
    adapter = ETLPipelineAdapter("postgresql://racing_user:racing_password@localhost:5432/racing_telemetry")
    await adapter.init()
    await adapter.ingest_from_your_etl(session_data, telemetry_df)
    await adapter.close()
    print("[OK] ETL integration complete!")

if __name__ == "__main__":
    asyncio.run(main())
