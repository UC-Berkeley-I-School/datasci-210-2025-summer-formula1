#!/usr/bin/env python3
"""
Fixed Simple F1 Data API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
from pathlib import Path
import json
import uvicorn

app = FastAPI(title="F1 Data API - Simple", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path("f1_processed_simple.db")

@app.get("/")
async def root():
    return {
        "message": "🏎️ Simple F1 Data API",
        "status": "active",
        "database": str(DB_PATH),
        "database_exists": DB_PATH.exists(),
        "endpoints": ["/stats", "/years", "/races/{year}", "/data/{year}"]
    }

@app.get("/stats")
async def get_stats():
    """Get data statistics"""
    try:
        if not DB_PATH.exists():
            raise HTTPException(status_code=404, detail="Database not found")
            
        conn = sqlite3.connect(str(DB_PATH))
        
        # Get counts by year
        df = pd.read_sql_query("""
            SELECT year, COUNT(*) as count, 
                   GROUP_CONCAT(DISTINCT file_type) as file_types
            FROM f1_race_data 
            GROUP BY year 
            ORDER BY year
        """, conn)
        
        total_count = pd.read_sql_query("SELECT COUNT(*) as total FROM f1_race_data", conn).iloc[0]['total']
        
        conn.close()
        
        return {
            "total_items": int(total_count),
            "years": df.to_dict('records'),
            "database_size": int(DB_PATH.stat().st_size),
            "database_path": str(DB_PATH)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/years")
async def get_years():
    """Get available years"""
    try:
        if not DB_PATH.exists():
            raise HTTPException(status_code=404, detail="Database not found")
            
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("SELECT DISTINCT year FROM f1_race_data ORDER BY year", conn)
        conn.close()
        return df['year'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/races/{year}")
async def get_races_by_year(year: int):
    """Get races for a specific year"""
    try:
        if not DB_PATH.exists():
            raise HTTPException(status_code=404, detail="Database not found")
            
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT race_name, file_path, file_type, processed_at
            FROM f1_race_data 
            WHERE year = ?
            ORDER BY race_name
        """, conn, params=(year,))
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for year {year}")
            
        return df.to_dict('records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/data/{year}")
async def get_data_summary(year: int):
    """Get data summary for a year"""
    try:
        if not DB_PATH.exists():
            raise HTTPException(status_code=404, detail="Database not found")
            
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT COUNT(*) as total_races,
                   COUNT(DISTINCT race_name) as unique_races,
                   GROUP_CONCAT(DISTINCT file_type) as file_types
            FROM f1_race_data 
            WHERE year = ?
        """, conn, params=(year,))
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for year {year}")
        
        result = df.to_dict('records')[0]
        result['year'] = year
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected" if DB_PATH.exists() else "not_found",
        "timestamp": pd.Timestamp.now().isoformat()
    }

# For debugging - show sample data
@app.get("/debug/sample")
async def get_sample_data():
    """Get sample data for debugging"""
    try:
        if not DB_PATH.exists():
            return {"error": "Database not found", "path": str(DB_PATH)}
            
        conn = sqlite3.connect(str(DB_PATH))
        
        # Get table info
        tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        # Get sample data
        sample_df = pd.read_sql_query("SELECT * FROM f1_race_data LIMIT 5", conn)
        
        conn.close()
        
        return {
            "database_path": str(DB_PATH),
            "database_size": int(DB_PATH.stat().st_size),
            "tables": tables_df['name'].tolist(),
            "sample_data": sample_df.to_dict('records')
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🏎️ Starting Simple F1 Data API...")
    print(f"Database: {DB_PATH}")
    print(f"Database exists: {DB_PATH.exists()}")
    if DB_PATH.exists():
        print(f"Database size: {DB_PATH.stat().st_size} bytes")
    print("Dashboard: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("Stats: http://localhost:8000/stats")
    
    # Fixed uvicorn call
    uvicorn.run(
        "simple_f1_api:app",  # Import string format
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
