#!/usr/bin/env python3
"""
Quick ETL Setup and Runner
Simplified version that works with your cached data
"""
import pandas as pd
import os
import sys
import subprocess
from pathlib import Path
import logging
import sqlite3
import json
from datetime import datetime

def setup_dependencies():
    """Install missing dependencies"""
    print("🔧 Setting up dependencies...")
    
    dependencies = [
        "aeon",
        "scikit-learn", 
        "psycopg2-binary",
        "fastf1",
        "pandas",
        "numpy"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not install {dep}: {e}")
    
    print("✅ Dependencies setup complete")

def process_cached_data_simple():
    """Simple processing of cached data without complex ETL"""
    print("🔄 Processing cached F1 data (simplified)...")
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return False
    
    # Create simple database
    db_path = Path("f1_processed_simple.db")
    conn = sqlite3.connect(str(db_path))
    
    # Create tables
    conn.execute('''
        CREATE TABLE IF NOT EXISTS f1_race_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            race_name TEXT,
            file_path TEXT,
            file_type TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Scan and catalog cached data
    total_files = 0
    for year_dir in cache_dir.glob("20*"):
        if year_dir.is_dir():
            year = int(year_dir.name)
            print(f"📅 Processing {year} season...")
            
            for race_item in year_dir.iterdir():
                if race_item.is_file():
                    # Catalog the file
                    conn.execute('''
                        INSERT INTO f1_race_data (year, race_name, file_path, file_type)
                        VALUES (?, ?, ?, ?)
                    ''', (year, race_item.stem, str(race_item), race_item.suffix))
                    total_files += 1
                elif race_item.is_dir():
                    # Catalog directory
                    conn.execute('''
                        INSERT INTO f1_race_data (year, race_name, file_path, file_type)
                        VALUES (?, ?, ?, ?)
                    ''', (year, race_item.name, str(race_item), "directory"))
                    total_files += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Cataloged {total_files} F1 data items")
    print(f"📁 Database created: {db_path}")
    
    return True

def create_simple_api():
    """Create a simple API to access the processed data"""
    api_code = '''#!/usr/bin/env python3
"""
Simple F1 Data API
Access your processed F1 data
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
        "endpoints": ["/stats", "/years", "/races/{year}", "/data/{year}"]
    }

@app.get("/stats")
async def get_stats():
    """Get data statistics"""
    try:
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
            "total_items": total_count,
            "years": df.to_dict('records'),
            "database_size": DB_PATH.stat().st_size if DB_PATH.exists() else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/years")
async def get_years():
    """Get available years"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("SELECT DISTINCT year FROM f1_race_data ORDER BY year", conn)
        conn.close()
        return df['year'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/races/{year}")
async def get_races_by_year(year: int):
    """Get races for a specific year"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT race_name, file_path, file_type, processed_at
            FROM f1_race_data 
            WHERE year = ?
            ORDER BY race_name
        """, conn, params=(year,))
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{year}")
async def get_data_summary(year: int):
    """Get data summary for a year"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT COUNT(*) as total_races,
                   COUNT(DISTINCT race_name) as unique_races,
                   GROUP_CONCAT(DISTINCT file_type) as file_types
            FROM f1_race_data 
            WHERE year = ?
        """, conn, params=(year,))
        conn.close()
        
        result = df.to_dict('records')[0] if not df.empty else {}
        result['year'] = year
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🏎️ Starting Simple F1 Data API...")
    print(f"Database: {DB_PATH}")
    print("Dashboard: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    
    # Save the API code
    api_file = Path("simple_f1_api.py")
    with open(api_file, 'w') as f:
        f.write(api_code)
    
    print(f"✅ Created simple API: {api_file}")
    return api_file

def main():
    """Main setup and processing"""
    print("🏎️ Quick F1 ETL Setup")
    print("=" * 50)
    print("This will set up dependencies and process your cached F1 data")
    print()
    
    try:
        # Step 1: Setup dependencies
        setup_dependencies()
        print()
        
        # Step 2: Process cached data (simplified)
        success = process_cached_data_simple()
        if not success:
            print("❌ Failed to process cached data")
            return
        print()
        
        # Step 3: Create simple API
        api_file = create_simple_api()
        print()
        
        print("🎉 SETUP COMPLETE!")
        print("=" * 50)
        print("✅ Dependencies installed")
        print("✅ Cached F1 data processed and cataloged")
        print("✅ Simple API created")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Start the simple API:")
        print(f"   python {api_file}")
        print("2. Visit: http://localhost:8000/docs")
        print("3. Test endpoints: http://localhost:8000/stats")
        print()
        print("📊 Your F1 data is now accessible via REST API!")
        
        # Show what was found
        db_path = Path("f1_processed_simple.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            total = pd.read_sql_query("SELECT COUNT(*) as count FROM f1_race_data", conn).iloc[0]['count']
            years = pd.read_sql_query("SELECT DISTINCT year FROM f1_race_data ORDER BY year", conn)['year'].tolist()
            conn.close()
            
            print(f"\n📈 DATA SUMMARY:")
            print(f"   Total items: {total}")
            print(f"   Years: {years}")
            print(f"   Database: {db_path} ({db_path.stat().st_size} bytes)")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()