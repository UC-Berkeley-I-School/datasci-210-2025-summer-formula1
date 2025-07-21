#!/usr/bin/env python3
"""
Simple debug script to fix the database path issue
"""

import sqlite3
from pathlib import Path
import os

def main():
    print("🏎️  F1 ETL Database Debug Script")
    print("=" * 50)
    
    # Show current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Create the directory structure
    print("\n1. Creating directory structure...")
    directories = [
        "database",
        "database/data", 
        "database/python",
        "logs"
    ]
    
    for dir_name in directories:
        dir_path = current_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_name}")
    
    # Test database creation
    print("\n2. Testing database creation...")
    db_path = current_dir / "database" / "data" / "f1_data.db"
    print(f"   Database path: {db_path}")
    
    try:
        # Create and test database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create a test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_connection (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                message TEXT
            )
        ''')
        
        # Insert test record
        cursor.execute("INSERT INTO test_connection (message) VALUES (?)", ("Database test successful",))
        conn.commit()
        
        # Read it back
        cursor.execute("SELECT * FROM test_connection")
        results = cursor.fetchall()
        
        print(f"   ✅ Database created successfully")
        print(f"   ✅ Test records: {len(results)}")
        
        # Clean up
        cursor.execute("DROP TABLE test_connection")
        conn.commit()
        conn.close()
        
        # Show file info
        print(f"   📁 File size: {db_path.stat().st_size} bytes")
        print(f"   📁 File exists: {db_path.exists()}")
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False
    
    # Test Python imports
    print("\n3. Testing Python imports...")
    required_modules = ["sqlite3", "pandas", "pathlib", "datetime", "logging"]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
    
    print("\n4. Directory structure created:")
    for item in sorted(current_dir.glob("*")):
        if item.is_dir() and item.name in ["database", "logs"]:
            print(f"   📁 {item.name}/")
            for subitem in sorted(item.rglob("*")):
                if subitem.is_dir():
                    relative = subitem.relative_to(current_dir)
                    print(f"      📁 {relative}/")
                elif subitem.suffix in [".db", ".log"]:
                    relative = subitem.relative_to(current_dir)
                    size = subitem.stat().st_size
                    print(f"      📄 {relative} ({size} bytes)")
    
    print("\n🎉 Debug complete! You can now run:")
    print("   python database/python/etl_cache_pipeline.py")
    
    return True

if __name__ == "__main__":
    main()