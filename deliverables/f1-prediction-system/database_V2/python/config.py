# database/python/config.py

import os
from pathlib import Path

class DatabaseConfig:
    """Configuration for database operations"""
    
    # Database settings
    DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # sqlite, postgresql, mysql
    DB_PATH = Path(__file__).parent.parent / 'data'
    DB_NAME = os.getenv('DB_NAME', 'f1_data.db')
    
    # For PostgreSQL/MySQL
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_USER = os.getenv('DB_USER', 'f1_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # Cache settings
    CACHE_RETENTION_DAYS = int(os.getenv('CACHE_RETENTION_DAYS', '7'))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '1000'))
    
    # ETL settings
    ETL_SCHEDULE = os.getenv('ETL_SCHEDULE', 'daily')  # daily, weekly, manual
    
    @classmethod
    def get_connection_string(cls):
        """Get database connection string based on type"""
        if cls.DB_TYPE == 'sqlite':
            return f"sqlite:///{cls.DB_PATH / cls.DB_NAME}"
        elif cls.DB_TYPE == 'postgresql':
            return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        elif cls.DB_TYPE == 'mysql':
            return f"mysql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        else:
            raise ValueError(f"Unsupported database type: {cls.DB_TYPE}")