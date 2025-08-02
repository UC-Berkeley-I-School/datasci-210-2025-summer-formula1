#!/usr/bin/env python3
"""Development server runner with hot-reload support."""

import os
import sys
from pathlib import Path

def run_dev():
    """Run Flask development server with hot-reload enabled."""
    # Set development environment variables
    os.environ['FLASK_APP'] = 'src.webapp.f1dev'
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Add src directory to Python path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    # Import and run Flask
    from flask import Flask
    from webapp.f1dev import app
    
    print("🚀 Starting Flask development server...")
    print("📍 Running on: http://localhost:5001")
    print("🔄 Hot-reload is enabled")
    print("⚡ Press CTRL+C to stop")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        use_reloader=True,
        use_debugger=True
    )

if __name__ == '__main__':
    run_dev()