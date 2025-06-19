#!/usr/bin/env python3
"""
Startup script for the Employment Match FastAPI server
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI server"""
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Warning: .env file not found. Please create one with your GEMINI_API_KEY")
        print("Example: echo 'GEMINI_API_KEY=your-api-key-here' > .env")
    
    # Check if data files exist
    if not os.path.exists('data/esco_skills.json'):
        print("Warning: ESCO skills data not found. Run the setup first:")
        print("python convert_esco_to_json.py")
        print("python generate_embeddings.py")
    
    print("Starting Employment Match API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Web Interface: http://localhost:8000/ui")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 