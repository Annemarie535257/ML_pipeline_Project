#!/usr/bin/env python3
"""
Plant Disease Classification System - Auto Starter
This script starts both the API server and Streamlit app automatically.
"""

import subprocess
import sys
import time
import os
import requests
import threading

def start_api_server():
    """Start the API server in background"""
    print("🚀 Starting API server...")
    try:
        # Start API server
        api_process = subprocess.Popen([sys.executable, "api_server.py"], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        print("⏳ Waiting for API server to start...")
        for i in range(10):  # Wait up to 10 seconds
            try:
                response = requests.get("http://127.0.0.1:5000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API server started successfully!")
                    return api_process
            except:
                time.sleep(1)
                print(f"⏳ Still waiting... ({i+1}/10)")
        
        print("❌ Failed to start API server")
        return None
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def start_streamlit():
    """Start the Streamlit app"""
    print("🌐 Starting Streamlit app...")
    try:
        # Change to UI directory and start Streamlit
        os.chdir("UI")
        streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlitapp.py"])
        print("✅ Streamlit app started!")
        print("🌐 Open your browser and go to: http://localhost:8501")
        return streamlit_process
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        return None

def main():
    print("🌱 Plant Disease Classification System")
    print("=" * 50)
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Cannot start Streamlit without API server")
        return
    
    # Start Streamlit app
    streamlit_process = start_streamlit()
    if not streamlit_process:
        print("❌ Failed to start Streamlit app")
        return
    
    print("\n🎉 Both services are running!")
    print("📱 Streamlit App: http://localhost:8501")
    print("🔌 API Server: http://localhost:5000")
    print("\nPress Ctrl+C to stop both services...")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main() 