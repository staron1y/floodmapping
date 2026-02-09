#!/usr/bin/env python3
"""
Setup script for Nairobi Flood Risk Dashboard
Run this script to install dependencies and launch the application
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🌊 Nairobi Flood Risk Dashboard Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this script from the webapp directory.")
        sys.exit(1)
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Launch the application
    print("\n🚀 Launching Nairobi Flood Risk Dashboard...")
    print("The app will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit. Please ensure it's installed correctly.")

if __name__ == "__main__":
    main()