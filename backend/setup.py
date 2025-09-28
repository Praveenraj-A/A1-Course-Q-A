import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""MONGO_URI=mongodb+srv://praveenrajacse2022_db_user:praveen_199@cluster0.v4wmkh4.mongodb.net/courseqa?retryWrites=true&w=majority&appName=Cluster0
MONGO_DB=courseqa
MONGO_COLLECTION=documents
GEMINI_API_KEY=AIzaSyCHBs-Cv7O5C7P9ZtrAONxgscCuNiGQOHI
PORT=8000
""")
        print("Created .env file")

if __name__ == "__main__":
    print("Setting up Course Q&A Backend...")
    create_env_file()
    install_requirements()
    print("Setup completed! Run 'python main.py' to start the server.")