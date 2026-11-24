import os
import sys

# Ensure Flask directory is in python path
flask_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Flask")
sys.path.insert(0, flask_dir)

from app import app

if __name__ == "__main__":
    print("Starting Plant AI server at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
