import os
import subprocess
import sys

if __name__ == "__main__":
    ui_script = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    print("Starting Streamlit Application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_script])
