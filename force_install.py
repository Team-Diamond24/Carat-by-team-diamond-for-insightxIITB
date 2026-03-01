import sys
import subprocess
import os

print(f"Executing with python: {sys.executable}")
print("Installing slowapi...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "slowapi"])
    print("Pip install returned success code.")
except subprocess.CalledProcessError as e:
    print(f"Pip install failed with error code: {e.returncode}")

print("Verifying import...")
try:
    import slowapi
    print("SUCCESS: slowapi is successfully imported.")
except ImportError as e:
    print(f"FAILED IMPORT: {e}")
