import sys
print(sys.executable)
try:
    import slowapi
    import pandas
    import plotly
    import openai
    import psycopg2
    from server import app
    print("ALL IMPORTS SUCCESSFUL")
except Exception as e:
    print(f"IMPORT FAILED: {e}")
