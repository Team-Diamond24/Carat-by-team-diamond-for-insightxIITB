import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def migrate():
    csv_path = "upi_transactions_2024.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Database path from .env
    db_path = os.environ.get("SQLITE_DB_PATH", "carat.db")

    print(f"Connecting to SQLite database: {db_path}...")
    conn = sqlite3.connect(db_path)

    print("Reading CSV...")
    df = pd.read_csv(csv_path)

    # Clean column names to match the SQL schema
    df.columns = [col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for col in df.columns]
    
    # Specific mappings if generic cleaning isn't enough
    df = df.rename(columns={
        "amount_inr": "amount_inr",
        "transaction_id": "transaction_id"
    })

    print(f"Migrating {len(df)} rows to table 'upi_transactions_2024'...")
    try:
        df.to_sql("upi_transactions_2024", conn, if_exists="replace", index=False)
        print("Migration successful!")
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
