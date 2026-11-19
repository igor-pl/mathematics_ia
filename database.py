import duckdb
from pathlib import Path


# Find the root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Database will live in the data/ folder
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "market.db"


# Connect to DuckDB
connection = duckdb.connect(str(DB_PATH))


# Create our companies table
connection.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY,
        ticker VARCHAR,
        name VARCHAR,
        wikipedia_title VARCHAR,
        wikidata_id VARCHAR
    )
""")


print(f"Database created at: {DB_PATH}")


connection.close()