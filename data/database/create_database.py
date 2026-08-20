import sqlite3
from pathlib import Path

import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PARQUET_DIR = BASE_DIR / "data" / "processed"
DATABASE_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "dashboard.db"
)

CLIENTS_PATH = PARQUET_DIR / "dashboard_clients.parquet"
FEATURE_IMPORTANCE_PATH = (
    PARQUET_DIR / "global_feature_importance.parquet"
)


# ==================================================
# CHARGEMENT
# ==================================================

print("Chargement des données...")

clients = pd.read_parquet(CLIENTS_PATH)
feature_importance = pd.read_parquet(
    FEATURE_IMPORTANCE_PATH
)

print(
    f"Clients : {len(clients):,} lignes, "
    f"{len(clients.columns)} colonnes"
)

print(
    f"Feature importance : "
    f"{len(feature_importance):,} lignes"
)


# ==================================================
# CREATION DE LA BASE
# ==================================================

print("\nCréation de la base SQLite...")

connection = sqlite3.connect(DATABASE_PATH)


# --------------------------------------------------
# TABLE CLIENTS
# --------------------------------------------------

clients.to_sql(
    "clients",
    connection,
    if_exists="replace",
    index=False
)


# --------------------------------------------------
# TABLE FEATURE IMPORTANCE
# --------------------------------------------------

feature_importance.to_sql(
    "global_feature_importance",
    connection,
    if_exists="replace",
    index=False
)


# ==================================================
# INDEX
# ==================================================

print("Création des index...")

connection.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_clients_id
    ON clients(SK_ID_CURR)
    """
)


# ==================================================
# VERIFICATION
# ==================================================

tables = connection.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
    """
).fetchall()

print("\nTables présentes :")

for table in tables:
    print(f"  - {table[0]}")


client_count = connection.execute(
    "SELECT COUNT(*) FROM clients"
).fetchone()[0]

importance_count = connection.execute(
    "SELECT COUNT(*) FROM global_feature_importance"
).fetchone()[0]

connection.close()


print("\nBase créée avec succès.")
print(f"Chemin : {DATABASE_PATH}")
print(f"Clients : {client_count:,}")
print(
    f"Feature importance : "
    f"{importance_count:,}"
)