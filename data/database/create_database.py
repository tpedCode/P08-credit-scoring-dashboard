import sqlite3
from pathlib import Path
import pandas as pd

# ==================================================
# CONFIGURATION
# ==================================================
# On remonte de deux niveaux depuis ce script pour retrouver la racine du projet.
# Cela permet d'avoir des chemins indépendants du dossier depuis lequel le script est lancé.
BASE_DIR = Path(__file__).resolve().parents[2]
PARQUET_DIR = BASE_DIR / "data" / "processed"
DATABASE_PATH = BASE_DIR / "data" / "database" / "dashboard.db"
CLIENTS_PATH = PARQUET_DIR / "dashboard_clients.parquet"
FEATURE_IMPORTANCE_PATH = PARQUET_DIR / "global_feature_importance.parquet"

# ==================================================
# VERIFICATION DES FICHIERS
# ==================================================
# Ces deux fichiers sont les sources nécessaires à la construction de la base.
# Le contrôle évite de créer une base vide ou incomplète si un fichier manque.
for path in [CLIENTS_PATH, FEATURE_IMPORTANCE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

# ==================================================
# CHARGEMENT DES DONNEES
# ==================================================
# Les fichiers Parquet contiennent les données préparées en amont du projet.
# - dashboard_clients : données nécessaires au profil et au scoring des clients.
# - global_feature_importance : importance globale des variables du modèle.
print("Chargement des données...")
clients = pd.read_parquet(CLIENTS_PATH)
feature_importance = pd.read_parquet(FEATURE_IMPORTANCE_PATH)
print(f"Clients : {len(clients):,} lignes, {len(clients.columns)} colonnes")
print(f"Feature importance : {len(feature_importance):,} lignes")

# ==================================================
# CREATION DE LA BASE SQLITE
# ==================================================
# SQLite fournit une base légère et locale permettant à l'API Données
# d'interroger rapidement les données sans recharger les fichiers Parquet.
print("\nCréation de la base SQLite...")
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
connection = sqlite3.connect(DATABASE_PATH)

# ==================================================
# TABLE CLIENTS
# ==================================================
# On transforme le DataFrame clients en table SQL.
# "replace" garantit que la base est reconstruite à partir des dernières données
# à chaque exécution du script.
clients.to_sql("clients", connection, if_exists="replace", index=False)

# ==================================================
# TABLE IMPORTANCE DES FEATURES
# ==================================================
# Cette table permet au dashboard d'afficher les variables les plus importantes
# du modèle sans avoir à recalculer les importances à chaque consultation.
feature_importance.to_sql("global_feature_importance", connection, if_exists="replace", index=False)

# ==================================================
# INDEX
# ==================================================
# L'index sur SK_ID_CURR accélère fortement les recherches d'un client précis.
# C'est important car l'API interroge très fréquemment la base avec cet identifiant.
print("Création des index...")
connection.execute("CREATE INDEX IF NOT EXISTS idx_clients_id ON clients(SK_ID_CURR)")
connection.commit()

# ==================================================
# VERIFICATION DE LA BASE
# ==================================================
# On vérifie que les deux tables attendues existent bien après leur création.
tables = connection.execute(
    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
).fetchall()
print("\nTables présentes :")
for table in tables:
    print(f"  - {table[0]}")

# Vérification métier : le nombre de clients et de lignes d'importance doit
# correspondre aux données chargées depuis les fichiers Parquet.
client_count = connection.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
importance_count = connection.execute("SELECT COUNT(*) FROM global_feature_importance").fetchone()[0]

# ==================================================
# FERMETURE ET RESULTAT
# ==================================================
# La connexion est fermée proprement afin de libérer la ressource SQLite.
connection.close()
print("\nBase créée avec succès.")
print(f"Chemin : {DATABASE_PATH}")
print(f"Clients : {client_count:,}")
print(f"Feature importance : {importance_count:,}")