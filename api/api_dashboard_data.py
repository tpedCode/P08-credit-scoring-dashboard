import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException

# ==================================================
# CONFIGURATION
# ==================================================
# On part de la racine du projet P08 afin que le chemin
# fonctionne quel que soit le dossier depuis lequel Uvicorn est lancé.
BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = BASE_DIR / "data" / "database" / "dashboard.db"

# ==================================================
# APPLICATION FASTAPI
# ==================================================
# Cette API sert d'intermédiaire entre Streamlit et SQLite.
# Métier : le dashboard ne manipule pas directement la base de données.
# Technique : FastAPI fournit des endpoints HTTP simples consommables par Streamlit.
app = FastAPI(
    title="P08 — API Données Dashboard",
    description="API permettant au dashboard Streamlit d'accéder aux données clients et aux données de référence.",
    version="1.0.0"
)

# ==================================================
# CONNEXION SQLITE
# ==================================================
def get_connection():
    """Ouvre une connexion SQLite avec accès aux colonnes par leur nom."""
    # Vérification explicite pour retourner une erreur compréhensible
    # plutôt qu'une erreur SQLite moins explicite.
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Base SQLite introuvable : {DATABASE_PATH}")
    connection = sqlite3.connect(DATABASE_PATH)
    # Permet d'accéder aux résultats sous la forme row["colonne"].
    connection.row_factory = sqlite3.Row
    return connection

# ==================================================
# HEALTH CHECK
# ==================================================
@app.get("/health")
def health_check():
    """Vérifie que l'API fonctionne et que la base SQLite est accessible."""
    connection = None
    try:
        connection = get_connection()
        # SELECT 1 permet de vérifier que la connexion à SQLite fonctionne
        # sans effectuer de traitement coûteux.
        connection.execute("SELECT 1").fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        # La connexion doit toujours être fermée pour éviter les ressources
        # ouvertes inutilement sur le serveur.
        if connection is not None:
            connection.close()

# ==================================================
# INFORMATIONS DATABASE
# ==================================================
@app.get("/database-info")
def database_info():
    """Retourne les tables présentes dans SQLite et leur nombre de lignes."""
    connection = None
    try:
        connection = get_connection()
        # sqlite_master contient les informations de structure de la base.
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        result = {}
        for table in tables:
            table_name = table["name"]
            # Le nom de table provient de SQLite et non de l'utilisateur.
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            result[table_name] = count
        return {"database": str(DATABASE_PATH), "tables": result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if connection is not None:
            connection.close()

# ==================================================
# CLIENT — PROFIL COMPLET
# ==================================================
@app.get("/clients/{client_id}")
def get_client(client_id: int):
    """
    Retourne toutes les données disponibles pour un client.
    Métier : permet au dashboard d'afficher le profil complet.
    Technique : le client est identifié par SK_ID_CURR et la requête est paramétrée.
    """
    connection = None
    try:
        connection = get_connection()
        row = connection.execute(
            "SELECT * FROM clients WHERE SK_ID_CURR = ?",
            (client_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Client {client_id} introuvable.")
        return dict(row)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if connection is not None:
            connection.close()

# ==================================================
# FEATURES CLIENT
# ==================================================
@app.get("/clients/{client_id}/features")
def get_client_features(client_id: int):
    """
    Retourne les variables d'un client sans SK_ID_CURR.
    Métier : fournit directement les données nécessaires au scoring.
    Technique : l'identifiant du client n'est pas une feature du modèle.
    """
    connection = None
    try:
        connection = get_connection()
        row = connection.execute(
            "SELECT * FROM clients WHERE SK_ID_CURR = ?",
            (client_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Client {client_id} introuvable.")
        data = dict(row)
        data.pop("SK_ID_CURR", None)
        return data
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if connection is not None:
            connection.close()

# ==================================================
# POPULATION
# ==================================================
@app.get("/population")
def get_population(features: str, limit: int = 5000):
    """
    Retourne un échantillon aléatoire de clients avec les variables demandées.
    Métier : permet au dashboard d'analyser une population de clients.
    Technique : seules les colonnes demandées sont récupérées pour limiter les données transférées.
    """
    # Limiter la taille protège l'API contre des requêtes trop volumineuses.
    if limit < 1 or limit > 50000:
        raise HTTPException(status_code=400, detail="Le paramètre limit doit être compris entre 1 et 50000.")
    requested_features = [feature.strip() for feature in features.split(",") if feature.strip()]
    if not requested_features:
        raise HTTPException(status_code=400, detail="Aucune variable demandée.")
    connection = None
    try:
        connection = get_connection()
        # On récupère les colonnes réellement présentes avant de construire la requête SQL.
        # Cela évite d'exécuter une requête avec une colonne inexistante.
        columns = connection.execute("PRAGMA table_info(clients)").fetchall()
        available_columns = {column["name"] for column in columns}
        invalid_features = [feature for feature in requested_features if feature not in available_columns]
        if invalid_features:
            raise HTTPException(status_code=400, detail=f"Variables inconnues : {invalid_features}")
        # Les noms de colonnes sont validés ci-dessus avant d'être intégrés à la requête.
        selected_columns = ", ".join(f'"{feature}"' for feature in requested_features)
        query = f"SELECT {selected_columns} FROM clients ORDER BY RANDOM() LIMIT ?"
        rows = connection.execute(query, (limit,)).fetchall()
        return {
            "count": len(rows),
            "features": requested_features,
            "clients": [dict(row) for row in rows]
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if connection is not None:
            connection.close()

# ==================================================
# IMPORTANCE GLOBALE DES FEATURES
# ==================================================
@app.get("/feature-importance")
def get_feature_importance():
    """
    Retourne l'importance globale des variables du modèle.
    Métier : permet au dashboard d'expliquer quelles variables contribuent
    le plus globalement aux décisions du modèle.
    Technique : les importances sont déjà calculées et stockées dans SQLite.
    """
    connection = None
    try:
        connection = get_connection()
        rows = connection.execute(
            "SELECT feature, importance FROM global_feature_importance ORDER BY importance DESC"
        ).fetchall()
        return {
            "count": len(rows),
            "features": [dict(row) for row in rows]
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if connection is not None:
            connection.close()