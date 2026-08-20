import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "dashboard.db"
)


# ==================================================
# APPLICATION
# ==================================================

app = FastAPI(
    title="P08 — API Données Dashboard",
    description=(
        "API permettant au dashboard Streamlit "
        "d'accéder aux données clients et aux "
        "données de référence."
    ),
    version="1.0.0"
)


# ==================================================
# CONNEXION SQLITE
# ==================================================

def get_connection():
    """
    Ouvre une connexion à la base SQLite.
    """

    if not DATABASE_PATH.exists():

        raise FileNotFoundError(
            f"Base SQLite introuvable : {DATABASE_PATH}"
        )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():
    """
    Vérifie que l'API fonctionne et que
    la base SQLite est accessible.
    """

    connection = None

    try:

        connection = get_connection()

        connection.execute(
            "SELECT 1"
        ).fetchone()

        return {
            "status": "healthy",
            "database": "connected"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if connection is not None:
            connection.close()


# ==================================================
# INFORMATIONS DATABASE
# ==================================================

@app.get("/database-info")
def database_info():
    """
    Retourne les tables présentes dans la base
    ainsi que leur nombre de lignes.
    """

    connection = None

    try:

        connection = get_connection()

        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        result = {}

        for table in tables:

            table_name = table["name"]

            count = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM "{table_name}"
                """
            ).fetchone()[0]

            result[table_name] = count

        return {
            "database": str(DATABASE_PATH),
            "tables": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if connection is not None:
            connection.close()


# ==================================================
# CLIENT — PROFIL COMPLET
# ==================================================

@app.get("/clients/{client_id}")
def get_client(client_id: int):
    """
    Retourne toutes les données disponibles
    pour un client donné.

    Utilisé par le dashboard pour construire
    le profil client et envoyer les features
    à l'API Scoring.
    """

    connection = None

    try:

        connection = get_connection()

        row = connection.execute(
            """
            SELECT *
            FROM clients
            WHERE SK_ID_CURR = ?
            """,
            (client_id,)
        ).fetchone()

        if row is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Client {client_id} "
                    "introuvable."
                )
            )

        return dict(row)

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if connection is not None:
            connection.close()


# ==================================================
# FEATURES CLIENT
# ==================================================

@app.get("/clients/{client_id}/features")
def get_client_features(client_id: int):
    """
    Retourne uniquement les variables utilisées
    comme features du modèle.

    L'identifiant SK_ID_CURR est exclu.
    """

    connection = None

    try:

        connection = get_connection()

        row = connection.execute(
            """
            SELECT *
            FROM clients
            WHERE SK_ID_CURR = ?
            """,
            (client_id,)
        ).fetchone()

        if row is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Client {client_id} "
                    "introuvable."
                )
            )

        data = dict(row)

        data.pop(
            "SK_ID_CURR",
            None
        )

        return data

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if connection is not None:
            connection.close()


# ==================================================
# POPULATION
# ==================================================

@app.get("/population")
def get_population(
    features: str,
    limit: int = 5000
):

    if limit < 1 or limit > 50000:
        raise HTTPException(
            status_code=400,
            detail=(
                "Le paramètre limit doit "
                "être compris entre 1 et 50000."
            )
        )

    requested_features = [
        feature.strip()
        for feature in features.split(",")
        if feature.strip()
    ]

    if not requested_features:
        raise HTTPException(
            status_code=400,
            detail="Aucune variable demandée."
        )

    try:

        connection = get_connection()

        # ------------------------------------------
        # Vérification des colonnes disponibles
        # ------------------------------------------

        columns = connection.execute(
            "PRAGMA table_info(clients)"
        ).fetchall()

        available_columns = {
            column["name"]
            for column in columns
        }

        invalid_features = [
            feature
            for feature in requested_features
            if feature not in available_columns
        ]

        if invalid_features:

            connection.close()

            raise HTTPException(
                status_code=400,
                detail=(
                    "Variables inconnues : "
                    f"{invalid_features}"
                )
            )

        # ------------------------------------------
        # Construction du SELECT
        # ------------------------------------------

        selected_columns = ", ".join(
            f'"{feature}"'
            for feature in requested_features
        )

        query = f"""
            SELECT {selected_columns}
            FROM clients
            ORDER BY RANDOM()
            LIMIT ?
        """

        rows = connection.execute(
            query,
            (limit,)
        ).fetchall()

        connection.close()

        return {
            "count": len(rows),
            "features": requested_features,
            "clients": [
                dict(row)
                for row in rows
            ]
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==================================================
# IMPORTANCE GLOBALE DES FEATURES
# ==================================================

@app.get("/feature-importance")
def get_feature_importance():
    """
    Retourne l'importance globale des variables
    du modèle.
    """

    connection = None

    try:

        connection = get_connection()

        rows = connection.execute(
            """
            SELECT
                feature,
                importance
            FROM global_feature_importance
            ORDER BY importance DESC
            """
        ).fetchall()

        return {
            "count": len(rows),
            "features": [
                dict(row)
                for row in rows
            ]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if connection is not None:
            connection.close()