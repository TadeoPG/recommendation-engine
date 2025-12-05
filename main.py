# -------------------------------------------------------
# API DE RECOMENDACIÓN CON FASTAPI
# Microservicio para sistema de recomendación de revistas
# Puerto: 8000
# -------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import hashlib
import time

# ------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "impplacc_destinos")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
LSA_PATH = os.path.join(MODEL_DIR, "lsa_model.joblib")
DF_PATH = os.path.join(MODEL_DIR, "revistas_df.joblib")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity_matrix.joblib")
ID_INDEX_PATH = os.path.join(MODEL_DIR, "id_index.joblib")
DATA_HASH_PATH = os.path.join(MODEL_DIR, "data_hash.txt")


# ------------------------------------------
# MODELOS PYDANTIC
# ------------------------------------------
class RecommendationRequest(BaseModel):
    magazine_id: int
    top_k: int = 5


class RecommendationResponse(BaseModel):
    id: int
    title: str
    region: str
    cover_image_url: Optional[str]
    similaridad: float


class SystemStatus(BaseModel):
    status: str
    total_magazines: int
    model_loaded: bool
    last_training: Optional[str]


# ------------------------------------------
# FASTAPI APP
# ------------------------------------------
app = FastAPI(
    title="API de Recomendación de Revistas",
    description="Microservicio de ML para recomendaciones basadas en contenido",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales del modelo
df = None
vectorizer = None
lsa = None
similaridades = None
id_to_idx = None


# ------------------------------------------
# FUNCIONES DE UTILIDAD
# ------------------------------------------
def calcular_hash_datos(df):
    datos_unicos = df[["id", "title"]].to_string()
    return hashlib.md5(datos_unicos.encode()).hexdigest()


def entrenar_modelo():
    print("\n🧠 Entrenando modelo...")
    start_time = time.time()

    query = """
    SELECT 
        id,
        title,
        COALESCE(description, '')   AS description,
        COALESCE(keywords, '')      AS keywords,
        COALESCE(topics, '')        AS topics,
        COALESCE(region, '')        AS region,
        COALESCE(cover_image_url,'') AS cover_image_url
    FROM magazine;
    """

    df_local = pd.read_sql(query, engine).fillna("")
    df_local["texto_final"] = (
        df_local["title"]
        + " "
        + df_local["description"]
        + " "
        + df_local["keywords"]
        + " "
        + df_local["topics"]
        + " "
        + df_local["region"]
    )

    nltk.download("stopwords", quiet=True)
    spanish_stopwords = stopwords.words("spanish")

    n_registros = len(df_local)
    max_features = min(3000, n_registros * 10)

    vectorizer_local = TfidfVectorizer(
        stop_words=spanish_stopwords,
        max_features=max_features,
        min_df=1,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer_local.fit_transform(df_local["texto_final"])

    n_features_real = tfidf_matrix.shape[1]
    n_components = min(100, max(10, n_features_real - 1))

    lsa_local = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa_local.fit_transform(tfidf_matrix)

    similaridades_local = cosine_similarity(lsa_matrix)
    id_to_idx_local = {id_val: idx for idx, id_val in enumerate(df_local["id"])}

    data_hash = calcular_hash_datos(df_local)

    joblib.dump(vectorizer_local, VECTORIZER_PATH)
    joblib.dump(lsa_local, LSA_PATH)
    joblib.dump(df_local, DF_PATH)
    joblib.dump(similaridades_local, SIMILARITY_PATH)
    joblib.dump(id_to_idx_local, ID_INDEX_PATH)

    with open(DATA_HASH_PATH, "w") as f:
        f.write(data_hash)

    elapsed = time.time() - start_time
    print(f"✅ Modelo entrenado en {elapsed:.2f}s")

    return df_local, vectorizer_local, lsa_local, similaridades_local, id_to_idx_local


def necesita_reentrenamiento():
    archivos_necesarios = [
        VECTORIZER_PATH,
        LSA_PATH,
        DF_PATH,
        SIMILARITY_PATH,
        ID_INDEX_PATH,
        DATA_HASH_PATH,
    ]

    if not all(os.path.exists(f) for f in archivos_necesarios):
        return True

    df_guardado = joblib.load(DF_PATH)

    with open(DATA_HASH_PATH, "r") as f:
        hash_anterior = f.read().strip()

    query = """
    SELECT 
        id,
        title,
        COALESCE(description, '')   AS description,
        COALESCE(keywords, '')      AS keywords,
        COALESCE(topics, '')        AS topics,
        COALESCE(region, '')        AS region,
        COALESCE(cover_image_url,'') AS cover_image_url
    FROM magazine;
    """
    df_actual = pd.read_sql(query, engine).fillna("")

    hash_actual = calcular_hash_datos(df_actual[["id", "title"]])

    return hash_actual != hash_anterior


# ------------------------------------------
# STARTUP EVENT
# ------------------------------------------
@app.on_event("startup")
async def startup_event():
    global df, vectorizer, lsa, similaridades, id_to_idx

    print("🚀 Iniciando API de Recomendación...")

    if necesita_reentrenamiento():
        df, vectorizer, lsa, similaridades, id_to_idx = entrenar_modelo()
    else:
        print("💾 Cargando modelos existentes...")
        start_time = time.time()

        vectorizer = joblib.load(VECTORIZER_PATH)
        lsa = joblib.load(LSA_PATH)
        df = joblib.load(DF_PATH)
        similaridades = joblib.load(SIMILARITY_PATH)
        id_to_idx = joblib.load(ID_INDEX_PATH)

        elapsed = time.time() - start_time
        print(f"✅ Modelos cargados en {elapsed:.2f}s")

    print(f"📊 Total revistas: {len(df)}")


# ------------------------------------------
# ENDPOINTS
# ------------------------------------------
@app.get("/", response_model=SystemStatus)
async def root():
    """Estado del sistema"""
    last_training = None
    if os.path.exists(DATA_HASH_PATH):
        mtime = os.path.getmtime(DATA_HASH_PATH)
        last_training = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

    return {
        "status": "running",
        "total_magazines": len(df) if df is not None else 0,
        "model_loaded": df is not None,
        "last_training": last_training,
    }


@app.post("/recommend", response_model=List[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    """
    Obtener recomendaciones para una revista específica
    """
    if df is None or id_to_idx is None or similaridades is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    if request.magazine_id not in id_to_idx:
        raise HTTPException(status_code=404, detail="Revista no encontrada")

    idx = id_to_idx[request.magazine_id]
    sim_scores = list(enumerate(similaridades[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]
    top_similares = sim_scores[: request.top_k]

    recomendaciones = []
    for i, score in top_similares:
        revista = df.iloc[i]
        recomendaciones.append(
            {
                "id": int(revista["id"]),
                "title": str(revista["title"]),
                "region": str(revista["region"]),
                "cover_image_url": str(revista.get("cover_image_url", "")),
                "similaridad": round(float(score), 3),
            }
        )

    return recomendaciones


@app.post("/retrain")
async def force_retrain():
    """
    Forzar reentrenamiento del modelo (útil cuando se agregan revistas)
    """
    global df, vectorizer, lsa, similaridades, id_to_idx

    try:
        df, vectorizer, lsa, similaridades, id_to_idx = entrenar_modelo()
        return {
            "status": "success",
            "message": "Modelo reentrenado correctamente",
            "total_magazines": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# ------------------------------------------
# EJECUTAR (solo para desarrollo)
# ------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Eliminar en producción
    )
