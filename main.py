# -------------------------------------------------------
# SISTEMA DE RECOMENDACIÓN OPTIMIZADO (TF-IDF + LSA)
# CON CACHÉ INTELIGENTE Y DETECCIÓN AUTOMÁTICA DE CAMBIOS
# Autor: Tadeo Manuel Portillo Guzmán (Optimizado)
# Proyecto: Plataforma "Destinos Turismo"
# -------------------------------------------------------

import os
import pandas as pd
import joblib
import hashlib
import time
from functools import lru_cache
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------
# 1️⃣ CONFIGURACIÓN DE CONEXIÓN A LA BD
# ------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "impplacc_destinos")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# Carpeta donde se guardarán los modelos
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Rutas de los archivos de modelo (OPTIMIZADO: agregamos similarity y hash)
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
LSA_PATH = os.path.join(MODEL_DIR, "lsa_model.joblib")
DF_PATH = os.path.join(MODEL_DIR, "revistas_df.joblib")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity_matrix.joblib")  # ⭐ NUEVO
ID_INDEX_PATH = os.path.join(MODEL_DIR, "id_index.joblib")  # ⭐ NUEVO
DATA_HASH_PATH = os.path.join(MODEL_DIR, "data_hash.txt")  # ⭐ NUEVO


# ------------------------------------------
# 2️⃣ FUNCIÓN PARA CALCULAR HASH DE LOS DATOS
# ------------------------------------------
def calcular_hash_datos(df):
    """
    Genera un hash MD5 de los IDs y títulos para detectar cambios.
    Más robusto que solo contar registros.
    """
    datos_unicos = df[["id", "title"]].to_string()
    return hashlib.md5(datos_unicos.encode()).hexdigest()


# ------------------------------------------
# 3️⃣ FUNCIÓN DE ENTRENAMIENTO OPTIMIZADA
# ------------------------------------------
def entrenar_modelo():
    print("\n🧠 Entrenando modelo TF-IDF + LSA optimizado...")
    start_time = time.time()

    query = """
    SELECT 
        id,
        title,
        COALESCE(description, '') AS description,
        COALESCE(keywords, '') AS keywords,
        COALESCE(topics, '') AS topics,
        COALESCE(region, '') AS region
    FROM magazine;
    """

    df = pd.read_sql(query, engine).fillna("")

    # Combinar campos semánticos
    df["texto_final"] = (
        df["title"]
        + " "
        + df["description"]
        + " "
        + df["keywords"]
        + " "
        + df["topics"]
        + " "
        + df["region"]
    )

    # Descargar stopwords
    nltk.download("stopwords", quiet=True)
    spanish_stopwords = stopwords.words("spanish")

    # ⭐ OPTIMIZACIÓN: Ajustar hiperparámetros dinámicamente
    n_registros = len(df)
    max_features = min(3000, n_registros * 10)

    print(f"📊 Dataset: {n_registros} revistas")

    # TF-IDF (sin min_df para datasets pequeños)
    vectorizer = TfidfVectorizer(
        stop_words=spanish_stopwords,
        max_features=max_features,
        min_df=1,  # ⭐ Ajustado para datasets pequeños
        max_df=0.95,  # ⭐ Más permisivo
    )
    tfidf_matrix = vectorizer.fit_transform(df["texto_final"])

    # ⭐ CRÍTICO: n_components debe ser menor que n_features
    n_features_real = tfidf_matrix.shape[1]
    n_components = min(100, max(10, n_features_real - 1))  # Siempre menor que features

    print(f"🔧 Features extraídos: {n_features_real}")
    print(f"🔧 Componentes LSA: {n_components}")

    # LSA
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # ⭐ OPTIMIZACIÓN: Calcular y guardar matriz de similitud
    print("🔄 Calculando matriz de similitud...")
    similaridades = cosine_similarity(lsa_matrix)

    # ⭐ OPTIMIZACIÓN: Crear índice de IDs para búsqueda O(1)
    id_to_idx = {id_val: idx for idx, id_val in enumerate(df["id"])}

    # Calcular hash de los datos
    data_hash = calcular_hash_datos(df)

    # Guardar todos los artefactos
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(lsa, LSA_PATH)
    joblib.dump(df, DF_PATH)
    joblib.dump(similaridades, SIMILARITY_PATH)  # ⭐ NUEVO
    joblib.dump(id_to_idx, ID_INDEX_PATH)  # ⭐ NUEVO

    with open(DATA_HASH_PATH, "w") as f:
        f.write(data_hash)

    elapsed = time.time() - start_time
    print(f"✅ Modelos entrenados y guardados en {elapsed:.2f}s\n")

    return df, vectorizer, lsa, similaridades, id_to_idx


# ------------------------------------------
# 4️⃣ DETECCIÓN INTELIGENTE DE CAMBIOS
# ------------------------------------------
def necesita_reentrenamiento():
    """
    Retorna True si:
    - No existen los modelos
    - El hash de los datos cambió (nuevos/modificados/eliminados registros)
    """
    # Verificar existencia de archivos
    archivos_necesarios = [
        VECTORIZER_PATH,
        LSA_PATH,
        DF_PATH,
        SIMILARITY_PATH,
        ID_INDEX_PATH,
        DATA_HASH_PATH,
    ]

    if not all(os.path.exists(f) for f in archivos_necesarios):
        print("⚙️ Modelos no encontrados. Se entrenará por primera vez.")
        return True

    # Cargar DataFrame guardado
    df_guardado = joblib.load(DF_PATH)

    # Cargar hash anterior
    with open(DATA_HASH_PATH, "r") as f:
        hash_anterior = f.read().strip()

    # Consultar datos actuales de la BD
    query = """
    SELECT 
        id,
        title,
        COALESCE(description, '') AS description,
        COALESCE(keywords, '') AS keywords,
        COALESCE(topics, '') AS topics,
        COALESCE(region, '') AS region
    FROM magazine;
    """
    df_actual = pd.read_sql(query, engine).fillna("")

    # Calcular hash actual
    hash_actual = calcular_hash_datos(df_actual[["id", "title"]])

    # Comparar hashes
    if hash_actual != hash_anterior:
        num_anterior = len(df_guardado)
        num_actual = len(df_actual)
        print(f"📈 Cambios detectados en la tabla magazine")
        print(f"   Registros: {num_anterior} → {num_actual}")
        print(f"   Hash anterior: {hash_anterior[:8]}...")
        print(f"   Hash actual: {hash_actual[:8]}...")
        return True

    print("✅ No se detectaron cambios en la tabla magazine.")
    return False


# ------------------------------------------
# 5️⃣ FUNCIÓN DE RECOMENDACIÓN OPTIMIZADA
# ------------------------------------------
@lru_cache(maxsize=200)  # ⭐ CACHÉ para recomendaciones frecuentes
def recomendar_revistas(id_revista, top_k=5):
    """
    Retorna las revistas más similares a una revista dada.
    OPTIMIZADO con búsqueda O(1) y caché LRU.
    """
    # ⭐ OPTIMIZACIÓN: Búsqueda O(1) con hash map
    if id_revista not in id_to_idx:
        print("❌ El ID de revista no existe en la base de datos.")
        return None

    idx = id_to_idx[id_revista]

    # Obtener similitudes (ya precalculadas)
    sim_scores = list(enumerate(similaridades[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Excluir la revista original
    sim_scores = [s for s in sim_scores if s[0] != idx]
    top_similares = sim_scores[:top_k]

    recomendaciones = df.iloc[[i for i, _ in top_similares]][
        ["id", "title", "region"]
    ].copy()
    recomendaciones["similaridad"] = [round(s, 3) for _, s in top_similares]

    return recomendaciones


# ------------------------------------------
# 6️⃣ CARGA O REENTRENAMIENTO AUTOMÁTICO
# ------------------------------------------
if necesita_reentrenamiento():
    df, vectorizer, lsa, similaridades, id_to_idx = entrenar_modelo()
    # Limpiar caché de recomendaciones
    recomendar_revistas.cache_clear()
else:
    print("💾 Cargando modelos existentes...")
    start_time = time.time()

    vectorizer = joblib.load(VECTORIZER_PATH)
    lsa = joblib.load(LSA_PATH)
    df = joblib.load(DF_PATH)
    similaridades = joblib.load(SIMILARITY_PATH)  # ⭐ CARGA DIRECTA
    id_to_idx = joblib.load(ID_INDEX_PATH)  # ⭐ CARGA DIRECTA

    elapsed = time.time() - start_time
    print(f"✅ Modelos cargados en {elapsed:.2f}s\n")


# ------------------------------------------
# 7️⃣ FUNCIÓN DE ESTADÍSTICAS DEL SISTEMA
# ------------------------------------------
def mostrar_estadisticas():
    """Muestra información sobre el sistema y el modelo cargado."""
    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DEL SISTEMA DE RECOMENDACIÓN")
    print("=" * 60)
    print(f"Total de revistas: {len(df)}")
    print(f"Dimensiones TF-IDF: {vectorizer.max_features}")
    print(f"Componentes LSA: {lsa.n_components}")
    print(f"Tamaño matriz similitud: {similaridades.shape}")
    print(f"Caché de recomendaciones: {recomendar_revistas.cache_info()}")
    print("=" * 60 + "\n")


# ------------------------------------------
# 8️⃣ EJEMPLO DE USO
# ------------------------------------------
if __name__ == "__main__":
    mostrar_estadisticas()

    print("🔍 SISTEMA DE RECOMENDACIÓN DE REVISTAS (OPTIMIZADO)\n")

    try:
        ejemplo_id = int(input("Ingrese el ID de la revista para recomendar: "))

        start_time = time.time()
        resultado = recomendar_revistas(ejemplo_id, top_k=5)
        elapsed = time.time() - start_time

        if resultado is not None:
            print("\n🧭 Revistas más similares:\n")
            print(resultado.to_string(index=False))
            print(f"\n⏱️  Tiempo de respuesta: {elapsed*1000:.2f}ms")

            # Mostrar si vino del caché
            cache_info = recomendar_revistas.cache_info()
            if cache_info.hits > 0:
                print(f"💾 (Servido desde caché)")

    except ValueError:
        print("\n⚠️ Por favor ingrese un número válido.")
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
