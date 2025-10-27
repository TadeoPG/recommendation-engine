# -------------------------------------------------------
# SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO (TF-IDF + LSA)
# CON PERSISTENCIA DE MODELO Y DETECCIÓN AUTOMÁTICA DE NUEVAS REVISTAS
# Autor: Tadeo Manuel Portillo Guzmán
# Proyecto: Plataforma "Destinos Turismo"
# -------------------------------------------------------

import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# ------------------------------------------
# 1️⃣ CONFIGURACIÓN DE CONEXIÓN A LA BD
# ------------------------------------------
DB_USER = "root"  # <-- coloca tu usuario MySQL
DB_PASS = "root"  # <-- coloca tu contraseña
DB_HOST = "localhost"
DB_NAME = "impplacc_destinos"

# Crear conexión SQLAlchemy
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# Carpeta donde se guardarán los modelos
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Rutas de los archivos de modelo
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
LSA_PATH = os.path.join(MODEL_DIR, "lsa_model.joblib")
DF_PATH = os.path.join(MODEL_DIR, "revistas_df.joblib")


# ------------------------------------------
# 2️⃣ FUNCIÓN DE ENTRENAMIENTO
# ------------------------------------------
def entrenar_modelo():
    print("\n🧠 Entrenando modelo TF-IDF + LSA...")

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

    # Combinar campos semánticos en un solo texto
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

    # Descargar stopwords en español
    nltk.download("stopwords", quiet=True)
    spanish_stopwords = stopwords.words("spanish")

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["texto_final"])

    # LSA (reducción de dimensionalidad)
    lsa = TruncatedSVD(n_components=100, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # Guardar los artefactos
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(lsa, LSA_PATH)
    joblib.dump(df, DF_PATH)

    print("✅ Modelos entrenados y guardados correctamente.\n")

    return df, vectorizer, lsa, lsa_matrix


# ------------------------------------------
# 3️⃣ DETECCIÓN AUTOMÁTICA DE NUEVOS REGISTROS
# ------------------------------------------
def necesita_reentrenamiento():
    """
    Retorna True si hay cambios en la cantidad de revistas
    o si no existen los modelos guardados.
    """
    # Si no existen archivos, entrenar desde cero
    if not (
        os.path.exists(VECTORIZER_PATH)
        and os.path.exists(LSA_PATH)
        and os.path.exists(DF_PATH)
    ):
        print("⚙️ Modelos no encontrados. Se entrenará por primera vez.")
        return True

    # Cargar DataFrame guardado del modelo anterior
    df_guardado = joblib.load(DF_PATH)
    num_guardado = len(df_guardado)

    # Consultar cantidad actual de revistas en BD
    query = "SELECT COUNT(*) AS total FROM magazine;"
    total_actual = pd.read_sql(query, engine)["total"][0]

    # Comparar
    if total_actual != num_guardado:
        print(
            f"📈 Cambios detectados en la tabla magazine "
            f"({total_actual} actuales vs {num_guardado} previos)."
        )
        return True

    print("✅ No se detectaron cambios en la tabla magazine.")
    return False


# ------------------------------------------
# 4️⃣ CARGA O REENTRENAMIENTO AUTOMÁTICO
# ------------------------------------------
if necesita_reentrenamiento():
    df, vectorizer, lsa, lsa_matrix = entrenar_modelo()
else:
    print("💾 Cargando modelos existentes...\n")
    vectorizer = joblib.load(VECTORIZER_PATH)
    lsa = joblib.load(LSA_PATH)
    df = joblib.load(DF_PATH)

    nltk.download("stopwords", quiet=True)
    spanish_stopwords = stopwords.words("spanish")
    tfidf_matrix = vectorizer.transform(df["texto_final"])
    lsa_matrix = lsa.transform(tfidf_matrix)
    print("✅ Modelos cargados correctamente.\n")

# ------------------------------------------
# 5️⃣ MATRIZ DE SIMILITUD
# ------------------------------------------
similaridades = cosine_similarity(lsa_matrix)


# ------------------------------------------
# 6️⃣ FUNCIÓN DE RECOMENDACIÓN
# ------------------------------------------
def recomendar_revistas(id_revista, top_k=5):
    """
    Retorna las revistas más similares a una revista dada.
    """
    if id_revista not in df["id"].values:
        print("❌ El ID de revista no existe en la base de datos.")
        return None

    idx = df.index[df["id"] == id_revista][0]
    sim_scores = list(enumerate(similaridades[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Excluir la revista original
    sim_scores = [s for s in sim_scores if s[0] != idx]
    top_similares = sim_scores[:top_k]

    recomendaciones = df.iloc[[i for i, _ in top_similares]][["id", "title", "region"]]
    recomendaciones["similaridad"] = [round(s, 3) for _, s in top_similares]

    return recomendaciones


# ------------------------------------------
# 7️⃣ EJEMPLO DE USO
# ------------------------------------------
if __name__ == "__main__":
    print(
        "🔍 SISTEMA DE RECOMENDACIÓN DE REVISTAS (con Joblib + detección de nuevos registros)\n"
    )
    try:
        ejemplo_id = int(input("Ingrese el ID de la revista para recomendar: "))
        resultado = recomendar_revistas(ejemplo_id, top_k=5)

        if resultado is not None:
            print("\n🧭 Revistas más similares:\n")
            print(resultado.to_string(index=False))
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
