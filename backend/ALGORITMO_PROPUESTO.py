# ==========================================
# IMPORTACIÓN DE BIBLIOTECAS
# ==========================================

import pickle
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# CARGA DE MODELOS SERIALIZADOS
# Todos estos archivos fueron generados por
# train.py y commiteados al repositorio.
# No se entrena nada aquí.
# ==========================================

MODELS_DIR = "models"

print(">>> Cargando modelos serializados...")

# Datos con clusters ya asignados (generados por train.py)
products  = pd.read_json(f"{MODELS_DIR}/products_with_clusters.json")
locations = pd.read_json(f"{MODELS_DIR}/locations_with_clusters.json")
ratings   = pd.read_json("data/ratings.json")
comments = pd.read_json("data/comments.json")

# Modelos sklearn
tfidf         = joblib.load(f"{MODELS_DIR}/tfidf.pkl")
similarity_df = joblib.load(f"{MODELS_DIR}/similarity_df.pkl")
lda           = joblib.load(f"{MODELS_DIR}/lda.pkl")
rating_matrix = joblib.load(f"{MODELS_DIR}/rating_matrix.pkl")
user_sim      = joblib.load(f"{MODELS_DIR}/user_sim.pkl")
rating_avg    = joblib.load(f"{MODELS_DIR}/rating_avg.pkl")
kmeans        = joblib.load(f"{MODELS_DIR}/kmeans.pkl")

# Tokenizer y modelo Keras
with open(f"{MODELS_DIR}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(f"{MODELS_DIR}/keras_model.keras")

print(">>> Modelos cargados correctamente.")


# ==========================================
# GEOLOCALIZACIÓN
# ==========================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat/2) * np.sin(dlat/2)
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon/2)
        * np.sin(dlon/2)
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def add_distance(user_id, lat_usuario=None, lon_usuario=None):

    if lat_usuario is not None and lon_usuario is not None:
        u_lat, u_lon = lat_usuario, lon_usuario
    else:
        user_loc = locations[locations['user_id'] == user_id]
        u_lat = user_loc['latitude'].values[0]
        u_lon = user_loc['longitude'].values[0]

    distances = [
        haversine(u_lat, u_lon, product['latitude'], product['longitude'])
        for _, product in products.iterrows()
    ]

    products['distance'] = distances

    return u_lat, u_lon


# ==========================================
# FUNCIONES DE NORMALIZACIÓN
# ==========================================

def normalize_distance(distances):
    max_d = distances.max()
    min_d = distances.min()
    return 1 - ((distances - min_d) / (max_d - min_d + 1e-6))


def normalize_rating(ratings):
    return (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-6)

# ==========================================
# CÁLCULO DE SCORES DE TEXTO (LDA + Keras)
# ==========================================
def compute_text_scores():
    """
    Calcula:
    - tfidf_score (ya existe indirectamente)
    - lda_score
    - sentiment_score (modelo Keras)
    """

    # ===== LDA SCORE =====
    try:
        lda_matrix = joblib.load(f"{MODELS_DIR}/lda_matrix.pkl")
        lda_score = pd.Series(lda_matrix.max(axis=1))  # tema dominante
        lda_score = normalize_rating(lda_score)
    except:
        lda_score = pd.Series(np.zeros(len(products)))

    # ===== SENTIMENT SCORE (Keras) =====
    try:
        comments_grouped = comments.groupby('item_id')['comment'].apply(' '.join)

        sequences = tokenizer.texts_to_sequences(comments_grouped)
        padded = pad_sequences(sequences, maxlen=100)

        preds = model.predict(padded, verbose=0).flatten()

        sentiment_score = pd.Series(preds, index=comments_grouped.index)
        sentiment_score = normalize_rating(sentiment_score)
    except Exception as e:
        print(f"Error en sentiment_score: {e}")
        sentiment_score = pd.Series(np.zeros(len(products)))

    return lda_score, sentiment_score

# ==========================================
# SISTEMA DE RECOMENDACIÓN HÍBRIDO
# ==========================================

def recommend(user_id, lat_manual=None, lon_manual=None, top_n=5):

    u_lat, u_lon = add_distance(user_id, lat_manual, lon_manual)

    rec = products.copy()

    df_rating_clean = rating_avg.drop_duplicates(subset=['item_id'])
    rec = rec.merge(df_rating_clean, on="item_id", how="left")
    rec["avg_rating"] = rec["avg_rating"].fillna(0)

    rec["geo_score"]    = normalize_distance(rec["distance"])
    rec["rating_score"] = normalize_rating(rec["avg_rating"])

    # --- TF-IDF (igual que antes) ---
    rec["tfidf_score"] = [
        similarity_df[item].mean() if item in similarity_df.columns else 0
        for item in rec["item_id"]
    ]

    # ===== LDA + SENTIMENT =====
    lda_score, sentiment_score = compute_text_scores()

    # Mapear scores a productos
    rec["lda_score"] = rec.index.map(lambda i: lda_score[i] if i < len(lda_score) else 0)

    rec["sentiment_score"] = rec["item_id"].map(
        sentiment_score
    ).fillna(0)

    # ===== NORMALIZAR POR SEGURIDAD =====
    rec["tfidf_score"]     = normalize_rating(pd.Series(rec["tfidf_score"]))
    rec["lda_score"]       = normalize_rating(pd.Series(rec["lda_score"]))
    rec["sentiment_score"] = normalize_rating(pd.Series(rec["sentiment_score"]))

    # --- Content score combinado ---
    rec["content_score"] = (
        0.5 * rec["tfidf_score"] +
        0.2 * rec["lda_score"] +
        0.3 * rec["sentiment_score"]
    )

    # --- Score final combinado ---
    rec["final_score"] = (
        0.4 * rec["geo_score"] +
        0.25 * rec["rating_score"] +
        0.35 * rec["content_score"]
    )

    rec["reason"] = rec.apply(generate_reason, axis=1)

    rec = rec.sort_values("final_score", ascending=False)

    rec["distance"] = rec["distance"].apply(lambda x: f"{x:.2f} km")

    return rec[["name", "distance", "avg_rating", "reason"]].head(top_n)


def generate_reason(row):
    reasons = []

    try:
        if float(row["geo_score"])     > 0.7: reasons.append("Cerca de tu ubicación")
        if float(row["rating_score"])  > 0.6: reasons.append("Muy bien calificado por usuarios")
        if float(row["content_score"]) > 0.5: reasons.append("Similar a productos populares")
    except Exception as e:
        print(f"Error en generate_reason: {e}")

    if not reasons:
        reasons.append("Recomendado por el sistema")

    return ", ".join(reasons)