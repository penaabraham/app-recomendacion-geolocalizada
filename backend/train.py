# ==========================================
# train.py
# Ejecutar UNA SOLA VEZ en tu máquina local:
#   cd backend
#   python train.py
#
# Genera la carpeta models/ con todos los
# artefactos serializados. Commitea esa carpeta
# y Render nunca volverá a entrenar nada.
# ==========================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

import sys
print("INICIO", flush=True)
sys.stdout.flush()

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print(">>> [1/7] Cargando datos...")
users     = pd.read_json("data/users.json")
locations = pd.read_json("data/locations.json")
products  = pd.read_json("data/products.json")
ratings   = pd.read_json("data/ratings.json")
comments  = pd.read_json("data/comments.json")

# ── TF-IDF + Similitud coseno ──────────────────────────
print(">>> [2/7] Entrenando TF-IDF y calculando similitud coseno...")
comments_grouped = comments.groupby('item_id')['comment'].apply(' '.join).reset_index()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(comments_grouped['comment'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
item_ids_unique = comments_grouped["item_id"].tolist()
similarity_df = pd.DataFrame(cosine_sim, index=item_ids_unique, columns=item_ids_unique)

joblib.dump(tfidf,         f"{MODELS_DIR}/tfidf.pkl")
joblib.dump(similarity_df, f"{MODELS_DIR}/similarity_df.pkl")
print("    Guardado: tfidf.pkl, similarity_df.pkl")

# ── LDA ───────────────────────────────────────────────
print(">>> [3/7] Entrenando LDA...")
lda = LDA(n_components=5, random_state=42)
lda_matrix = lda.fit_transform(tfidf_matrix)

joblib.dump(lda,        f"{MODELS_DIR}/lda.pkl")
joblib.dump(lda_matrix, f"{MODELS_DIR}/lda_matrix.pkl")
print("    Guardado: lda.pkl, lda_matrix.pkl")

# ── Filtrado colaborativo ──────────────────────────────
print(">>> [4/7] Calculando similitud entre usuarios...")
rating_matrix = ratings.pivot_table(
    index='user_id', columns='item_id', values='rating'
).fillna(0)
user_sim = cosine_similarity(rating_matrix, rating_matrix)

rating_avg = ratings.groupby("item_id")["rating"].mean().reset_index()
rating_avg.rename(columns={"rating": "avg_rating"}, inplace=True)

joblib.dump(rating_matrix, f"{MODELS_DIR}/rating_matrix.pkl")
joblib.dump(user_sim,      f"{MODELS_DIR}/user_sim.pkl")
joblib.dump(rating_avg,    f"{MODELS_DIR}/rating_avg.pkl")
print("    Guardado: rating_matrix.pkl, user_sim.pkl, rating_avg.pkl")

# ── KMeans ────────────────────────────────────────────
print(">>> [5/7] Entrenando KMeans...")
kmeans = KMeans(n_clusters=3, random_state=42)
locations['cluster'] = kmeans.fit_predict(locations[['latitude', 'longitude']])
products['cluster']  = kmeans.predict(products[['latitude', 'longitude']])

joblib.dump(kmeans,    f"{MODELS_DIR}/kmeans.pkl")
products.to_json(      f"{MODELS_DIR}/products_with_clusters.json", orient='records')
locations.to_json(     f"{MODELS_DIR}/locations_with_clusters.json", orient='records')
print("    Guardado: kmeans.pkl, products_with_clusters.json, locations_with_clusters.json")

# ── Tokenizer + modelo Keras ──────────────────────────
print(">>> [6/7] Tokenizando y entrenando modelo Keras (esto tarda un poco)...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments['comment'])

sequences  = tokenizer.texts_to_sequences(comments['comment'])
word_index = tokenizer.word_index
data       = pad_sequences(sequences, maxlen=100)

labels = np.array([
    1 if r > 3 else 0
    for r in ratings['rating'].head(len(data))
])

model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=100),
    Flatten(),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

model.save(f"{MODELS_DIR}/keras_model.keras")

with open(f"{MODELS_DIR}/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("    Guardado: keras_model.keras, tokenizer.pkl")

# ── Verificación final ────────────────────────────────
print("\n>>> [7/7] Archivos generados en models/:")
for fname in sorted(os.listdir(MODELS_DIR)):
    size_kb = os.path.getsize(f"{MODELS_DIR}/{fname}") / 1024
    print(f"    {fname:<45} {size_kb:>8.1f} KB")

print("\n✅ Entrenamiento completo. Commitea la carpeta models/ y haz deploy.")