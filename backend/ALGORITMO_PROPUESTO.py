# ==========================================
# IMPORTACIÓN DE BIBLIOTECAS
# ==========================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# CARGA DE DATOS
# ==========================================

users = pd.read_json("data/users.json")
locations = pd.read_json("data/locations.json")
products = pd.read_json("data/products.json")
ratings = pd.read_json("data/ratings.json")
comments = pd.read_json("data/comments.json")


# ==========================================
# PREPROCESAMIENTO DE DATOS
# ==========================================

# ------------------------------------------------
# FILTRADO BASADO EN CONTENIDO
# ------------------------------------------------

# Vectorización de comentarios usando TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(comments['comment'])

# Similitud coseno entre productos basados en comentarios
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Crear DataFrame de similitud usando item_id como índice
item_ids = comments["item_id"].tolist()
similarity_df = pd.DataFrame(cosine_sim, index=item_ids, columns=item_ids)


# ------------------------------------------------
# MODELADO DE TÓPICOS (LDA)
# ------------------------------------------------

lda = LDA(n_components=5, random_state=42)
lda_matrix = lda.fit_transform(tfidf_matrix)


# ------------------------------------------------
# FILTRADO COLABORATIVO
# ------------------------------------------------

# Crear matriz usuario-producto
rating_matrix = ratings.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0)

# Similitud entre usuarios
user_sim = cosine_similarity(rating_matrix, rating_matrix)


# ------------------------------------------------
# CÁLCULO DE RATING PROMEDIO POR PRODUCTO
# ------------------------------------------------

rating_avg = ratings.groupby("item_id")["rating"].mean().reset_index()
rating_avg.rename(columns={"rating": "avg_rating"}, inplace=True)


# ==========================================
# GEOLOCALIZACIÓN
# ==========================================

# Cálculo de distancia geográfica usando fórmula de Haversine

def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # Radio de la Tierra en km

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
    d = R * c

    return d


# ------------------------------------------------
# FUNCIÓN PARA AGREGAR DISTANCIAS A PRODUCTOS
# ------------------------------------------------

def add_distance(user_id, lat_usuario=None, lon_usuario=None):

    # Si el celular manda coordenadas, se usan directamente
    if lat_usuario is not None and lon_usuario is not None:

        u_lat, u_lon = lat_usuario, lon_usuario

    else:

        user_loc = locations[locations['user_id'] == user_id]

        u_lat = user_loc['latitude'].values[0]
        u_lon = user_loc['longitude'].values[0]

    distances = []

    for _, product in products.iterrows():

        dist = haversine(
            u_lat,
            u_lon,
            product['latitude'],
            product['longitude']
        )

        distances.append(dist)

    products['distance'] = distances

    return u_lat, u_lon


# Asignar distancias iniciales
# for user_id in locations['user_id']:
#     add_distance(user_id)


# ==========================================
# CLUSTERING DE UBICACIONES
# ==========================================

kmeans = KMeans(n_clusters=3, random_state=42)

locations['cluster'] = kmeans.fit_predict(
    locations[['latitude', 'longitude']]
)

products['cluster'] = kmeans.predict(
    products[['latitude', 'longitude']]
)


# ==========================================
# MODELO DE APRENDIZAJE PROFUNDO (NLP)
# ==========================================

# Tokenización de comentarios

tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments['comment'])

sequences = tokenizer.texts_to_sequences(comments['comment'])
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=100)


# Crear modelo de embeddings

model = Sequential()

model.add(
    Embedding(
        input_dim=len(word_index) + 1,
        output_dim=128,
        input_length=100
    )
)

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Crear etiquetas binarias (rating > 3 = positivo)

labels = np.array([
    1 if r > 3 else 0
    for r in ratings['rating'].head(len(data))
])


# Entrenar modelo

model.fit(
    data,
    labels,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# ==========================================
# FUNCIONES DE NORMALIZACIÓN
# ==========================================

def normalize_distance(distances):

    max_d = distances.max()
    min_d = distances.min()

    return 1 - ((distances - min_d) / (max_d - min_d + 1e-6))


def normalize_rating(ratings):

    return (
        (ratings - ratings.min())
        /
        (ratings.max() - ratings.min() + 1e-6)
    )


# ==========================================
# SISTEMA DE RECOMENDACIÓN HÍBRIDO
# ==========================================

def recommend(user_id, lat_manual=None, lon_manual=None, top_n=5):

    u_lat, u_lon = add_distance(user_id, lat_manual, lon_manual)

    rec = products.copy()

    rec = rec.merge(rating_avg, on="item_id", how="left")
    rec["avg_rating"].fillna(0, inplace=True)

    rec["geo_score"] = normalize_distance(rec["distance"])
    rec["rating_score"] = normalize_rating(rec["avg_rating"])

    content_scores = []

    for item in rec["item_id"]:
        if item in similarity_df.columns:
            score = similarity_df[item].mean()
        else:
            score = 0
        content_scores.append(score)

    rec["content_score"] = content_scores

    rec["final_score"] = (
        0.5 * rec["geo_score"] +
        0.3 * rec["rating_score"] +
        0.2 * rec["content_score"]
    )

    rec["reason"] = rec.apply(generate_reason, axis=1)

    rec = rec.sort_values("final_score", ascending=False)

    rec["distance"] = rec["distance"].apply(lambda x: f"{x:.2f} km")

    return rec[
        ["name", "distance", "avg_rating", "reason"]
    ].head(top_n)

def generate_reason(row):

    reasons = []

    if row["geo_score"] > 0.7:
        reasons.append("Cerca de tu ubicación")

    if row["rating_score"] > 0.6:
        reasons.append("Muy bien calificado por usuarios")

    if row["content_score"] > 0.5:
        reasons.append("Similar a productos populares")

    if len(reasons) == 0:
        reasons.append("Recomendado por el sistema")

    return reasons


# ==========================================
# PRUEBA DEL SISTEMA
# ==========================================

# for user_id in locations['user_id']:

#     print(f"Recomendaciones para Usuario {user_id}:")

#     print(recommend(user_id))

#     print("\n")