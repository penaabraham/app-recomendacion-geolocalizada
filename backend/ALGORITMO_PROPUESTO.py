# Importación de Bibliotecas
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

# Cargar Datos
# Datos de calificaciones (simulando varios usuarios calificando diferentes productos)
ratings = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'item_id': [101, 102, 103, 104, 105],
    'rating': [5, 4, 3, 5, 4],
    'timestamp': [1622476800, 1622563200, 1622649600, 1622736000, 1622822400]
})

# Datos de comentarios
comments = pd.DataFrame({
    'item_id': [101, 102, 103, 104, 105],
    'comment': [
        "Excelente iPhone 12, muy recomendable", 
        "Samsung Galaxy S21, buena compra",
        "Google Pixel 5, buena relación calidad-precio",
        "Refrigerador LG, muy eficiente",
        "Refrigerador Samsung, satisface mis necesidades"
    ]
})

# Datos de ubicaciones (GPS) de los usuarios
locations = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'latitude': [34.0522, 36.1699, 40.7128, 41.8781, 34.0622],  # Los Ángeles, Las Vegas, Nueva York, Chicago, Cerca de Los Ángeles
    'longitude': [-118.2437, -115.1398, -74.0060, -87.6298, -118.2537]
})

# Datos de productos con ubicaciones en diferentes ciudades (ligeramente diferentes)
products = pd.DataFrame({
    'item_id': [101, 102, 103, 104, 105],
    'name': ["iPhone 12", "Samsung Galaxy S21", "Google Pixel 5", "Refrigerador LG", "Refrigerador Samsung"],
    'latitude': [34.0525, 36.1600, 40.7100, 41.8800, 34.0630],  # Coordenadas ajustadas para evitar coincidencias exactas
    'longitude': [-118.2450, -115.1500, -74.0070, -87.6300, -118.2500]
})

# Preprocesamiento de Datos

# Filtro Basado en Contenido
# Vectorizar comentarios
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(comments['comment'])

# Similitud coseno entre productos basados en comentarios
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Modelado de Tópicos con LDA
lda = LDA(n_components=5, random_state=42)
lda_matrix = lda.fit_transform(tfidf_matrix)

# Filtro Colaborativo
# Crear matriz de calificaciones
rating_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Similitud coseno entre usuarios
user_sim = cosine_similarity(rating_matrix, rating_matrix)

# Geolocalización
# Calcular distancias geográficas (usando la fórmula de Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

# Añadir distancia a los productos
# --- MODIFICACIÓN EN LA FUNCIÓN DE DISTANCIA ---
def add_distance(user_id, lat_usuario=None, lon_usuario=None):
    # Si el celular manda coordenadas (lat/lon), las usamos.
    # Si no, buscamos en el DataFrame 'locations' como antes.
    if lat_usuario is not None and lon_usuario is not None:
        u_lat, u_lon = lat_usuario, lon_usuario
    else:
        user_loc = locations[locations['user_id'] == user_id]
        u_lat, u_lon = user_loc['latitude'].values[0], user_loc['longitude'].values[0]
    
    distances = []
    for index, product in products.iterrows():
        dist = haversine(u_lat, u_lon, product['latitude'], product['longitude'])
        distances.append(dist)
    products['distance'] = distances
    return u_lat, u_lon # Devolvemos las coordenadas usadas

# Asignar la distancia para todos los usuarios antes de la recomendación
for user_id in locations['user_id']:
    add_distance(user_id)

# Clustering de Ubicaciones
kmeans = KMeans(n_clusters=3, random_state=42)
locations['cluster'] = kmeans.fit_predict(locations[['latitude', 'longitude']])

# Asignar clusters a productos (necesario para filtrar por cluster en las recomendaciones)
products['cluster'] = kmeans.predict(products[['latitude', 'longitude']])

# Implementación de un Modelo de Aprendizaje Profundo

# Tokenización de comentarios
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments['comment'])
sequences = tokenizer.texts_to_sequences(comments['comment'])
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=100)

# Crear modelo de embeddings
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=100))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del Modelo
labels = np.array([1 if rating > 3 else 0 for rating in ratings['rating'].values])  # Simplificar a clasificación binaria
model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)

# Generación de Recomendaciones

# --- MODIFICACIÓN EN LA FUNCIÓN RECOMMEND ---
def recommend(user_id, lat_manual=None, lon_manual=None):
    # 1. Actualizar distancias con la ubicación real del celular
    u_lat, u_lon = add_distance(user_id, lat_manual, lon_manual)
    
    # 2. Determinar a qué cluster pertenece la ubicación actual del celular
    # El modelo KMeans predice el cluster para la posición recibida
    user_cluster = kmeans.predict([[u_lat, u_lon]])[0]
    
    # 3. Filtrar productos por el cluster detectado
    cluster_products = products[products['cluster'] == user_cluster].copy()

    # 4. Si el cluster está vacío (porque estás muy lejos), mostrar los más cercanos globales
    if cluster_products.empty:
        recommendations = products.sort_values('distance').head(5).copy()
    else:
        recommendations = cluster_products.sort_values('distance').head(5).copy()
    
    # 5. Formatear salida
    recommendations['distance'] = recommendations['distance'].apply(lambda x: f"{x:.2f} km")
    
    return recommendations[['name', 'distance']]

# Obtener recomendaciones para cada usuario
for user_id in locations['user_id']:
    print(f"Recomendaciones para Usuario {user_id}:")
    print(recommend(user_id), end="\n\n")
