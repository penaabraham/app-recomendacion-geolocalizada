workspace "Sistema de Recomendación Híbrido" "Modelo C4 - Contexto, Contenedores y Componentes" {

    model {

        # ── Personas ───────────────────────────────────────────
        usuario = person "Usuario" "Persona que utiliza la aplicación móvil para consultar recomendaciones de productos y servicios cercanos a su ubicación."

        # ── Sistema principal ──────────────────────────────────
        sistemaRecomendacion = softwareSystem "Sistema de Recomendación Híbrido" "Aplicación móvil con backend de procesamiento que genera recomendaciones personalizadas de productos combinando geolocalización, filtrado colaborativo y análisis de contenido." {

            # ── Contenedores ───────────────────────────────────
            appMovil = container "Aplicación Móvil" "Interfaz de usuario multiplataforma. Obtiene la ubicación GPS del dispositivo, envía la solicitud al backend y presenta las recomendaciones con paginación y búsqueda en tiempo real." "Flutter / Dart" "MobileApp"

            backend = container "Backend + Algoritmo" "Expone el endpoint REST de recomendación, orquesta el algoritmo híbrido y devuelve los resultados ordenados. Carga los modelos serializados en memoria al iniciar el servidor." "FastAPI 0.110.0 / Python 3 / scikit-learn / TensorFlow" "Backend" {

                # ── Componentes ────────────────────────────────

                endpointRecomendacion = component "Endpoint /recomendar/{user_id}" "Recibe user_id, latitud y longitud como parámetros. Valida la entrada, establece coordenadas por defecto si no se proporcionan, invoca el motor de recomendación y serializa la respuesta como JSON." "FastAPI route handler"

                cargaModelos = component "Cargador de Modelos" "Carga en memoria al iniciar el servidor todos los artefactos serializados: TF-IDF, matriz de similitud coseno, LDA, K-Means, rating_avg, tokenizer y modelo Keras. Evita reentrenamiento en tiempo de solicitud." "joblib / pickle / tf.keras"

                motorGeo = component "Motor Geográfico" "Calcula la distancia entre el usuario y cada producto mediante la fórmula de Haversine. Normaliza las distancias de forma inversa para producir el geo_score." "Python / NumPy"

                motorColaborativo = component "Motor Colaborativo" "Calcula el rating_score normalizando el promedio de calificaciones históricas por producto a partir de la matriz usuario-producto." "pandas / NumPy"

                motorContenido = component "Motor de Contenido" "Combina tres señales textuales: similitud coseno TF-IDF (peso 0.5), intensidad de tópico dominante LDA (peso 0.2) y probabilidad de sentimiento positivo del modelo Keras (peso 0.3)." "scikit-learn / TensorFlow"

                combinador = component "Combinador de Scores" "Integra los tres scores parciales en el score final ponderado: geo (0.40), rating (0.25) y contenido (0.35). Ordena los productos y genera las etiquetas de razón de recomendación." "Python"
            }

            dataStore = container "Almacén de Datos" "Archivos JSON estáticos que contienen usuarios, ubicaciones, productos, calificaciones y comentarios generados sintéticamente." "JSON / Sistema de archivos" "DataStore"

            modelos = container "Modelos Serializados" "Artefactos de ML generados por train.py: TF-IDF, similitud coseno, LDA, K-Means, embeddings Keras, tokenizer y rating_avg." "joblib / pickle / Keras (.keras)" "DataStore"
        }

        # ── Sistemas externos ──────────────────────────────────
        servicioGPS = softwareSystem "Servicio GPS" "Servicio del sistema operativo del dispositivo móvil que proporciona las coordenadas geográficas actuales del usuario." "External"

        # ── Relaciones de contexto ─────────────────────────────
        usuario -> appMovil "Busca productos y consulta recomendaciones" "Interfaz táctil"
        appMovil -> servicioGPS "Solicita coordenadas geográficas" "API del SO"
        servicioGPS -> appMovil "Retorna latitud y longitud actuales" "API del SO"

        # ── Relaciones entre contenedores ──────────────────────
        appMovil -> backend "Envía user_id, latitud y longitud; recibe lista de productos recomendados" "HTTPS / JSON"
        backend -> dataStore "Lee usuarios, ubicaciones, productos, calificaciones y comentarios" "pandas / JSON"
        backend -> modelos "Carga y consulta artefactos de ML" "joblib / pickle"

        # ── Relaciones entre componentes ───────────────────────
        appMovil -> endpointRecomendacion "GET /recomendar/{user_id}?lat=&lon=" "HTTPS / JSON"

        endpointRecomendacion -> motorGeo "Pasa user_id y coordenadas" ""
        endpointRecomendacion -> motorColaborativo "Pasa user_id" ""
        endpointRecomendacion -> motorContenido "Pasa lista de productos" ""
        endpointRecomendacion -> combinador "Pasa los tres scores parciales" ""

        cargaModelos -> motorGeo "Provee productos con distancias" ""
        cargaModelos -> motorColaborativo "Provee rating_avg y rating_matrix" ""
        cargaModelos -> motorContenido "Provee similarity_df, lda_matrix y modelo Keras" ""

        motorGeo -> dataStore "Lee products.json y locations.json" "pandas"
        motorColaborativo -> dataStore "Lee ratings.json" "pandas"
        motorContenido -> dataStore "Lee comments.json" "pandas"

        motorGeo -> combinador "geo_score" ""
        motorColaborativo -> combinador "rating_score" ""
        motorContenido -> combinador "content_score" ""

        cargaModelos -> modelos "Deserializa artefactos al iniciar" "joblib / pickle"
    }

    views {

        # ── Vista de contexto ──────────────────────────────────
        systemContext sistemaRecomendacion "Contexto" {
            include *
            autoLayout tb
            title "Diagrama de Contexto – Sistema de Recomendación Híbrido"
        }

        # ── Vista de contenedores ──────────────────────────────
        container sistemaRecomendacion "Contenedores" {
            include *
            autoLayout tb
            title "Diagrama de Contenedores – Sistema de Recomendación Híbrido"
        }

        # ── Vista de componentes (Backend + Algoritmo) ─────────
        component backend "Componentes" {
            include *
            autoLayout tb
            title "Diagrama de Componentes – Backend + Algoritmo"
        }

        styles {
            element "Person" {
                shape Person
                background #1168BD
                color #ffffff
                fontSize 14
            }
            element "Software System" {
                background #1168BD
                color #ffffff
                fontSize 14
            }
            element "External" {
                background #999999
                color #ffffff
                fontSize 14
            }
            element "MobileApp" {
                shape MobileDeviceLandscape
                background #1168BD
                color #ffffff
                fontSize 13
            }
            element "Backend" {
                shape Hexagon
                background #2E7D32
                color #ffffff
                fontSize 13
            }
            element "DataStore" {
                shape Cylinder
                background #E65100
                color #ffffff
                fontSize 13
            }
            element "Component" {
                background #4A148C
                color #ffffff
                fontSize 12
            }
        }

    }

}
