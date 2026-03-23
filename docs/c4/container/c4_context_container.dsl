workspace "Sistema de Recomendación Híbrido" "Modelo C4 - Contexto y Contenedores" {

    model {

        # ── Personas ───────────────────────────────────────────
        usuario = person "Usuario" "Persona que utiliza la aplicación móvil para consultar recomendaciones de productos y servicios cercanos a su ubicación."

        # ── Sistema principal ──────────────────────────────────
        sistemaRecomendacion = softwareSystem "Sistema de Recomendación Híbrido" "Aplicación móvil con backend de procesamiento que genera recomendaciones personalizadas de productos combinando geolocalización, filtrado colaborativo y análisis de contenido." {

            # ── Contenedores ───────────────────────────────────
            appMovil = container "Aplicación Móvil" "Interfaz de usuario multiplataforma. Obtiene la ubicación GPS del dispositivo, envía la solicitud al backend y presenta las recomendaciones con paginación y búsqueda en tiempo real." "Flutter / Dart" "MobileApp"

            backend = container "Backend API" "Expone el endpoint REST de recomendación. Carga los modelos serializados al iniciar y orquesta la llamada al módulo del algoritmo." "FastAPI 0.110.0 / Python 3" "Backend"

            algoritmo = container "Módulo del Algoritmo" "Implementa el sistema de recomendación híbrido: cálculo de scores geográfico, de calificación y de contenido (TF-IDF, LDA, sentimiento). Devuelve los productos ordenados por score final." "Python 3 / scikit-learn / TensorFlow" "Algorithm"

            dataStore = container "Almacén de Datos" "Archivos JSON estáticos que contienen usuarios, ubicaciones, productos, calificaciones y comentarios generados sintéticamente." "JSON / Sistema de archivos" "DataStore"

            modelos = container "Modelos Serializados" "Artefactos de ML generados por train.py y cargados en memoria al iniciar el servidor: TF-IDF, similitud coseno, LDA, K-Means, embeddings Keras, tokenizer y rating_avg." "joblib / pickle / Keras (.keras)" "DataStore"
        }

        # ── Sistemas externos ──────────────────────────────────
        servicioGPS = softwareSystem "Servicio GPS" "Servicio del sistema operativo del dispositivo móvil que proporciona las coordenadas geográficas actuales del usuario." "External"

        # ── Relaciones de contexto ─────────────────────────────
        usuario -> appMovil "Busca productos y consulta recomendaciones" "Interfaz táctil"
        appMovil -> servicioGPS "Solicita coordenadas geográficas" "API del SO"
        servicioGPS -> appMovil "Retorna latitud y longitud actuales" "API del SO"

        # ── Relaciones entre contenedores ──────────────────────
        appMovil -> backend "Envía user_id, latitud y longitud; recibe lista de productos recomendados" "HTTPS / JSON"
        backend -> algoritmo "Invoca recommend() con user_id y coordenadas" "Llamada interna Python"
        algoritmo -> dataStore "Lee usuarios, ubicaciones, productos, calificaciones y comentarios" "pandas / JSON"
        algoritmo -> modelos "Carga modelos serializados al iniciar; los consulta en cada recomendación" "joblib / pickle"
        backend -> modelos "Carga modelos en memoria al arrancar el servidor" "joblib / pickle"
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
            element "Algorithm" {
                shape Component
                background #6A1B9A
                color #ffffff
                fontSize 13
            }
            element "DataStore" {
                shape Cylinder
                background #E65100
                color #ffffff
                fontSize 13
            }
        }

    }

}
