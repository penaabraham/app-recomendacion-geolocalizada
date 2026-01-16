workspace "Arquitectura del Sistema de Recomendación" "Diagrama de Contenedores (C4)" {

    model {
        usuario = person "Usuario Final" "Persona que usa la app móvil."

        sistemaRecomendacion = softwareSystem "App de Recomendaciones Híbridas" {
            
            appMovil = container "Aplicación Móvil (Flutter)" "Provee la interfaz de usuario y captura la ubicación GPS." "Dart/Flutter"
            
            apiBackend = container "API de Recomendación (FastAPI)" "Expone el algoritmo de recomendación como un servicio web." "Python/FastAPI" {
                tags "Algoritmo"
            }

            baseDatos = container "Base de Datos" "Almacena perfiles de usuario, productos y logs de ubicación." "SQLite/CSV" "Database"
            
            modeloIA = container "Modelo de ML/Deep Learning" "Script que ejecuta el filtro colaborativo, contenido y clustering." "TensorFlow/Scikit-Learn" "Component"
        }

        # Relaciones entre contenedores
        usuario -> appMovil "Usa para ver recomendaciones"
        appMovil -> apiBackend "Envía user_id y coordenadas GPS" "JSON/HTTPS"
        apiBackend -> modeloIA "Llama a la función recommend()"
        apiBackend -> baseDatos "Lee y escribe datos de usuarios/productos"
        modeloIA -> baseDatos "Extrae datos para entrenamiento y predicción"
    }

    views {
        container sistemaRecomendacion "DiagramaContenedores" {
            include *
            autolayout lr
        }

        styles {
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Database" {
                shape Cylinder
                background #85bbf0
            }
            element "Algoritmo" {
                background #1168bd
            }
            element "Component" {
                shape Component
                background #85bbf0
                color #000000
            }
        }
    }
}