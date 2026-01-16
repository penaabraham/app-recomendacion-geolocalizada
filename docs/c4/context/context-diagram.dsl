workspace "Sistema de Recomendación por Geolocalización" "Proyecto basado en el algoritmo híbrido del artículo de investigación." {

    model {
        usuario = person "Usuario Final" "Persona que busca productos o servicios cercanos según sus gustos."
        
        sistemaRecomendacion = softwareSystem "App de Recomendaciones Híbridas" "Permite visualizar productos sugeridos basados en gustos, comportamiento y ubicación GPS." {
            tags "SistemaPrincipal"
        }

        # Sistemas Externos
        gps = softwareSystem "Servicios de Ubicación (GPS)" "Provee las coordenadas de latitud y longitud del dispositivo móvil." "External System"
        apiExterna = softwareSystem "Fuentes de Datos (TripAdvisor/Foursquare)" "APIs externas de donde se recopilaron los datasets originales para el entrenamiento." "External System"

        # Relaciones
        usuario -> sistemaRecomendacion "Busca productos y recibe sugerencias personalizadas"
        sistemaRecomendacion -> gps "Solicita coordenadas actuales del usuario"
        sistemaRecomendacion -> apiExterna "Obtiene datos históricos de productos y comentarios"
    }

    views {
        systemContext sistemaRecomendacion "DiagramaContexto" {
            include *
            autolayout lr
        }

        styles {
            element "Person" {
                shape Person
                background #08427b
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "External System" {
                background #999999
                color #ffffff
            }
            element "SistemaPrincipal" {
                background #00a000
                color #ffffff
            }
        }
    }
}