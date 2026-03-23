workspace "Sistema de Recomendación Híbrido" "Modelo C4 - Diagrama de Contexto" {

    model {

        # ── Personas ──────────────────────────────────────────
        usuario = person "Usuario" "Persona que utiliza la aplicación móvil para consultar recomendaciones de productos y servicios cercanos a su ubicación."

        # ── Sistema principal ──────────────────────────────────
        sistemaRecomendacion = softwareSystem "Sistema de Recomendación Híbrido" "Aplicación móvil con backend de procesamiento que genera recomendaciones personalizadas de productos combinando geolocalización, filtrado colaborativo y análisis de contenido."

        # ── Sistemas externos ──────────────────────────────────
        servicioGPS = softwareSystem "Servicio GPS" "Servicio del sistema operativo del dispositivo móvil que proporciona las coordenadas geográficas actuales del usuario." "External"

        # ── Relaciones ─────────────────────────────────────────
        usuario -> sistemaRecomendacion "Consulta recomendaciones de productos cercanos" "HTTP / Interfaz móvil"
        sistemaRecomendacion -> servicioGPS "Solicita coordenadas geográficas en tiempo real" "API del SO"
        servicioGPS -> sistemaRecomendacion "Retorna latitud y longitud actuales" "API del SO"

    }

    views {

        systemContext sistemaRecomendacion "Contexto" {
            include *
            autoLayout tb
            title "Diagrama de Contexto – Sistema de Recomendación Híbrido"
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
        }

    }

}
