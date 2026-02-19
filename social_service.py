from mastodon import Mastodon
import re

class MastodonService:
    def __init__(self):
        self.api = Mastodon(api_base_url='https://mastodon.social')

    def limpiar_html(self, html_text):
        # Limpia las etiquetas <p> y <br> que vienen en el contenido
        return re.sub(r'<[^>]*>', '', html_text)

    def obtener_propuesta_datos(self, hashtag):
        try:
            posts = self.api.timeline_hashtag(hashtag, limit=5)
            resultados = []
            
            for p in posts:
                # Extraemos lo vital para tu tesis
                data = {
                    "id_post": p['id'],
                    "fecha": p['created_at'], # Tiempo Real
                    "texto_limpio": self.limpiar_html(p['content']), # Filtro de Contenido
                    "popularidad": p['replies_count'] + p['reblogs_count'] + p['favourites_count'], # Proxy de Rating
                    "usuario": {
                        "id": p['account']['id'],
                        "nombre": p['account']['display_name'],
                        "bio": self.limpiar_html(p['account']['note']), # Vectorización de Usuario
                        "seguidores": p['account']['followers_count']
                    },
                    "etiquetas": [t['name'] for t in p['tags']]
                }
                resultados.append(data)
            return resultados
        except Exception as e:
            print(f"Error: {e}")
            return []