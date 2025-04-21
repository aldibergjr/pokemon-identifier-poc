import cv2
import os
import numpy as np

class PokemonFinder:
    def __init__(self, pokemon_name: str):
        self.pokemon_name = pokemon_name.lower().strip()
        self.assets_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'assets', 'pokemons'
        )
        self.pokemon_image_path = os.path.join(self.assets_path, f'{self.pokemon_name}.png')

        if not os.path.exists(self.pokemon_image_path):
            raise FileNotFoundError(f'Imagem do Pokémon "{self.pokemon_name}" não encontrada em {self.pokemon_image_path}')

        self.pokemon_image = cv2.imread(self.pokemon_image_path, cv2.IMREAD_UNCHANGED)  # Lê com canal alfa se existir
        if self.pokemon_image is None:
            raise ValueError(f'Não foi possível carregar a imagem do Pokémon: {self.pokemon_image_path}')
        
        # Se a imagem tem 4 canais (BGRA), remova o canal alfa
        if self.pokemon_image.shape[2] == 4:
            self.pokemon_image = cv2.cvtColor(self.pokemon_image, cv2.COLOR_BGRA2BGR)

        self.pokemon_image_gray = cv2.cvtColor(self.pokemon_image, cv2.COLOR_BGR2GRAY)

    def find_on_screen(self, screen_image: np.ndarray):
        """
        Encontra o Pokémon na imagem fornecida.
        
        :param screen_image: Imagem de entrada onde a busca será feita (deve ser um array NumPy).
        :return: Tupla com as coordenadas do canto superior esquerdo e inferior direito
                 da área onde o Pokémon foi encontrado, ou None se não encontrado.
        """
        # Converte a imagem da tela para escala de cinza
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)

        # Realiza o template matching
        result = cv2.matchTemplate(screen_gray, self.pokemon_image_gray , cv2.TM_CCOEFF_NORMED)

        # Encontra a posição da melhor correspondência
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Se a correspondência for boa o suficiente, retorna a posição
        if max_val >= 0.7:  # Ajuste do valor de limiar (0.7)
            top_left = max_loc
            h, w = self.pokemon_image.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Desenha o retângulo na imagem para visualizar o resultado
            cv2.rectangle(screen_image, top_left, bottom_right, (0, 255, 0), 2)  # Verde e espessura 2

            # Exibe a imagem com o retângulo
            cv2.imshow('Imagem com Pokémon Encontrado', screen_image)
            cv2.waitKey(0)  # Espera até que uma tecla seja pressionada
            cv2.destroyAllWindows()  # Fecha a janela de visualização

            return top_left, bottom_right
        else:
            print(f"Correspondência não encontrada com valor máximo {max_val:.2f}")
            return None