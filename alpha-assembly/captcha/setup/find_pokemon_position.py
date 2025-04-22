import cv2
import os
import numpy as np
from captcha.process_captcha import crop_img
class PokemonFinder:
    def __init__(self, pokemon_name: str):
        self.pokemon_name = pokemon_name.lower().strip()
        self.assets_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'assets', 'combined'
        )
        self.pokemon_image_path = os.path.join(self.assets_path, f'{self.pokemon_name}.png')

        if not os.path.exists(self.pokemon_image_path):
            raise FileNotFoundError(f'Imagem do Pokémon "{self.pokemon_name}" não encontrada em {self.pokemon_image_path}')

        self.pokemon_image = cv2.imread(self.pokemon_image_path, cv2.IMREAD_UNCHANGED)
        if self.pokemon_image is None:
            raise ValueError(f'Não foi possível carregar a imagem do Pokémon: {self.pokemon_image_path}')
        
        # Se a imagem tem 4 canais (BGRA), crie uma máscara do canal alfa
        if self.pokemon_image.shape[2] == 4:
            # Extrair o canal alfa
            self.alpha_mask = self.pokemon_image[:, :, 3]
            # Converter para BGR
            self.pokemon_image = cv2.cvtColor(self.pokemon_image, cv2.COLOR_BGRA2BGR)
            # Criar uma máscara binária
            _, self.alpha_mask = cv2.threshold(self.alpha_mask, 127, 255, cv2.THRESH_BINARY)

        self.pokemon_image_gray = cv2.cvtColor(self.pokemon_image, cv2.COLOR_BGR2GRAY)
        # Aplicar a máscara na imagem em escala de cinza se existir
        if hasattr(self, 'alpha_mask'):
            self.pokemon_image_gray = cv2.bitwise_and(self.pokemon_image_gray, self.pokemon_image_gray, mask=self.alpha_mask)

    def find_on_screen(self, screen_image: np.ndarray):
        """
        Encontra o Pokémon na imagem fornecida usando template matching multi-escala.
        
        :param screen_image: Imagem de entrada onde a busca será feita (deve ser um array NumPy).
        :return: Tupla com as coordenadas do canto superior esquerdo e inferior direito
                 da área onde o Pokémon foi encontrado, ou None se não encontrado.
        """
        # Converte a imagem da tela para escala de cinza
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        # screen_gray = crop_img(screen_gray, 0.5, 2.2, 2.4)

        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        best_match = None
        best_scale = 1.0
        max_correlation = 0
        
        template_h, template_w = self.pokemon_image_gray.shape[:2]
        
        for scale in scales:
            # Redimensiona o template
            width = int(template_w * scale)
            height = int(template_h * scale)
            resized_template = cv2.resize(self.pokemon_image_gray, (width, height))
            
            if hasattr(self, 'alpha_mask'):
                resized_mask = cv2.resize(self.alpha_mask, (width, height))
                resized_template = cv2.bitwise_and(resized_template, resized_template, mask=resized_mask)
            
            # Realiza o template matching
            result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            
            # Encontra a posição da melhor correspondência
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > max_correlation:
                max_correlation = max_val
                best_match = (max_loc, (width, height))
                best_scale = scale

        # Fecha todas as janelas de heatmap
        cv2.destroyAllWindows()

        # Se a melhor correspondência for boa o suficiente, retorna a posição
        if max_correlation >= 0.1:  # Ajuste do valor de limiar (0.7)
            top_left = best_match[0]
            w, h = best_match[1]
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Calcula o centro do retângulo para o círculo
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2
            
            # Desenha o círculo no centro da detecção
            radius = min(w, h) // 4  # Raio proporcional ao tamanho do template
            cv2.circle(screen_gray, (center_x, center_y), radius, (0, 0, 255), 2)  # Círculo vermelho
            
            # Desenha o retângulo na imagem para visualizar o resultado
            cv2.rectangle(screen_gray, top_left, bottom_right, (0, 255, 0), 2)  # Verde e espessura 2
            
            # Adiciona texto mostrando a escala e correlação
            text = f"Scale: {best_scale:.2f}, Conf: {max_correlation:.2f}"
            cv2.putText(screen_gray, text, (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Exibe a imagem com o retângulo e círculo
            # cv2.imshow('Imagem com Pokémon Encontrado', screen_gray)
            # cv2.waitKey(0)  # Espera até que uma tecla seja pressionada
            # cv2.destroyAllWindows()  # Fecha a janela de visualização

            return center_x, center_y
        else:
            print(f"Correspondência não encontrada. Melhor valor: {max_correlation:.2f} na escala {best_scale:.2f}")
            return None