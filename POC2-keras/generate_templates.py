import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class FontTemplateGenerator:
    def __init__(self, font_path, font_size=24):
        """Initialize the font template generator with the custom font."""
        self.font_size = font_size
        self.pil_font = ImageFont.truetype(font_path, font_size)
        print(f"Using font: {font_path} (size: {font_size})")

def main():
    # Create directories
    os.makedirs('assets/text_templates', exist_ok=True)
    os.makedirs('debug/templates', exist_ok=True)
    
    # Initialize font generator
    FONT_PATH = 'assets/fonts/Ketchum.otf'
    OUTPUT_DIR = 'assets/text_templates'
    generator = FontTemplateGenerator(FONT_PATH)
    
    # List of all Gen 1 Pokemon
    pokemon_names = [
        "Bulbasaur", "Ivysaur", "Venusaur",
        "Charmander", "Charmeleon", "Charizard",
        "Squirtle", "Wartortle", "Blastoise",
        "Caterpie", "Metapod", "Butterfree",
        "Weedle", "Kakuna", "Beedrill",
        "Pidgey", "Pidgeotto", "Pidgeot",
        "Rattata", "Raticate",
        "Spearow", "Fearow",
        "Ekans", "Arbok",
        "Pikachu", "Raichu",
        "Sandshrew", "Sandslash",
        "Nidoranf", "Nidorina", "Nidoqueen",
        "Nidoranm", "Nidorino", "Nidoking",
        "Clefairy", "Clefable",
        "Vulpix", "Ninetales",
        "Jigglypuff", "Wigglytuff",
        "Zubat", "Golbat",
        "Oddish", "Gloom", "Vileplume",
        "Paras", "Parasect",
        "Venonat", "Venomoth",
        "Diglett", "Dugtrio",
        "Meowth", "Persian",
        "Psyduck", "Golduck",
        "Mankey", "Primeape",
        "Growlithe", "Arcanine",
        "Poliwag", "Poliwhirl", "Poliwrath",
        "Abra", "Kadabra", "Alakazam",
        "Machop", "Machoke", "Machamp",
        "Bellsprout", "Weepinbell", "Victreebel",
        "Tentacool", "Tentacruel",
        "Geodude", "Graveler", "Golem",
        "Ponyta", "Rapidash",
        "Slowpoke", "Slowbro",
        "Magnemite", "Magneton",
        "Farfetch'd",
        "Doduo", "Dodrio",
        "Seel", "Dewgong",
        "Grimer", "Muk",
        "Shellder", "Cloyster",
        "Gastly", "Haunter", "Gengar",
        "Onix",
        "Drowzee", "Hypno",
        "Krabby", "Kingler",
        "Voltorb", "Electrode",
        "Exeggcute", "Exeggutor",
        "Cubone", "Marowak",
        "Hitmonlee", "Hitmonchan",
        "Lickitung",
        "Koffing", "Weezing",
        "Rhyhorn", "Rhydon",
        "Chansey",
        "Tangela",
        "Kangaskhan",
        "Horsea", "Seadra",
        "Goldeen", "Seaking",
        "Staryu", "Starmie",
        "Mr. Mime",
        "Scyther",
        "Jynx",
        "Electabuzz",
        "Magmar",
        "Pinsir",
        "Tauros",
        "Magikarp", "Gyarados",
        "Lapras",
        "Ditto",
        "Eevee", "Vaporeon", "Jolteon", "Flareon",
        "Porygon",
        "Omanyte", "Omastar",
        "Kabuto", "Kabutops",
        "Aerodactyl",
        "Snorlax",
        "Articuno", "Zapdos", "Moltres",
        "Dratini", "Dragonair", "Dragonite",
        "Mewtwo", "Mew"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, 50)
    with open(f"{OUTPUT_DIR}/mapping.txt", "w") as ann_file:
        for i, text in enumerate(pokemon_names):
            pokemon_name = text
            text = f"Onde está\n{pokemon_name}?"
            # Create a temporary image to calculate text size
            temp_img = Image.new("RGB", (1, 1), color="black")
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create final image with padding
            padding = 40
            img_width = text_width + (2 * padding)
            img_height = text_height + (2 * padding)
            img = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Calculate position to center text
            x = (img_width - text_width) // 2
            y = (img_height - text_height) // 2
            
            # Draw stroke (outline) by drawing text multiple times with offset
            stroke_width = 3
            stroke_color = "black"
            
            # Draw the stroke by offsetting the text in 8 directions
            for offset_x in range(-stroke_width, stroke_width + 1):
                for offset_y in range(-stroke_width, stroke_width + 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    draw.text((x + offset_x, y + offset_y), text, font=font, fill=stroke_color)
            
            # Draw the main text on top
            draw.text((x, y), text, font=font, fill="yellow")
            
            # Convert to RGB with black background
            background = Image.new("RGB", (img_width, img_height), color="black")
            background.paste(img, (0, 0), img)
            img = background
            
            img.save(f"{OUTPUT_DIR}/img_{i}.png")
            ann_file.write(f"{os.path.join('assets\\text_templates', f'img_{i}.png')}\t{pokemon_name}\n")
   
main() 