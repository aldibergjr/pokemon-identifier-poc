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

    def create_template(self, text, color=(255, 255, 0), stroke_width=3, stroke_color=(0, 0, 128)):
        """Create template using PIL with stroke effect."""
        # Get text size
        bbox = self.pil_font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create image with padding for stroke
        padding = 10 + stroke_width  # Extra padding for stroke
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding
        
        # Create image with transparent background
        image = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw text with stroke
        draw.text((padding, padding), text, font=self.pil_font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
        
        # Convert to numpy array and then to BGR
        template = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        
        # Save a debug copy to see how it looks
        debug_dir = 'debug/templates'
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f'{debug_dir}/{text}_preview.png', template)
        
        return template

def main():
    # Create directories
    os.makedirs('assets/text_templates', exist_ok=True)
    os.makedirs('debug/templates', exist_ok=True)
    
    # Initialize font generator
    font_path = 'assets/fonts/Ketchum.otf'
    generator = FontTemplateGenerator(font_path)
    
    # List of all Gen 1 Pokemon and status texts
    templates = [
        # Gen 1 Pokémon (151)
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
        "Mewtwo", "Mew",
        # Status texts
        "Wild", "Shiny", "Legendary"
    ]
    
    print("Generating templates with stroke effect...")
    for text in templates:
        try:
            # Generate template with blue stroke
            template = generator.create_template(text, color=(255, 255, 0), stroke_width=3, stroke_color=(0, 0, 128))
            
            # Save template
            filename = f'assets/text_templates/{text}.png'
            cv2.imwrite(filename, template)
            
            # Get template size
            h, w = template.shape[:2]
            print(f"Generated template for '{text}' (size: {w}x{h})")
            
        except Exception as e:
            print(f"Failed to generate template for '{text}': {e}")
    
    print("\nTemplate generation complete!")
    print(f"Templates saved in: {os.path.abspath('assets/text_templates')}")
    print(f"Preview templates saved in: {os.path.abspath('debug/templates')}")
    print(f"Total templates generated: {len(templates)}")

if __name__ == "__main__":
    main() 