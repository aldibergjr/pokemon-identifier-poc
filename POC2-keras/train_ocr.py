import keras_ocr
import os
from sklearn.model_selection import train_test_split

# Load annotations
with open("assets/text_templates/mapping.txt") as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
image_paths, texts = zip(*lines)

# Split into train/test
train_images, val_images, train_texts, val_texts = train_test_split(
    image_paths, texts, test_size=0.1, random_state=42
)


# Create recognizer instance (it downloads weights if not found)
recognizer = keras_ocr.recognition.Recognizer()

# Compile (optional but recommended)
recognizer.compile()

# Fit the recognizer
recognizer.fit(
    image_filenames=train_images,
    labels=train_texts,
    validation_data=(val_images, val_texts),
    epochs=25,
    batch_size=4
)

# Save model weights
recognizer.model.save_weights("pokemon_recognizer_weights.h5")