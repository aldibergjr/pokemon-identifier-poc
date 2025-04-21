import keras_ocr
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_image(image_path, target_height=32, target_width=200):
    """Load and preprocess a single image for OCR."""
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Resize image while maintaining aspect ratio
    width, height = img.size
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    
    if new_width > target_width:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new image with padding
    new_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    # Convert to numpy array and normalize
    img_array = np.array(new_img) / 255.0
    return img_array

def create_alphabet(texts):
    """Create an alphabet from the texts."""
    chars = set()
    for text in texts:
        chars.update(text)
    return ''.join(sorted(chars))

def main():
    # Set paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    mapping_file = os.path.join(base_path, 'assets', 'text_templates', 'mapping.txt')
    
    # Load annotations
    with open(mapping_file) as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
    
    # Extract image paths and texts
    image_paths = []
    texts = []
    for img_path, text in lines:
        # Convert relative path to absolute path
        abs_path = os.path.join(base_path, img_path)
        if os.path.exists(abs_path):
            image_paths.append(abs_path)
            texts.append(text)
        else:
            print(f"Warning: Image not found: {abs_path}")
    
    if not image_paths:
        print("No images found. Please check your image paths.")
        return
    
    print(f"Loaded {len(image_paths)} images")
    
    # Create alphabet from texts
    alphabet = create_alphabet(texts)
    print(f"Created alphabet with {len(alphabet)} characters: {alphabet}")
    
    # Split into train/test
    train_images, val_images, train_texts, val_texts = train_test_split(
        image_paths, texts, test_size=0.1, random_state=42
    )
    
    print(f"Training on {len(train_images)} images, validating on {len(val_images)} images")
    
    # Create a custom model using keras-ocr's build_model function
    height = 32
    width = 200
    color = True
    filters = [64, 128, 256, 512, 512, 512, 512]
    rnn_units = [128, 128]  # Fixed: Now a list with 2 elements
    dropout = 0.25
    rnn_steps_to_discard = 2
    pool_size = 2
    
    # The build_model function returns a tuple of 4 models
    backbone, model, training_model, prediction_model = keras_ocr.recognition.build_model(
        alphabet=alphabet,
        height=height,
        width=width,
        color=color,
        filters=filters,
        rnn_units=rnn_units,
        dropout=dropout,
        rnn_steps_to_discard=rnn_steps_to_discard,
        pool_size=pool_size
    )
    
    # Compile the training model with an optimizer
    # The CTC loss is already built into the model
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    # Preprocess training data
    print("Preprocessing training data...")
    X_train = np.array([preprocess_image(img_path) for img_path in train_images])
    X_val = np.array([preprocess_image(img_path) for img_path in val_images])
    
    # Create a simple training loop
    print("Training the model...")
    batch_size = 8
    epochs = 25
    
    # Create a data generator
    def data_generator():
        indices = np.arange(len(X_train))
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images = X_train[batch_indices]
                batch_texts = [train_texts[idx] for idx in batch_indices]
                
                # Create inputs for the CTC loss
                batch_labels = np.zeros((len(batch_indices), 1))
                batch_label_length = np.ones((len(batch_indices), 1))
                batch_input_length = np.ones((len(batch_indices), 1)) * 50  # Arbitrary value
                
                # The model expects these specific input keys
                yield {
                    'input_1': batch_images,
                    'labels': batch_labels,
                    'input_3': batch_label_length,
                    'input_4': batch_input_length
                }, np.zeros(len(batch_indices))  # Dummy target
    
    # Train the model
    training_model.fit(
        data_generator(),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(
            {
                'input_1': X_val,
                'labels': np.zeros((len(X_val), 1)),
                'input_3': np.ones((len(X_val), 1)),
                'input_4': np.ones((len(X_val), 1)) * 50
            },
            np.zeros(len(X_val))
        ),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Save model weights
    model.save_weights("pokemon_recognizer_weights.h5")
    print("Model weights saved as 'pokemon_recognizer_weights.h5'")
    
    # Test the model on a few images
    print("\nTesting the model on a few images:")
    test_images = val_images[:5]  # Test on first 5 validation images
    test_images_processed = np.array([preprocess_image(img_path) for img_path in test_images])
    
    # Use the prediction model for inference
    predictions = prediction_model.predict(test_images_processed)
    
    for i, (image_path, prediction) in enumerate(zip(test_images, predictions)):
        print(f"Image {i+1}: {os.path.basename(image_path)}")
        print(f"  True text: {val_texts[i]}")
        print(f"  Recognized: {prediction}")
        print()

if __name__ == "__main__":
    main() 