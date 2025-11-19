# @Jahid Hasan Emon
# @SID : 001360753-7

# Project : Handwritten Digit Recognition with Custom Dataset and CNN.
# Method I Used : Pygame for Drawing, TensorFlow/Keras for CNN and Image Preprocessing with PIL.

# System Process : First collect custom digit data using Pygame, 
# then train a CNN model on this data, and finally predict digits
# drawn in Pygame using the trained model.

"""

Description of the code: will update later

"""

# ->> Optional Environment settings for Error suppression and CPU usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                  # disable GPU (CPU is enough)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"                   # suppress TF logs
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  # fix macOS crash


# ->> Import Libraries
import numpy as np                                            # Numpy to handle arrays
from PIL import Image, ImageOps                               # PIL for image processing
import pygame                                                 # Pygame for drawing interface

# TensorFlow/Keras for CNN -> Seq for model and load_model to load saved model
from tensorflow.keras.models import Sequential, load_model 
# Conv2d for pattern, edges maxpooing for downsampling, flatten helps to convert
# 2d feature map to 1d vector, dense for neuron conncetion, droupout to prevent overfitting
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# to_categorical to convert labels to one-hot encoding
from tensorflow.keras.utils import to_categorical
# Adam works as an optimizer that adjusts learning rate during training automatically
from tensorflow.keras.optimizers import Adam

DATASET_DIR = "JahidEmon/My_Dataset"                    # Directory to store custom dataset


# ->> Preprocess Images to make it compatible for CNN
def preprocess_image(path):                   
   
    img = Image.open(path).convert("L")       # Convert to grayscale
    img = ImageOps.invert(img)                # Invert colors (black bg, white digit)
    arr = np.array(img)                       # Convert to numpy array

    # Find bounding box of the drawn pixels
    coordinates = np.argwhere(arr > 20)            
    if coordinates.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    arr = arr[y0:y1 + 1, x0:x1 + 1]

    # Resize to fit within 20x20 while keeping aspect ratio
    img = Image.fromarray(arr)
    weight, height = img.size
    scale = 20.0 / max(weight, height)                  
    new_weight, new_height = int(weight * scale), int(height * scale)
    img = img.resize((new_weight, new_height), Image.LANCZOS)

    # Center on a 28x28 black background
    new_img = Image.new("L", (28, 28), color=0)
    left = (28 - new_weight) // 2
    top = (28 - new_height) // 2
    new_img.paste(img, (left, top))

    arr = np.array(new_img).astype("float32") / 255.0     
    return arr



# ->> Load Dataset
def load_custom_dataset():                  

    X, y = [], []                                       # Arrays for images and labels
    for label in range(10):
        folder = os.path.join(DATASET_DIR, str(label))  # joint helps to create path
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            X.append(preprocess_image(path))            # preprocess and add image
            y.append(label)                             # add corresponding label

    if len(X) == 0:                                     # Check if dataset is empty 
        print("No samples found! Collect some data first.")
        return np.empty((0, 28, 28)), np.empty((0,))    # Return empty arrays

    print(f"Loaded {len(X)} samples from My_Dataset.")
    return np.array(X), np.array(y)                     # Else we return the dataset



# ->> Build Convolutional Neural Network model
def build_cnn():          
    
    model = Sequential([                      
        Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),  # first CNN layer
        MaxPooling2D(2,2),                                            # first pooling layer
        Conv2D(64, (3,3), activation="relu"),                         # second conv layer
        MaxPooling2D(2,2),                                            # second pooling layer
        Flatten(),                                                    # flatten to 1D
        Dense(128, activation="relu"),         # Using Relu activation func to introduce non-linearity
        Dropout(0.3),                          # dropout for regularization
        Dense(10, activation="softmax")        # Useges softmax for multi-class classification
    ])
    model.compile(optimizer=Adam(0.001),       # Adam optimizer bacause it will adjust learning rate during training
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model                               



# ->> User input and also Data entry using Pygame Drawing
def draw_digit():
    pygame.init()                                  # Initialize Pygame
    window = pygame.display.set_mode((280, 280))   # instantiate a window with specified size
    pygame.display.set_caption("Draw a digit (Enter=save, Esc=quit)")
    window.fill((0, 0, 0))
    drawing = False

    while True:                                 # Loops handles drawing, image saving
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if e.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    pygame.image.save(window, "user.png")    
                    pygame.quit()
                    return "user.png"
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); exit()
        if drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(window, (255, 255, 255), (x, y), 10)
        pygame.display.update()



# ->> Main Program that handle modes: data collection, training, prediction
if __name__ == "__main__":
    print("\n[1] Collect data  [2] Train model  [3] Predict")
    mode = input("Choose mode (1/2/3): ").strip()

    # ->> Collect data 
    if mode == "1":
        label = input("Enter the digit (0–9): ")                      # Collect label from user 
        os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)  # makdir if not exists and handle error if exists.
        count = len(os.listdir(os.path.join(DATASET_DIR, label)))     # count existing images and prevent image overwrite
        path = os.path.join(DATASET_DIR, label, f"img_{count+1}.png") # we create file path
        img = draw_digit()
        os.replace(img, path)                                         # Move new image to dataset folder
        print(f"Saved: {path}")                                    

    # ->> Train model 
    elif mode == "2":
        X, y = load_custom_dataset()         # Load all images and labels into numpy arrays
        if len(X) == 0:
            print("No data found! Collect samples first.")
            exit()
        X = X.reshape(-1, 28, 28, 1)        # Reshape the image data for CNN input
        y = to_categorical(y, 10)           # Convert labels (0–9) into one-hot encoded vectors

        model = build_cnn()                 # Build the CNN model                    
        print("Training CNN on your dataset...")
        # Train with 20 epochs, 
        # Small batch size so model updates weights more frequently to reduce overfitting
        # Less validation so more data for training
        # Verbose for progress output
        model.fit(X, y, epochs=20, batch_size=8, validation_split=0.1, verbose=1) 
        model.save("custom_digit_cnn.keras")       # Save the trained model      
        print("Model saved as custom_digit_cnn.keras")

    # ->> Predict drawn digit 
    elif mode == "3":
        if not os.path.exists("custom_digit_cnn.keras"):   # Check if trained model file exists 
            print("Please Train your model first! :') -> Choose Option 2")  # If not then ask
            exit()

        model = load_model("custom_digit_cnn.keras") # make sure the model load befor pygame strarts
        img_path = draw_digit()      # Open pygame drawing interface
        arr = preprocess_image(img_path).reshape(1, 28, 28, 1)  # Preprocess the drawn image
        # Pass the image through CNN to get prediction probabilities
        pred = np.argmax(model.predict(arr))                  
        print(f"The Predicted Digit: {pred}")
