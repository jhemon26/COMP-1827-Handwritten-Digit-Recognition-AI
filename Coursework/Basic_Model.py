# ==========================================
# Simple Neural Network (from scratch)
# Custom Handwritten Digit Dataset
# + Pygame Drawing Input
# ==========================================

import numpy as np
import pygame
import os
from PIL import Image, ImageOps

# ---------------------------
# 1. Create and Label Dataset
# ---------------------------
DATASET_DIR = "My_Dataset"

def save_drawn_digit(label):
    """Draw a digit and save it under my_dataset/label/"""
    pygame.init()
    win = pygame.display.set_mode((280, 280))
    pygame.display.set_caption(f"Draw digit {label} - Press Enter to save")
    win.fill((0, 0, 0))
    drawing = False

    os.makedirs(os.path.join(DATASET_DIR, str(label)), exist_ok=True)

    print(f"\n🎨 Draw the digit '{label}' and press Enter to save it.")
    count = len(os.listdir(os.path.join(DATASET_DIR, str(label))))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    img_name = f"img_{count+1}.png"
                    img_path = os.path.join(DATASET_DIR, str(label), img_name)
                    pygame.image.save(win, img_path)
                    print(f"✅ Saved as {img_path}")
                    pygame.quit()
                    return
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit(); exit()

        if drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(win, (255, 255, 255), (x, y), 10)

        pygame.display.update()

# ---------------------------
# 2. Load Dataset into Arrays
# ---------------------------
def load_dataset():
    """Load images and labels from the my_dataset folder"""
    X, y = [], []
    for label in range(10):
        folder = os.path.join(DATASET_DIR, str(label))
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = Image.open(path).convert('L')
            img = ImageOps.invert(img)
            img = img.resize((28, 28))
            X.append(np.array(img).flatten() / 255.0)
            y.append(label)
    print(f"📦 Loaded {len(X)} images from dataset.")
    return np.array(X), np.array(y)

# ---------------------------
# 3. Simple Neural Network (NumPy)
# ---------------------------
class SimpleNN:
    def __init__(self, input_size=784, hidden_size=64, output_size=10, lr=0.1):
        self.lr = lr
        np.random.seed(1)
        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(self, x): return x * (1 - x)

    def one_hot(self, y, num_classes=10):
        onehot = np.zeros((y.size, num_classes))
        onehot[np.arange(y.size), y] = 1
        return onehot

    def train(self, X, y, epochs=10):
        y_oh = self.one_hot(y)
        for epoch in range(epochs):
            acc = 0
            for i, (x, target) in enumerate(zip(X, y_oh)):
                x = x.reshape(-1, 1)
                target = target.reshape(-1, 1)

                # Forward pass
                h = self.sigmoid(np.dot(self.W1, x) + self.b1)
                o = self.sigmoid(np.dot(self.W2, h) + self.b2)

                # Backprop
                delta_o = (o - target) * self.sigmoid_deriv(o)
                delta_h = np.dot(self.W2.T, delta_o) * self.sigmoid_deriv(h)

                # Update
                self.W2 -= self.lr * np.dot(delta_o, h.T)
                self.b2 -= self.lr * delta_o
                self.W1 -= self.lr * np.dot(delta_h, x.T)
                self.b1 -= self.lr * delta_h

                if np.argmax(o) == np.argmax(target):
                    acc += 1

            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {acc/len(X):.2f}")

    def predict(self, x):
        x = x.reshape(-1, 1)
        h = self.sigmoid(np.dot(self.W1, x) + self.b1)
        o = self.sigmoid(np.dot(self.W2, h) + self.b2)
        return np.argmax(o)

# ---------------------------
# 4. Helper: prepare new drawing
# ---------------------------
def prepare_image(img_path):
    img = Image.open(img_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    return np.array(img).flatten() / 255.0

# ---------------------------
# 5. Pygame drawing for prediction
# ---------------------------
def draw_for_prediction():
    pygame.init()
    win = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Draw a digit to predict")
    win.fill((0, 0, 0))
    drawing = False

    print("\n🎨 Draw any digit and press Enter to predict.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.image.save(win, "to_predict.png")
                    pygame.quit()
                    return "to_predict.png"
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit(); exit()

        if drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(win, (255, 255, 255), (x, y), 10)
        pygame.display.update()

# ---------------------------
# 6. Main Program
# ---------------------------
if __name__ == "__main__":
    print("\n1️⃣ Collect data  2️⃣ Train model  3️⃣ Predict")
    choice = input("Choose mode (1/2/3): ")

    if choice == '1':
        label = input("Enter the digit (0-9) you want to draw: ")
        save_drawn_digit(label)

    elif choice == '2':
        X, y = load_dataset()
        if len(X) == 0:
            print("⚠️ No dataset found! Collect some samples first.")
        else:
            nn = SimpleNN()
            nn.train(X, y, epochs=10)
            np.savez("model_weights.npz", W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)
            print("✅ Model saved as model_weights.npz")

    elif choice == '3':
        if not os.path.exists("model_weights.npz"):
            print("⚠️ Train a model first!")
        else:
            data = np.load("model_weights.npz")
            nn = SimpleNN()
            nn.W1, nn.b1, nn.W2, nn.b2 = data["W1"], data["b1"], data["W2"], data["b2"]

            img_path = draw_for_prediction()
            img_arr = prepare_image(img_path)
            pred = nn.predict(img_arr)
            print(f"🧠 Predicted Digit: {pred}")
