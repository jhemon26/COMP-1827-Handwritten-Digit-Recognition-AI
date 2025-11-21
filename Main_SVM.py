#Name; Amanuel Adameseged
#STD; 001398810

# Project: Handwritten Digit Recognition using Support Vector Machine (SVM)
# Method I Used: I used the built-in scikit-learn digits dataset (8x8 images)
# and trained an SVM classifier to recognise digits from 0 to 9.
# System Process: First, I load and split the dataset into training and test sets.
# Then I train an SVM model and evaluate its accuracy.
# I also allow the user to enter an index number to view a test image
# and see the predicted digit for that sample.
"""
Main_SVM.py
Handwritten Digit Recognition using Support Vector Machine (SVM)

What this script does:
1. Loads the built-in digits dataset from scikit-learn (1797 images, size 8x8).
2. Prepares the data (flattens each image to a vector of length 64).
3. Splits the data into training and test sets.
4. Trains an SVM classifier.
5. Evaluates accuracy on the test set.
6. Lets the user choose one test image and shows:
   - the true label
   - the predicted label
   - the image itself.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def main():

    # === 1. LOAD DATASET ====================================================
    # I load the digits dataset. It contains 1797 handwritten digits.
    digits = load_digits()
    images = digits.images       # shape: (1797, 8, 8)
    labels = digits.target       # shape: (1797,)

    print("=== DATASET INFO ===")
    print("Number of samples:", len(images))
    print("Image shape:", images[0].shape)
    print("Classes (digits):", np.unique(labels))
    print()

    # 2. PREPARE FEATURES (X) AND LABELS (y)
    # SVM expects a 2D array: [n_samples, n_features]
    # Each 8x8 image = 64 pixels -> flatten to 1D of length 64
    X = images.reshape((len(images), -1))   # (1797, 64)
    y = labels

    # === 3. TRAIN/TEST SPLIT ================================================
    # Use 75% for training, 25% for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    print("=== SPLIT INFO ===")
    print("Train set size:", len(X_train))
    print("Test set size:", len(X_test))
    print()

    # === 4. TRAIN SVM MODEL =================================================
    print("Training SVM model...")
    # RBF kernel works very well on this dataset.
    model = SVC(kernel="rbf", gamma="scale", C=10)
    model.fit(X_train, y_train)
    print()

    # === 5. EVALUATION ======================================================
    print("=== EVALUATION ===")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy on test set: {acc * 100:.2f}%")
    print()

    print("Classification report:\n")
    print(classification_report(y_test, predictions))
    print()

    # === 6. EXAMPLE PREDICTION =============================================
    # The user chooses an index from the test set.
    # If they just press Enter or type something invalid, we use 0.
    print("\n=== EXAMPLE PREDICTION ===")
    user_input = input("Enter an index from 0 to 449 to visualise a test digit (press Enter for 0): ")

    if user_input.strip() == "":
        index = 0
    else:
        try:
            index = int(user_input)
        except ValueError:
            print("Invalid input.")
            return

    # VALIDATE INDEX
    if not (0 <= index < len(y_test)):
        print("Invalid index. Please choose a number between 0 and 449.")
        return

    # Get test sample
    test_index = index
    sample = X_test[test_index]
    true_label = y_test[test_index]
    pred = model.predict([sample])[0]

    print(f"\nYou chose index: {index}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred}")

    # ------ SHOW THE IMAGE ------
    plt.imshow(digits.images[test_index], cmap='gray')
    plt.title(f"True: {true_label} | Predicted: {pred}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
