Handwritten Digit Recognition using SVM

This is a simple coursework project where I built a handwritten digit recognition system using a Support Vector Machine (SVM) and the scikit-learn digits dataset.

What my project does:

Loads the built-in digits dataset (1797 images, each 8×8 pixels).

Prepares the data by flattening images into feature vectors.

Splits the data into 75% training and 25% testing.

Trains an SVM classifier to recognise digits 0–9.

Shows the accuracy and full classification report.

Allows the user to choose an index number to display one test image along with:

the true label,

the predicted label,

and the image itself.

How to run the project:

Install dependencies:

pip install -r requirements.txt


Run the main script:

python Main_SVM.py

Requirements:

See requirements.txt.