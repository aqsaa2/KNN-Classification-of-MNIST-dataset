import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

def apply_feature_engineering(images):
    processed_images = []
    for image in images:
        edges = cv2.Canny(image, 100, 200)
        equalized = cv2.equalizeHist(image)
        processed_image = np.concatenate((image, edges, equalized), axis=1)
        processed_images.append(processed_image)
    return np.array(processed_images)


def preprocess_image(image):
    image_gray = image.convert('L')
    image_resized = image_gray.resize((28, 28))
    image_np = np.array(image_resized)
    image_inv = 255 - image_np
    _, image_thresh = cv2.threshold(image_inv, 128, 255, cv2.THRESH_BINARY)
    image_normalized = image_thresh.astype(np.float32) / 255.0
    image_reshaped = image_normalized.reshape(1, -1)
    return image_reshaped


#dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train_processed = apply_feature_engineering(X_train)
X_test_processed = apply_feature_engineering(X_test)

X_train_processed = X_train_processed.reshape(X_train_processed.shape[0], -1) / 255.0
X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], -1) / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


class ImageClassifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier")
        self.geometry("300x150")

        self.process_button = tk.Button(self, text="Classify Images", command=self.process_images)
        self.process_button.pack(pady=20)

    def process_images(self):
        y_pred = knn.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print("Validation Accuracy:", accuracy)

        y_pred_test = knn.predict(X_test_processed)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        print("Test Accuracy:", accuracy_test)

        num_samples = 5
        sample_indices = np.random.randint(0, X_test.shape[0], num_samples)

        for i, idx in enumerate(sample_indices):
            original_image = X_test[idx, :784].reshape(28, 28).astype(np.uint8)
            processed_image = (X_test_processed[idx, :784] * 255).reshape(28, -1).astype(np.uint8)
            label = y_pred_test[idx]

            # Converting images 
            original_pil = Image.fromarray(original_image)
            processed_pil = Image.fromarray(processed_image)

            # Saving images
            original_pil.save(f"original_image_{i}.png")
            processed_pil.save(f"processed_image_{i}.png")

            print(f"Image {i + 1} - Label: {label}")

            # Display 
            self.display_image(original_pil, f"Original Image {i + 1}")
            self.display_image(processed_pil, f"Processed Image {i + 1}")
            self.display_label(label)

    def display_image(self, image, title):
        photo = ImageTk.PhotoImage(image)

        label = tk.Label(self, image=photo)
        label.image = photo  
        label.pack()

   
        title_label = tk.Label(self, text=title)
        title_label.pack()

    def display_label(self, label):
        label_text = f"Predicted Label: {label}"
        label_result = tk.Label(self, text=label_text)
        label_result.pack()


if __name__ == "__main__":
    app = ImageClassifierGUI()
    app.mainloop()
