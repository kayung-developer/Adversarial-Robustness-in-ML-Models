import os
import urllib.request
import tkinter as tk
import customtkinter as ctk
import tensorflow as tf
import tf_keras
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# CustomTkinter Application Setup
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class AdversarialRobustnessApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Adversarial Robustness in ML Models")
        self.geometry("1200x800")

        # Variables
        self.model = None
        self.dataset = None
        self.attack_type = tk.StringVar(value="FGSM")
        self.defense_type = tk.StringVar(value="None")
        self.epsilon = tk.DoubleVar(value=0.1)

        # Layout
        self.create_widgets()

    def create_widgets(self):
        # Sidebar
        global content
        sidebar = ctk.CTkFrame(self, width=300)
        sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(sidebar, text="Adversarial Robustness", font=("Arial", 20)).pack(pady=20)

        # Model Selection
        ctk.CTkLabel(sidebar, text="Select Model").pack(pady=10)
        self.model_button = ctk.CTkButton(sidebar, text="Load Model", command=self.load_model)
        self.model_button.pack(pady=5)

        # Attack Configuration
        ctk.CTkLabel(sidebar, text="Adversarial Attack").pack(pady=10)
        attack_menu = ctk.CTkOptionMenu(sidebar, variable=self.attack_type, values=["FGSM", "PGD", "CW"])
        attack_menu.pack(pady=5)

        ctk.CTkLabel(sidebar, text="Epsilon (Perturbation Strength)").pack(pady=10)
        epsilon_slider = ctk.CTkSlider(sidebar, from_=0.0, to=1.0, variable=self.epsilon)
        epsilon_slider.pack(pady=5)

        # Defense Configuration
        ctk.CTkLabel(sidebar, text="Defense Mechanism").pack(pady=10)
        defense_menu = ctk.CTkOptionMenu(sidebar, variable=self.defense_type,
                                         values=["None", "Adversarial Training", "Gradient Masking",
                                                 "Defensive Distillation"])
        defense_menu.pack(pady=5)

        # Run Button
        run_button = ctk.CTkButton(sidebar, text="Run Attack", command=self.run_attack)
        run_button.pack(pady=20)

        # Main Content
        content = ctk.CTkFrame(self)
        content.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Canvas for Plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, content)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Analytics Panel
        self.analytics_panel = ctk.CTkLabel(content, text="Metrics: \n", font=("Arial", 14))
        self.analytics_panel.pack(pady=10)

    def load_model(self):
        """Load MobileNetV2 model pre-trained on ImageNet."""
        try:
            self.model = tf_keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
            ctk.CTkLabel(self, text="Model loaded successfully!", fg_color="green").place(x=500, y=10)
        except Exception as e:
            ctk.CTkLabel(self, text=f"Error loading model: {e}", fg_color="red").place(x=500, y=10)

    def run_attack(self):
        """Run adversarial attack and display results."""
        if self.model is None:
            ctk.CTkLabel(content, text="Load a model first!", fg_color="red").place(x=10, y=10)
            return

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf_keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255.0  # Normalize pixel values to [0, 1]
        x_test = np.expand_dims(x_test, -1)  # Add channel dimension (for grayscale)

        # Resize images to MobileNetV2 input size and convert grayscale to RGB
        x_test_resized = tf.image.resize(x_test, (224, 224))  # Resize to 224x224
        x_test_resized = np.repeat(x_test_resized, 3, axis=-1)  # Convert grayscale to RGB

        # Select an image
        image = x_test_resized[0:1]  # Select the first image and add batch dimension
        true_label = y_test[0]

        # Generate Adversarial Example
        if self.attack_type.get() == "FGSM":
            adversarial_image = self.fgsm_attack(image, self.epsilon.get())
        else:
            adversarial_image = image  # Placeholder for other attacks

        # Plot original and adversarial images
        self.ax.clear()
        self.ax.set_title("Original vs Adversarial")
        self.ax.imshow(np.hstack([image[0], adversarial_image[0]]))  # Display side-by-side
        self.canvas.draw()

        # Update metrics
        self.analytics_panel.configure(
            text=f"Metrics:\nEpsilon: {self.epsilon.get()}\nTrue Label: {true_label}\n"
        )

    def fgsm_attack(self, image, epsilon):
        """Generate adversarial example using FGSM."""
        image_tensor = tf.convert_to_tensor(image)
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            prediction = self.model(image_tensor)
            loss = tf_keras.losses.SparseCategoricalCrossentropy()(tf.convert_to_tensor([7]),
                                                                   prediction)  # Example target

        gradient = tape.gradient(loss, image_tensor)
        adversarial_image = image + epsilon * tf.sign(gradient)
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
        return adversarial_image.numpy()


# Run the application
if __name__ == "__main__":
    app = AdversarialRobustnessApp()
    app.mainloop()
