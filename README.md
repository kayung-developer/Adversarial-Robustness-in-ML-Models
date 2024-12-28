# Adversarial Robustness in ML Models

This project is an advanced Python application built using CustomTkinter. It provides a user-friendly interface to demonstrate defense mechanisms against adversarial attacks in machine learning models. The application allows users to:

- Load pre-trained models.
- Simulate adversarial attacks such as FGSM, PGD, and CW.
- Apply defense mechanisms like adversarial training, gradient masking, and defensive distillation.
- Visualize and analyze the impact of these attacks and defenses interactively.

## Features

- **User-Friendly Interface**: Modern and intuitive UI built with CustomTkinter.
- **Adversarial Attacks**: Supports FGSM, PGD, and CW attack simulations.
- **Defense Mechanisms**: Includes popular defenses such as adversarial training and defensive distillation.
- **Visualization**: Real-time visualization of adversarial examples and their metrics.
- **Extensibility**: Easily extendable for custom attacks and defenses.

## Prerequisites

Ensure the following dependencies are installed before running the application:

- Python 3.7+
- CustomTkinter
- TensorFlow
- Matplotlib
- Numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kayung-developer/Adversarial-Robustness-in-ML-Models.git
   cd Adversarial-Robustness-in-ML-Models
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a pre-trained TensorFlow model (`model.h5`) in the working directory or update the `load_model` method with your model path.

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Use the sidebar to:
   - Load a pre-trained model.
   - Select the attack type (FGSM, PGD, CW).
   - Adjust the attack strength (Epsilon).
   - Choose a defense mechanism.

3. Click "Run Attack" to visualize the results and view metrics.

## File Structure

```
.
├── app.py                # Main application code
├── requirements.txt      # Dependencies
├── model.h5              # Placeholder for pre-trained model
├── README.md             # Project documentation
```

## Requirements File

```
customtkinter
matplotlib
numpy
tensorflow
```
## How It Works
1. Loading a Model
The application uses a pre-trained Keras model. When you click the "Load Model" button, it will download a model if it's not already present. The current implementation uses a sample model URL, but you can easily change it to load different models.

2. Configuring Adversarial Attacks
You can select from a variety of adversarial attack methods:

- FGSM (Fast Gradient Sign Method): A popular attack that creates perturbations to mislead the model.
- PGD (Projected Gradient Descent): A more advanced iterative attack.
- CW (Carlini-Wagner): An attack based on optimization techniques.
- The attack's strength can be controlled with the Epsilon slider.

3. Applying Defenses
- You can choose a defense mechanism to protect the model against adversarial attacks. The available defenses are:

- None: No defense applied.
- Adversarial Training: Trains the model with adversarial examples.
- Gradient Masking: Obscures the gradient used to generate adversarial examples.
- Defensive Distillation: Uses distillation to reduce the model's vulnerability to attacks.
4. Visualizing Results
- Once you configure the attack and defense settings, clicking the "Run Attack" button will execute the attack on an image from the MNIST dataset. The original and adversarial images will be displayed side by side. The app also updates the metrics, showing the current attack settings and model performance.



## Contributions

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Screenshots

_Add screenshots of the application interface here._
[Screenshots](screenshot.png)

---

For any queries or issues, please contact [princelillwitty@gmail.com].

