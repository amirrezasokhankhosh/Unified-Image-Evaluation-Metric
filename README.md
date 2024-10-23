# CIFAR-10 Generative Model Evaluation with Diffusion and Custom Metrics

This repository evaluates generative models using diffusion models on the CIFAR-10 dataset, with custom metrics such as Precision, Recall, Generalization Rate, and F1 Score. The evaluation involves using a pre-trained ResNet-50 model as a classifier to extract features from both real and generated images, and then computing the metrics based on nearest-neighbor searches.

## File Structure

- **`main.py`**:
  - Utilizes a **ResNet-50** model pre-trained on CIFAR-10 to extract features from real and generated images.
  - **Diffusion model** is used for generating images.
  - Computes custom metrics: **Precision**, **Recall**, **Generalization Rate**, and **F1 Score**.
  
- **`precision.py`**, **`recall.py`**, **`generalization.py`** (if separated):
  - Implements the metric calculations for precision, recall, and generalization using nearest-neighbors.

## Features

1. **Diffusion Model**:
   - The `Diffusion` model generates images from noise using pre-trained weights on the CIFAR-10 dataset.
   
2. **ResNet-50 Classifier**:
   - A pre-trained ResNet-50 model is fine-tuned for the CIFAR-10 classification task, modified to extract features from images for evaluation purposes.

3. **Custom Metrics**:
   - **Precision**: Measures how well the generated samples align with real training samples using nearest neighbors.
   - **Recall**: Measures how well the generated samples cover the diversity of the real dataset.
   - **Generalization Rate**: A custom metric that assesses how well the model generalizes to unseen test data.
   - **F1 Score**: Combines precision, recall, and generalization rate into a single metric.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Download the CIFAR-10 dataset:
    - The CIFAR-10 dataset is automatically downloaded using torchvision.datasets.CIFAR10.

4. Place the pre-trained ResNet-50 CIFAR-10 classifier model (CIFAR10-Classifier.pth) in the root folder or update the path in the script.

5. Run the evaluation:

    ```bash
    python main.py

## Metrics Output

The script will output the following metrics:

- **Precision**: How well the generated samples align with real training samples.
- **Recall**: How well the generated samples cover the diversity of the real dataset.
- **Generalization Rate**: A custom metric indicating how well the model generalizes to unseen test data.
- **UIEM**: A combined score using precision, recall, and generalization.

## Dependencies
- `torch`: For building and running the generator and classifier models.
- `torchvision`: For datasets and model utilities.
- `scikit-learn`: For nearest neighbors calculations in precision, recall, and generalization metrics.
- `numpy`: General numerical operations.