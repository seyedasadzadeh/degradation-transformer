# Degradation Transformer

A Deep Learning model for predicting degradation trajectories in time-series data using Transformers and Reinforcement Learning from Human Feedback (RLHF).

## 🚀 Hugging Face Demo

You can try out the model directly in your browser on Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/smasadzadeh/degradation-transformer-app)


## 📖 Overview

This project implements a Transformer-based model designed to forecast the future degradation path of a system based on a historical context window. To address the issue of error accumulation in long-term autoregressive generation, the model is fine-tuned using a Reinforcement Learning (RLHF) approach, optimizing for accurate long-term trajectories rather than just single-step token prediction.

## ✨ Features

- **Transformer Architecture**: Robust time-series forecasting.
- **RLHF Fine-tuning**: Improved long-horizon prediction accuracy.
- **Interactive Web App**: Built with Gradio for easy model interaction.
- **Docker Support**: Containerized environment for reproducibility.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/degradation-transformer.git
    cd degradation-transformer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements-app.txt
    ```

## 💻 Usage

### Running the Web App Locally

To launch the Gradio interface locally:

```bash
python app.py
```

The app will start at `http://0.0.0.0:7860`.

### Using Docker

You can also run the app using Docker:

```bash
docker build -f Dockerfile.app -t degradation-app .
docker run -p 7860:7860 degradation-app
```

### Training

-   **Pre-training**: Check `main.ipynb` for the initial training loop.
-   **RLHF Fine-tuning**: See `rlhf_training.ipynb` and `RLHF_PLAN.md` for the reinforcement learning implementation details.

## 📂 Project Structure

-   `app.py`: The Gradio web application.
-   `src/`: Core source code for the model and utilities.
-   `main.ipynb`: Notebook for initial experiments and pre-training.
-   `rlhf_training.ipynb`: Notebook for RLHF fine-tuning.
-   `RLHF_PLAN.md`: Detailed plan and theoretical background for the RLHF approach.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
