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

### Degradation Corpus

The project includes a structured degradation corpus generator for training with
more normalized shape variety and richer mechanism metadata:

```python
from src.generation import (
    CorpusConfig,
    corpus_metadata_summary,
    corpus_diagnostics_report,
    degradation_mechanism_family_tree,
    degradation_shape_diagnostics,
    generate_degradation_corpus_from_config,
)

config = CorpusConfig(
    episode_length=100,
    n_episodes=5000,
    seed=42,
    source_weights={"battery": 0.3, "fatigue": 0.2, "corrosion": 0.2},
)

episodes, metadata = generate_degradation_corpus_from_config(config)
print(degradation_shape_diagnostics(episodes))
print(corpus_metadata_summary(metadata))
print(degradation_mechanism_family_tree())
print(corpus_diagnostics_report(episodes, metadata, context_window=60, future_window=60))
```

This generator samples from a mechanism registry with a growing family tree:
battery SEI growth, cycle aging, lithium-plating knees, and mixed capacity fade;
fatigue crack growth, threshold incubation, and overload retardation; creep
deformation and rupture acceleration; corrosion pitting, uniform corrosion, and
passivation/repassivation; and wear transitions, abrasive wear, and lubrication
failure. The old
`generate_diverse_degradation_episodes` wrapper remains available for existing
notebooks and scripts.

### Real Data Ingestion

Real degradation sources can be loaded into the same `(episodes, metadata)`
format and combined with synthetic corpora:

```python
from src.ingestion import CSVDatasetSource, combine_degradation_corpora

real_source = CSVDatasetSource(
    name="nasa_battery_local",
    domain="battery",
    observed_variable="capacity",
    mechanism_family="capacity_fade_lab_cycles",
    parent_mechanism="battery_capacity_fade",
    path="data/nasa_battery_capacity.csv",
    value_column="capacity",
    id_column="cell_id",
    time_column="cycle",
    direction="decreasing",
    episode_length=100,
)

real_episodes, real_metadata = real_source.load()
episodes, metadata = combine_degradation_corpora(
    (episodes, metadata),
    (real_episodes, real_metadata),
)
```

## 📂 Project Structure

-   `app.py`: The Gradio web application.
-   `src/generation.py`: Synthetic degradation processes and diversity diagnostics.
-   `src/ingestion.py`: Real/local degradation dataset ingestion and corpus mixing.
-   `src/preprocessing.py`: Normalization, digitization, metadata features, and datasets.
-   `src/model.py`: Transformer architecture.
-   `src/learner.py`: Supervised training and autoregressive prediction loop.
-   `src/evaluation.py`: Forecast metrics over sliding context/future windows.
-   `src/rlhf.py`: RLHF datasets, rewards, and learners.
-   `src/callbacks.py`: Training callbacks and experiment logging.
-   `main.ipynb`: Notebook for initial experiments and pre-training.
-   `rlhf_training.ipynb`: Notebook for RLHF fine-tuning.
-   `RLHF_PLAN.md`: Detailed plan and theoretical background for the RLHF approach.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
