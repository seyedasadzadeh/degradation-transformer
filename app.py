import json
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model

from src.utils import DegradationTransformer, Learner

MODEL_FILENAME = "degradation_transformer_model.safetensors"
CONFIG_FILENAME = "degradation_transformer_model_config.json"
HF_REPO_ID = "smasadzadeh/degradation-transformer"
WANDB_ARTIFACT = "smasadzadeh-freelancer/degradation-transformer/degradation-transformer-model:Production"

MAX_BATCH_SIZE = 256
MAX_SEQUENCE_LENGTH = 5000
MIN_SEQUENCE_LENGTH = 2

learner = None


def _load_model_from_files(model_weights_path, config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        model_params = json.load(f)
    model = DegradationTransformer(**model_params)
    load_model(model, model_weights_path)
    return model


def get_model_from_hf_repo(
    safetensor_filename=MODEL_FILENAME,
    config_filename=CONFIG_FILENAME,
):
    model_weights = hf_hub_download(repo_id=HF_REPO_ID, filename=safetensor_filename)
    config_path = hf_hub_download(repo_id=HF_REPO_ID, filename=config_filename)
    return _load_model_from_files(model_weights, config_path)


def get_model_from_wandb(
    safetensor_filename=MODEL_FILENAME,
    config_filename=CONFIG_FILENAME,
):
    import wandb

    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    artifact = api.artifact(WANDB_ARTIFACT)
    artifact_dir = artifact.download(root="/tmp/wandb_artifacts")
    model_weights_path = os.path.join(artifact_dir, safetensor_filename)
    config_path = os.path.join(artifact_dir, config_filename)
    return _load_model_from_files(model_weights_path, config_path)


def get_model_local(
    safetensor_filename=MODEL_FILENAME,
    config_filename=CONFIG_FILENAME,
):
    if not os.path.exists(safetensor_filename):
        raise FileNotFoundError(f"Missing local weights file: {safetensor_filename}")
    if not os.path.exists(config_filename):
        raise FileNotFoundError(f"Missing local config file: {config_filename}")
    return _load_model_from_files(safetensor_filename, config_filename)


def load_model_with_fallback():
    errors = []
    for source_name, loader in (
        ("local files", get_model_local),
        ("Hugging Face", get_model_from_hf_repo),
        ("Weights & Biases", get_model_from_wandb),
    ):
        try:
            return loader()
        except Exception as e:
            errors.append(f"{source_name}: {e}")
    raise RuntimeError("Unable to load model from any source.\n" + "\n".join(errors))


def get_learner():
    global learner
    if learner is None:
        model = load_model_with_fallback()
        learner = Learner(model, optim=None, loss_func=None, train_loader=None, test_loader=None, cbs=[])
    return learner


def validate_input_data(data):
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim not in (1, 2):
        raise ValueError("Input must be 1D or 2D numeric data.")
    if arr.size == 0:
        raise ValueError("Input is empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or infinite values.")
    if arr.shape[-1] < MIN_SEQUENCE_LENGTH:
        raise ValueError(f"Each sequence must contain at least {MIN_SEQUENCE_LENGTH} values.")
    if arr.shape[-1] > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence is too long. Maximum supported length is {MAX_SEQUENCE_LENGTH}.")
    if arr.ndim == 2 and arr.shape[0] > MAX_BATCH_SIZE:
        raise ValueError(f"Too many sequences. Maximum supported batch size is {MAX_BATCH_SIZE}.")
    return arr


def parse_text_input(text_input):
    values = [x.strip() for x in text_input.split(",")]
    values = [x for x in values if x]
    if not values:
        raise ValueError("No numeric values were provided in text input.")
    return np.array([float(x) for x in values], dtype=np.float32)


def predict_and_plot(input_data, num_periods=60):
    input_data = validate_input_data(input_data)
    model_learner = get_learner()
    y_predict, _ = model_learner.predict(input_data, num_periods=int(num_periods))

    fig, ax = plt.subplots(figsize=(12, 6))
    input_len = input_data.shape[-1]
    pred_len = y_predict.shape[-1]

    ax.plot(range(input_len), input_data.T, "b-", label="Original", linewidth=2)
    ax.plot(range(input_len, input_len + pred_len), y_predict.T, "r--", label="Predicted", linewidth=2)
    ax.axvline(x=input_len, color="gray", linestyle=":", alpha=0.7)

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Degradation Value")
    ax.set_title("Degradation Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def process_input(file, text_input, num_periods):
    try:
        if file is not None:
            data = np.load(file.name, allow_pickle=False)
        elif text_input:
            data = parse_text_input(text_input)
        else:
            raise ValueError("Provide either a .npy file or comma-separated numeric values.")

        return predict_and_plot(data, num_periods)
    except Exception as e:
        raise gr.Error(str(e))


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Degradation Prediction App")
        with gr.Column():
            with gr.Row():
                data_file = gr.File(file_count="single", label="Input .npy file shape=n, context_window")
                data_text = gr.Text(
                    label="Input comma-separated numbers (recommended context length: 40)",
                    placeholder="e.g. 1, 2, 3, ...",
                )
            with gr.Row():
                num_periods = gr.Slider(value=60, minimum=1, maximum=300, step=1, label="Number of periods to predict")
            with gr.Row():
                predict_button = gr.Button("Predict Degradation")
            with gr.Row():
                output_plot = gr.Plot(label="Degradation Prediction Plot")
                predict_button.click(fn=process_input, inputs=[data_file, data_text, num_periods], outputs=output_plot)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
