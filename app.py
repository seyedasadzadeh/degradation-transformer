import gradio as gr
import numpy as np
import torch
import json
from huggingface_hub import hf_hub_download


from src.utils import DegradationTransformer, Learner
from safetensors.torch import load_model
import json

import matplotlib.pyplot as plt

def predict_and_plot(input_data, num_periods=60):
    # Convert input to numpy array
    input_data = np.array(input_data) # shape context_window or n, context_window
    
    # Run prediction
    y_predict = learner.predict(input_data, num_periods=num_periods) # shape n, context_window + num_periods
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original data
    ax.plot(range(input_data.shape[-1]), input_data.T, 'b-', label='Original', linewidth=2)
    
    # Plot predicted data (continues from original)
    ax.plot(range(y_predict.shape[-1]), y_predict.T, 'r--', label='Predicted', linewidth=2)
    
    # Add vertical line to show where prediction starts
    ax.axvline(x=len(input_data), color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Degradation Value')
    ax.set_title('Degradation Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def process_input(file, text_input, num_periods):
    if file is not None:
        data = np.load(file.name)
    elif text_input:
        data = np.array([float(x.strip()) for x in text_input.split(',')])
    else:
        return None  # No input provided
    
    return predict_and_plot(data, num_periods)

# Download model and config
def get_model_from_safetensors_and_config(safetensor_filename="degradation_transformer_model.safetensors",
                                           config_filename="degradation_transformer_model_config.json"):
    model_path = hf_hub_download(
        repo_id="smasadzadeh/degradation-transformer",
        filename=safetensor_filename
    )
    config_path = hf_hub_download(
        repo_id="smasadzadeh/degradation-transformer",
        filename=config_filename
    )

    # Load config and create model
    model_params = json.load(open(config_path))
    model = DegradationTransformer(**model_params)
    load_model(model, model_path)
    return model

# Create minimal learner
model = get_model_from_safetensors_and_config()
learner = Learner(model, optim=None, loss_func=None, 
                  train_loader=None, test_loader=None, cbs=[])


with gr.Blocks() as demo:
    gr.Markdown("# Degradation Prediction App")
    with gr.Column():

        with gr.Row():
            # Input options here
            data_file = gr.File(file_count='single', label='Input .npy file shape=n, context_window')
            data_text = gr.Text(label='input comma seperated numbers with len context window', placeholder='e.g. 1, 2, 3, ..., ')
        with gr.Row():   
            num_periods = gr.Slider(value=60, label='Number of periods to predict', precision=0)

        with gr.Row():
        
            # Button and output here
            predict_button = gr.Button("Predict Degradation")
        with gr.Row():
            output_plot = gr.Plot(label='Degradation Prediction Plot')
            predict_button.click(fn=process_input, 
                                inputs=[data_file, data_text, num_periods], 
                                outputs=output_plot)
    
demo.launch(server_name="0.0.0.0", server_port=7860)

