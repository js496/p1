import gradio as gr
import requests
import docker
import os
from vllm import LLM

client = docker.from_env()

def search_models(query):
    response = requests.get(f"https://huggingface.co/api/models?search={query}")
    models = response.json()
    return [model['modelId'] for model in models]

def start_container(model, max_model_len, tensor_parallel_size, gpu_memory_utilization):
    container = client.containers.run(
        "vllm_image",
        environment={
            "MODEL_NAME": model,
            "MAX_MODEL_LEN": max_model_len,
            "TENSOR_PARALLEL_SIZE": tensor_parallel_size,
            "GPU_MEMORY_UTILIZATION": gpu_memory_utilization
        },
        detach=True,
        runtime="nvidia",
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
    )
    return container.id

def stop_container(container_id):
    container = client.containers.get(container_id)
    container.stop()
    container.remove()

with gr.Blocks() as demo:
    search_query = gr.Textbox(label="Suchbegriff für Hugging Face Modelle")
    model_list = gr.Dropdown(label="Verfügbare Modelle")
    search_button = gr.Button("Modelle durchsuchen")
    search_button.click(fn=search_models, inputs=search_query, outputs=model_list)
    
    model = gr.Dropdown(label="Modell auswählen")
    max_model_len = gr.Number(label="Max Model Length")
    tensor_parallel_size = gr.Number(label="Tensor Parallel Size")
    gpu_memory_utilization = gr.Number(label="GPU Memory Utilization")
    start_button = gr.Button("Container starten")
    start_button.click(fn=start_container, inputs=[model, max_model_len, tensor_parallel_size, gpu_memory_utilization])
    
    container_list = gr.Dropdown(label="Laufende Container")
    stop_button = gr.Button("Container stoppen")
    stop_button.click(fn=stop_container, inputs=container_list)

    demo.launch()

# Inference Script
model_name = os.getenv("MODEL_NAME")
max_model_len = int(os.getenv("MAX_MODEL_LEN", 512))
tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", 0.8))

llm = LLM(model_name=model_name, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)