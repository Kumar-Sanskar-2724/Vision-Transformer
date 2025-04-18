### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple,Dict

# Setup class names
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

### 2. Model and transforms perparation ###
tiny_vit,tiny_vit_transforms = create_model(num_classes=10)

# Load save weights
tiny_vit.load_state_dict(torch.load(f='ViT_feature_extractor.pth',
                                    map_location=torch.device('cpu')))

# Predict function
def predict(image):
    # Make sure the image is in RGB
    image = image.convert("RGB")

    # Apply the necessary transformation for your model
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Get the model's raw output (logits)
        output = model(input_tensor)

        # Apply softmax to convert logits to probabilities
        pred_probs = torch.softmax(output, dim=1)

    # Create a dictionary mapping class names to their probabilities
    pred_label_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    return pred_label_and_probs

# Gradio app
title="Vision Transformer CIFAR-10 Classifier",
description="Upload a CIFAR-10 image, and the ViT Tiny model will predict the class."

# Creating example list
example_list =[['examples/'+example] for example in os.listdir('examples')]

# Create the gradio demo
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs='text',
    examples= example_list,
    title="Vision Transformer CIFAR-10 Classifier",
    description="Upload a CIFAR-10 image, and the ViT Tiny model will predict the class."
)
demo.launch()
