import gradio as gr
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# --- CONFIGURATION ---
# Removed Google Drive paths. Files must be in the same folder as this script.
MODEL_PATH = "herb_identifier.h5"
DATA_PATH = "herb_data.csv"

# Load model and data
model = load_model(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
herbal_info = dict(zip(df["herb"], df["benefit"]))

# Get class names (Assumes folder "dataset_HPI" is uploaded with your images)
if os.path.exists("dataset_HPI"):
    class_names = sorted(os.listdir("dataset_HPI"))
else:
    # Fallback: List your 20 plants here manually if you don't want to upload the whole dataset
    class_names = ["Aloe Vera", "Basil", "Mint", "Neem", "Tulsi"] # Add all 20 names

def symptom_recommend(symptom):
    symptom = symptom.lower()
    matched = []
    for i in range(len(df)):
        for s in str(df.iloc[i]["symptoms"]).lower().split():
            if s in symptom:
                matched.append(df.iloc[i]["herb"])
                break
    return list(set(matched)) if matched else ["Tulsi", "Turmeric"]

def predict(img, symptom):
    img = img.resize((160, 160))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)
    plant = class_names[np.argmax(pred)]
    confidence = round(pred.max() * 100, 2)

    benefit = herbal_info.get(plant, "No data available")
    recommend = symptom_recommend(symptom)

    return plant, f"{confidence} %", benefit, ", ".join(recommend)

# --- UI DESIGN ---
css = """
body {background: linear-gradient(to right, #e8f5e9, #ffffff);}
#title {text-align: center; color: #1b5e20; font-family: 'Arial';}
.gradio-container {border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);}
"""

with gr.Blocks(css=css, title="AI Herb Identifier") as app:
    gr.Markdown("# 🌿 AI Herbal Plant Identification & Recommendation System", elem_id="title")
    gr.Markdown("Identify medicinal plants and get recommendations based on symptoms.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Leaf Image")
            symptom_input = gr.Textbox(label="Enter Symptoms (e.g. cough, fever)", placeholder="Describe how you feel...")
            btn = gr.Button("Analyze with AI", variant="primary")
        
        with gr.Column():
            with gr.Row():
                out1 = gr.Textbox(label="Predicted Plant")
                out2 = gr.Textbox(label="Confidence Level")
            out3 = gr.Textbox(label="Herbal Benefits", lines=3)
            out4 = gr.Textbox(label="Recommended Herbs for Symptoms")

    btn.click(predict, inputs=[img_input, symptom_input], outputs=[out1, out2, out3, out4])

# --- LAUNCH LOGIC FOR DEPLOYMENT ---
if __name__ == "__main__":
    # This allows the server (Render/Hugging Face) to assign a port
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
