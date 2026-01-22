import gradio as gr
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model

# --- LOAD FILES ---
# For deployment, files must be in the same folder as this script.
# Replace the old "/content/drive/MyDrive/..." paths with local filenames.
model = load_model("herb_identifier.h5")

# Since we aren't using the dataset folder anymore, 
# you should define your 20 plant names here manually:
class_names = [
    "Aloe Vera", "Amla", "Basil", "Corriander", "Curry Leaf",
    "Fenugreek", "Guava", "Hibiscus", "Jasmine", "Lemon",
    "Mint", "Neem", "Peppermint", "Rosemary", "Sandalwood",
    "Spinach", "Stevia", "Tulsi", "Turmeric", "Valerian"
] 

df = pd.read_csv("herb_data.csv")
herbal_info = dict(zip(df["herb"], df["benefit"]))

def symptom_recommend(symptom):
    symptom = symptom.lower()
    matched=[]
    for i in range(len(df)):
        for s in str(df.iloc[i]["symptoms"]).lower().split():
            if s in symptom:
                matched.append(df.iloc[i]["herb"])
                break
    return list(set(matched)) if matched else ["Tulsi","Turmeric"]

def predict(img, symptom):
    img = img.resize((160,160))
    img_arr = np.array(img)/255
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)
    plant = class_names[np.argmax(pred)]
    confidence = round(pred.max()*100,2)

    benefit = herbal_info.get(plant,"No data available")
    recommend = symptom_recommend(symptom)

    return plant, f"{confidence} %", benefit, ", ".join(recommend)

# --- YOUR GRADIO UI ---
css = """
body {background: linear-gradient(to right,#e8f5e9,#ffffff);}
h1 {text-align:center;color:#1b5e20;}
"""

with gr.Blocks(css=css) as app:
    gr.Markdown("<h1>🌿 AI Herbal Plant Identification & Recommendation System</h1>")
    gr.Markdown("Upload a leaf image and enter symptoms to get AI-based herbal guidance.")

    with gr.Row():
        img = gr.Image(type="pil", label="Upload Leaf Image")
        symptom = gr.Textbox(label="Enter Symptoms (e.g. cough, fever)")

    btn = gr.Button("Analyze with AI", variant="primary")

    with gr.Row():
        out1 = gr.Textbox(label="Predicted Plant")
        out2 = gr.Textbox(label="Confidence")

    out3 = gr.Textbox(label="Herbal Benefits")
    out4 = gr.Textbox(label="Recommended Herbs")

    btn.click(predict, inputs=[img, symptom], outputs=[out1, out2, out3, out4])

# --- ADDED DEPLOYMENT LOGIC ---
if __name__ == "__main__":
    # This ensures Render or Hugging Face can assign a port
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
