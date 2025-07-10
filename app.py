import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import torch
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64

# Load pre-trained models
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
image_gen = pipeline("text-to-image-generation", model="CompVis/stable-diffusion-v1-4")
ner = pipeline("ner", grouped_entities=True)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# 1. Sentence Classification
def classify_text(text):
    result = classifier(text, top_k=1)[0]
    return f"Label: {result['label']}, Confidence: {round(result['score'], 3)}"

# 2. Image Generation
def generate_image(prompt):
    image = image_gen(prompt)[0]["image"]
    return image

# 3. NER Graph Mapping
def ner_to_graph(text):
    entities = ner(text)
    G = nx.Graph()

    for ent in entities:
        label = ent['entity_group']
        word = ent['word']
        G.add_node(word, label=label)
        G.add_edge(label, word)

    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=2000, font_size=10)
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)
    return Image.open(img_bytes)

# 4. Fill-in-the-Blank
def fill_in_masked(sentence):
    if "[MASK]" not in sentence:
        return "Please include a [MASK] token in your sentence."
    predictions = fill_mask(sentence)
    return [f"{p['token_str']} (score: {round(p['score'], 3)})" for p in predictions]

# Gradio UI
with gr.Blocks(title="EcoAI Studio") as demo:
    gr.Markdown("## 🌱 EcoAI Studio - Language & Vision Tools for Environmental Intelligence")

    with gr.Tab("1️⃣ Sentence Classification"):
        input_text1 = gr.Textbox(label="Enter Environmental Sentence")
        output1 = gr.Textbox(label="Classification Result")
        btn1 = gr.Button("Classify")
        btn1.click(fn=classify_text, inputs=input_text1, outputs=output1)

    with gr.Tab("2️⃣ Image Generation"):
        input_text2 = gr.Textbox(label="Describe the Environmental Scene")
        output2 = gr.Image(label="Generated Image")
        btn2 = gr.Button("Generate")
        btn2.click(fn=generate_image, inputs=input_text2, outputs=output2)

    with gr.Tab("3️⃣ NER Graph Mapping"):
        input_text3 = gr.Textbox(label="Enter Text with Named Entities (e.g. climate reports)")
        output3 = gr.Image(label="NER Graph")
        btn3 = gr.Button("Generate Graph")
        btn3.click(fn=ner_to_graph, inputs=input_text3, outputs=output3)

    with gr.Tab("4️⃣ Fill in the Blank (Masked LM)"):
        input_text4 = gr.Textbox(label="Enter Sentence with [MASK] (e.g. Trees absorb [MASK].)")
        output4 = gr.Textbox(label="Top Masked Predictions")
        btn4 = gr.Button("Predict")
        btn4.click(fn=fill_in_masked, inputs=input_text4, outputs=output4)

demo.launch()
