import gradio as gr
import requests

def query_model(input_text):
    url = "http://localhost:8000/predict/"
    response = requests.post(url, json={"text": input_text})
    return response.json()['prediction']

iface = gr.Interface(fn=query_model, 
                     inputs=gr.Textbox(label="Enter your query text please!"), 
                     outputs=gr.Textbox(label="Predicted banking intention:"))
iface.launch(share=True)
