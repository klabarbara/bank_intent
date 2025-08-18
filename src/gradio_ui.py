import gradio as gr
import requests

API_URL = "http://localhost:8000/predict/"  
TIMEOUT = 10


def query_model(text: str) -> str:
    if not text.strip():
        return "Please enter some text."
    try:
        resp = requests.post(API_URL, json={"text": text}, timeout=TIMEOUT)
        # if backend responds with error code, display more than timeout error
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError: 
                detail = resp.text
            return f"{resp.status_code}: {detail}"
        
        # happy path
        return resp.json().get("prediction", "no prediction in response.")
    except requests.exceptions.RequestException as exc:
        return f"Request failed: {exc}"


with gr.Blocks(title="Bank Intent Demo") as demo:
    gr.Markdown("### Predict a banking intent")
    input_box = gr.Textbox(label="Type your question", lines=2, autofocus=True)
    output_box = gr.Textbox(
        label="Intent", interactive=False, placeholder="Prediction appears here"
    )

    input_box.submit(query_model, input_box, output_box)

demo.launch()
