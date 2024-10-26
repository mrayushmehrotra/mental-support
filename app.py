from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained conversational model and emotion detector
model_name = "microsoft/DialoGPT-small"  # Or use a BlenderBot for empathy
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Function to detect emotion
def detect_emotion(input_text):
    emotions = emotion_detector(input_text)
    # Get the highest scoring emotion
    highest_emotion = max(emotions[0], key=lambda x: x["score"])
    return highest_emotion["label"]

# Function to generate supportive response
def generate_supportive_response(input_text):
    # Detect emotion to customize response
    emotion = detect_emotion(input_text)
    prompt = f"The user feels {emotion}. Respond supportively: {input_text}"

    # Encode input and generate response with sampling for variety
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,  # Nucleus sampling
        top_k=50,   # Top-k sampling
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    # Append empathetic message based on emotion
    if emotion == "sadness":
        response += " Remember, you're not alone, and talking about your feelings can really help. Take it one step at a time."
    elif emotion == "anger":
        response += " It's okay to feel angry sometimes. Taking deep breaths and focusing on what you can control can help."
    elif emotion == "fear":
        response += " Facing fears is tough, but you're stronger than you think. Talk it out, and take it at your own pace."

    return response

# Define a route for mental support
@app.route('/mental_support', methods=['POST'])
def mental_support():
    # Get input text
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data['text']
    supportive_response = generate_supportive_response(input_text)

    # Return supportive response
    return jsonify({
        "input_text": input_text,
        "supportive_response": supportive_response
    })

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
