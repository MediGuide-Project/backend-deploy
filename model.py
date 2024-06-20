import keras, keras_nlp, os
from flask import Flask, request, jsonify
from google.cloud import translate_v2 as translate

app = Flask(__name__)

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# load the gemma 2b model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
    "kaggle://leonardomarcellino/bangkit-medical-chatbot/keras/gemma2b"
)


# target language: 'en', 'id'
def translate_text(text, target_language):
    # Set up the path to the service account key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "e-analogy-309012-fe19989f7806.json"

    # Initialize the Google Cloud Translate client
    translate_client = translate.Client()

    # Detect the language of the text
    detection = translate_client.detect_language(text)
    detected_language = detection["language"]

    # Perform the translation only if the detected language is not the target language
    if detected_language != target_language:
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"], detected_language

    return text, detected_language


@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json  # data langsung diambil dari POST method
    patient_input = data.get("Patient", "")
    translated_patient_input, detected_language = translate_text(
        patient_input, target_language="en"
    )

    # Define the template string
    template = "Instruction:\n{Patient}\n\nResponse:\n{Doctor}"
    prompt = template.format(
        Patient=translated_patient_input,
        Doctor="",
    )

    sampler = keras_nlp.samplers.TopKSampler(k=1, seed=2)
    gemma_lm.compile(sampler=sampler)
    generated_text = gemma_lm.generate(prompt, max_length=256)

    # If the original input was not in English, translate the response back to the original language
    if detected_language != "en":
        translated_response, _ = translate_text(
            generated_text, target_language=detected_language
        )
    else:
        translated_response = generated_text

    return jsonify({"response": translated_response})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8081)