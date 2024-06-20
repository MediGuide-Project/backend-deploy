from flask import Flask, request, jsonify
import requests
import logging

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# URL dari endpoint model Flask API yang dibuat oleh tim ML
ML_API_URL = 'http://35.229.220.76:8081'  # Ganti dengan URL yang sesuai jika endpoint model di-hosting di tempat lain

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        # Ambil data dari permintaan POST
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        logging.debug(f'Received data: {data}')

        # Kirim data ke endpoint model Flask API
        response = requests.post(f'{ML_API_URL}/generate', json=data)

        # Cek apakah permintaan berhasil
        if response.status_code == 200:
            try:
                model_response = response.json()
                logging.debug(f'Response from model: {model_response}')
                return jsonify(model_response)
            except ValueError:
                logging.error('Response is not in JSON format')
                return jsonify({'error': 'Response from ML model is not valid JSON'}), 500
        else:
            logging.error(f'Error from model: {response.text}')
            return jsonify({'error': 'Failed to get response from ML model'}), response.status_code
    except Exception as e:
        logging.error(f'Exception: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
