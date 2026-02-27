from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import json

app = Flask(__name__)
CORS(app)

print("Model yüklənir...")
model = tf.keras.models.load_model('model.h5')

with open('mapping.json', 'r') as f:
    mapping = json.load(f)

print("Hazırdır!")


def preprocess_image(image_data_url):
    header, data = image_data_url.split(',', 1)
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Şəkli kəs - boş sahələri sil
    image = image.convert('RGBA')
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Ağ fon əlavə et, kvadrat et
    max_side = max(image.size)
    padded = Image.new('RGBA', (max_side, max_side), (0, 0, 0, 255))
    offset = ((max_side - image.size[0]) // 2, (max_side - image.size[1]) // 2)
    padded.paste(image, offset)
    
    # Grayscale + 28x28
    image = padded.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    
    img_array = np.array(image).astype('float32')
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        img = preprocess_image(image_data)
        predictions = model.predict(img)[0]
        top3_indices = np.argsort(predictions)[-3:][::-1]
        results = []
        for idx in top3_indices:
            results.append({
                'character': mapping[str(idx)],
                'confidence': float(predictions[idx]) * 100
            })
        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)