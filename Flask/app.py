import base64
import os
import sys
from flask import Flask, render_template, jsonify, request
try:
    from markupsafe import Markup
except ImportError:
    from flask import Markup

# Ensure Flask directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import predict_image
import utils

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    catalog = utils.get_catalog()
    total_diseases = len(utils.disease_dic)
    total_crops = len(catalog)
    return render_template('index.html', catalog=catalog, total_crops=total_crops, total_diseases=total_diseases)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or file.filename == '':
                if (request.is_json or 
                    'application/json' in request.headers.get('Accept', '') or 
                    request.headers.get('X-Requested-With') == 'XMLHttpRequest'):
                    return jsonify({'success': False, 'error': 'No file uploaded'}), 400
                return render_template('index.html', status=400, res="No file selected"), 400

            img_bytes = file.read()
            pred_class, confidence = predict_image(img_bytes, return_confidence=True)
            print(f"Prediction: {pred_class} ({confidence}%)")
            
            info = utils.parse_disease_info(pred_class)
            info['confidence'] = confidence
            
            # Base64 thumbnail for preview
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            mime = file.content_type or 'image/jpeg'
            data_uri = f"data:{mime};base64,{b64_img}"
            info['image_data'] = data_uri

            # Check if API / Fetch request
            is_ajax = (request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 
                       'application/json' in request.headers.get('Accept', '') or
                       request.args.get('format') == 'json')
            
            if is_ajax:
                return jsonify({'success': True, 'data': info})

            res = Markup(utils.disease_dic[pred_class])
            return render_template('display.html', 
                                   status=200, 
                                   result=res, 
                                   info=info, 
                                   confidence=confidence, 
                                   image_uri=data_uri)
        except Exception as e:
            print("Error during prediction:", e)
            if (request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 
                'application/json' in request.headers.get('Accept', '')):
                return jsonify({'success': False, 'error': str(e)}), 500
                
    return render_template('index.html', status=500, res="Internal Server Error")


@app.route('/api/catalog', methods=['GET'])
def catalog_api():
    return jsonify({
        'success': True,
        'catalog': utils.get_catalog(),
        'total': len(utils.disease_dic)
    })


if __name__ == "__main__":
    app.run(debug=True)
