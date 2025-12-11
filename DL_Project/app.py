# app.py
from flask import Flask, request, jsonify, send_from_directory
import os
from eval_clean import run_sanet
from Evaluation_metrics import evaluate_images
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if files exist
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'status':'error','message':'Files not provided'})
    
    content_file = request.files['content_image']
    style_file = request.files['style_image']
    
    content_filename = secure_filename(content_file.filename)
    style_filename = secure_filename(style_file.filename)
    
    content_path = os.path.join(UPLOAD_FOLDER, content_filename)
    style_path = os.path.join(UPLOAD_FOLDER, style_filename)
    
    content_file.save(content_path)
    style_file.save(style_path)
    
    return jsonify({'status':'success','content_path': content_path, 'style_path': style_path})

@app.route('/stylize', methods=['POST'])
def stylize():
    data = request.get_json()
    content_path = data['content_path']
    style_path = data['style_path']
    
    # Retrieve advanced parameters from UI
    # Defaults: alpha=1.0 (Full Style), size=512, preserve_color=False
    alpha = float(data.get('alpha', 1.0))
    image_size = int(data.get('image_size', 512))
    preserve_color = data.get('preserve_color', False)
    
    output_filename = 'stylized_' + os.path.basename(content_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    try:
        # Run inference with new parameters
        run_sanet(
            content_path, 
            style_path, 
            output_path,
            alpha=alpha,
            preserve_color=preserve_color,
            image_size=image_size
        )
    except Exception as e:
        return jsonify({'status':'error','message':str(e)})
    
    return jsonify({'status':'success','output_path': output_path})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    content_path = data['content_path']
    stylized_path = data['stylized_path']
    
    try:
        psnr_val, ssim_val, lpips_val = evaluate_images(content_path, stylized_path)
        return jsonify({
            'status':'success',
            'PSNR': float(psnr_val),
            'SSIM': float(ssim_val),
            'LPIPS': float(lpips_val),
            'output_path': stylized_path
        })
    except Exception as e:
        return jsonify({'status':'error','message':str(e)})

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)


