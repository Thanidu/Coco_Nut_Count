from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
model = YOLO(model_path)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads directory

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    print(f"Uploaded image saved at: {filepath}")

    results = model(filepath, conf=0.1)
    print(f"Detected boxes: {results[0].boxes}")
    count = len(results[0].boxes)
    print(f"Coconuts detected: {count}")

    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
    results[0].save(result_path)
    if os.path.exists(result_path):
        print(f"Result image saved at: {result_path}")
    else:
        print(f"Failed to save result image at: {result_path}")

    return render_template('index.html', count=count, image=f'uploads/result_{file.filename}')

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=True)