from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO('runs/detect/train3/weights/best.pt')
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    # Predict and count coconuts
    results = model(filepath)
    count = len(results[0].boxes)
    results[0].save(os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename))

    return render_template('index.html', count=count, image='static/uploads/result_' + file.filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)