from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')
results = model('test.jpg')
count = len(results[0].boxes)
print(f"Coconuts detected: {count}")
results[0].show()