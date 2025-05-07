from ultralytics import YOLO

model = YOLO(r'C:\Users\Thanidu\Desktop\Coco_Nut_Count\runs\detect\train3\weights\best.pt')
metrics = model.val(data=r'C:\Users\Thanidu\Desktop\Coco_Nut_Count\coconut_dataset\data.yaml')
print(metrics.box.map)  # Mean Average Precision (mAP@0.5)