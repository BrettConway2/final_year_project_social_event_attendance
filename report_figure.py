import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from ultralytics import YOLO

################### FOR REPORT FIGURES VISUALISATION ###################

# Load image
image_path = 'runtime_temp/event/image_4.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load seg.model
model_path = "selfie_multiclass_256x256.tflite"
base_options = python.BaseOptions(model_asset_path=model_path)

ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageSegmenterOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)

segmenter = ImageSegmenter.create_from_options(options)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb.astype(np.uint8))

segmented_masks = segmenter.segment(mp_image)

category_mask = segmented_masks.category_mask.numpy_view()

clothes_mask = (category_mask == 4).astype(np.uint8) * 255
face_mask = (category_mask == 3).astype(np.uint8) * 255
body_mask = (category_mask == 2).astype(np.uint8) * 255
hair_mask = (category_mask == 1).astype(np.uint8) * 255

# Binary to 3-channel boolean
clothes_mask_3ch = np.repeat((clothes_mask > 0)[..., np.newaxis], 3, axis=-1)
face_mask_3ch = np.repeat((face_mask > 0)[..., np.newaxis], 3, axis=-1)
body_mask_3ch = np.repeat((body_mask > 0)[..., np.newaxis], 3, axis=-1)
hair_mask_3ch = np.repeat((hair_mask > 0)[..., np.newaxis], 3, axis=-1)


final_graphic = np.ones_like(image_rgb) * 255
final_graphic[clothes_mask > 0] = [255, 0, 0]
final_graphic[face_mask > 0] = [255, 255, 0]
final_graphic[body_mask > 0] = [0, 255, 255]
final_graphic[hair_mask > 0] = [0, 0, 255]


# Load model
model = YOLO('yolov8n.pt')

results = model(image_rgb)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])


        if cls_id == 0:
            cv2.rectangle(final_graphic, (x1, y1), (x2, y2), (0, 0, 0), 2)
            label = f'Person: {conf:.2f}'
            cv2.putText(final_graphic, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


plt.figure(figsize=(10, 10))
plt.imshow(final_graphic)
plt.axis('off')
plt.title("Segmented & Detected Output")
#plt.show()
