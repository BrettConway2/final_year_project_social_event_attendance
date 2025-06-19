import mediapipe as mp
from mediapipe.tasks import python

import numpy as np
from ultralytics import YOLO


class Segmenter:
    def __init__(self):
        
        # Initialise the Google image segmenter
        model_path: str = "backend/selfie_multiclass_256x256.tflite"
        base_options = python.BaseOptions(model_asset_path=model_path)

        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a image segmenter instance with the image mode:
        options = ImageSegmenterOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True)
       
        self.segmenter = ImageSegmenter.create_from_options(options)
        
        self.instance_segmenter = YOLO("yolov8x-seg.pt")  
    
    # Segment image into categories
    def segment_figure(self, mp_image: mp.Image, display_bounding=False) -> np.ndarray:

        segmented_masks: mp.tasks.python.vision.image_segmenter.ImageSegmenterResult = self.segmenter.segment(mp_image)
        
        return (segmented_masks.category_mask)