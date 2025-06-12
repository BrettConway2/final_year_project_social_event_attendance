import sys
import os
sys.path.append(os.getcwd())
from ColourSelection import get_detailed_colour_kmeans

import cv2
import unittest
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import cv2
import cv2



folder_path = os.getcwd() + "/Tests/colour_selection_tests"
files = os.listdir(folder_path)
file_names = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
file_keywords = [name[:name.rfind('_')].split("_") for name in file_names]

model_path = "selfie_multiclass_256x256.tflite"
base_options = python.BaseOptions(model_asset_path=model_path)

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a image segmenter instance with the image mode:
options = ImageSegmenterOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)

segmenter = ImageSegmenter.create_from_options(options)

accessory_tests = ["black_1", "black_4", "brown_1", "light_yellow_3"]

class TestColourSelection(unittest.TestCase):
        
    def test_complex_hue_selection(self):

            for image_index, file_name in enumerate(file_names):

                with self.subTest(image_index=image_index, file_name=file_name):
                    
                    # Load  current clothing image
                    current_clothing_image = os.path.join(folder_path, file_name)

                    # Converts the jpeg/jpg into a cv2 RGB image
                    image_array = cv2.imread(current_clothing_image)
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                    # Extract clothing name + keywords
                    current_clothing_name = file_name[:file_name.rfind('.')]
                    current_clothing_keywords = current_clothing_name[:current_clothing_name.rfind('_')].split("_")

                    # Get the detected color
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array.astype(np.uint8))

                    category_mask = segmenter.segment(mp_image).category_mask

                    if (current_clothing_name in accessory_tests):
                        mask = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == 5
                    else:
                        mask = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == 4
                    
                    colour = get_detailed_colour_kmeans(image_array, mask)
                    print("Detected color: " + str(colour))

                    # Check if the detected color is in the keywords
                    self.assertIn(colour, current_clothing_keywords, 
                                "Detected color '" + str(colour) + "' not found in keywords " + str(current_clothing_keywords))
            


if __name__ == "__main__":
    unittest.main()