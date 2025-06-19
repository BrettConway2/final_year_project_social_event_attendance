import numpy as np
import mediapipe as mp

def np_image_to_mp(numpy_image):

    array = np.array(numpy_image, dtype=np.uint8)
    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)

    return mediapipe_image