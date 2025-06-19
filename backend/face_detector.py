import numpy as np
import cv2
import cv2
import torch
from backend.constants import MTCNN_FACE_CONFIDENCE_THRESHOLD
from facenet_pytorch import MTCNN, InceptionResnetV1
from retinaface import RetinaFace



class FaceDetector:


    def __init__(self):

        # initialise MTCNN
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.device = device


    ################## OLD CODE FORM RETINAFACE TESTING ##################
    def detect_face_retinaface(self, image: np.ndarray, display_bounding=False):

        if image.shape[-1] == 3:  # Check if it's a colour image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = RetinaFace.extract_faces(image, align = True)
        face = faces[0]

        return None, None, face
    

    # Facial Detection (using MTCNN)
    def detect_face(self, image: np.ndarray):

        # Get image width and height
        (w, h, _) = image.shape

        # Normalise image x (width) coords for cropping
        def normalise_x(x):
            if x < 0:
                return 0
            elif x > w - 1:
                return w - 1
            else:
                return x
       
        # Normalise image y (height) coords for cropping
        def normalise_y(y):
            if y < 0:
                return 0
            elif y > h - 1:
                return h - 1
            else:
                return y

        # Handle invalid images
        if image is None:
            raise ValueError("Input image is None.")

        if image.shape[-1] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input image must have 3 channels (H, W, 3).")

        # Detect faces
        boxes, probs = self.mtcnn.detect(image_rgb)

        if boxes is None:
            return None, None, np.empty((0, 0, 0)), ((0, 0), (0, 0))

        box = boxes[0]

        # Ensure it's a 2D numpy array with shape (1, 4)
        if not isinstance(box, np.ndarray):
            box = np.array(box)

        box = box.reshape(1, 4)

        # Get embedding
        x_aligned = self.mtcnn.extract(image_rgb, box, save_path=None)
        x_aligned = x_aligned.to(self.device)
        embedding = self.resnet(x_aligned.unsqueeze(0)).detach().cpu()

        # Get bounding box coords
        x1, y1, x2, y2 = map(int, boxes[0])
        confidence = probs[0]

        # MTCNN may predict end of face but we cut it off relative to picture
        x1 = normalise_x(x1)
        x2 = normalise_x(x2)
        y1 = normalise_y(y1)
        y2 = normalise_y(y2)

        # Crop face
        face_img = image_rgb.copy()[y1:y2, x1:x2]
        
        return embedding, confidence, face_img, ((x1, y1), (x2, y2))
       