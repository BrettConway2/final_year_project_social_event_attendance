import torch
import numpy as np
import cv2
from transformers import SegformerFeatureExtractor
import matplotlib.patches as mpatches


class ClothingDetector:
    def __init__(self, segmenter, device, model, preprocess):

        # Mediapipe segmenter is initialised
        self.segmenter = segmenter

        # Initialise the human part segmentation model (for clothing data)
        model = torch.load("../human-body-segmentation/body-seg-IIG.pth", map_location='cpu')
        model.eval()
        
        self.hps_model = model
        self.hps_feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")


    def detect_clothing_per_bodypart(self, numpy_image: np.ndarray):
        
        # BGR to RGB
        image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        # Feature extract for inputs
        inputs = self.hps_feature_extractor(images=image, return_tensors="pt")

        # Inference on human part seg. model
        with torch.no_grad():
            outputs = self.hps_model(**inputs)
        
        # Fetch logits
        logits = outputs.logits

        #Deifne height/width of image
        height, width = image.shape[:2]

        # Upsample hps CNN output
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        # Get predicted (max) class per pixel
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Define class names from model labels
        id2label = self.hps_model.config.id2label


        hat_mask = np.zeros((height, width, 3), dtype=np.uint8)
        hat_mask[pred_seg == 1] = [1, 1, 1]

        sunglasses_mask = np.zeros((height, width, 3), dtype=np.uint8)
        sunglasses_mask[pred_seg == 3] = [1, 1, 1]

        upper_clothes_mask = np.zeros((height, width, 3), dtype=np.uint8)
        upper_clothes_mask[pred_seg == 4] = [1, 1, 1]

        skirt_mask = np.zeros((height, width, 3), dtype=np.uint8)
        skirt_mask[pred_seg == 5] = [1, 1, 1]

        pants_mask = np.zeros((height, width, 3), dtype=np.uint8)
        pants_mask[pred_seg == 6] = [1, 1, 1]

        dress_mask = np.zeros((height, width, 3), dtype=np.uint8)
        dress_mask[pred_seg == 7] = [1, 1, 1]

        belt_mask = np.zeros((height, width, 3), dtype=np.uint8)
        belt_mask[pred_seg == 8] = [1, 1, 1]

        left_shoe_mask = np.zeros((height, width, 3), dtype=np.uint8)
        left_shoe_mask[pred_seg == 9] = [1, 1, 1]

        right_shoe_mask = np.zeros((height, width, 3), dtype=np.uint8)
        right_shoe_mask[pred_seg == 10] = [1, 1, 1]

        bag_mask = np.zeros((height, width, 3), dtype=np.uint8)
        bag_mask[pred_seg == 16] = [1, 1, 1]

        scarf_mask = np.zeros((height, width, 3), dtype=np.uint8)
        scarf_mask[pred_seg == 17] = [1, 1, 1]

        # plt.imshow(hat_mask)
        # plt.title("Hat mask")
        # plt.axis("off")
        # plt.show()

        ############ VISUALISAITON CODE BELOW ############

        # class_colors = {
        #     0: (0, 0, 0),         # Background - Black             #
        #     1: (0, 255, 0),       # Hat - Green                    #
        #     2: (0, 0, 255),     # Hair - Blue                      #
        #     3: (0, 255, 255),     # Sunglasses - Cyan              #
        #     4: (255, 0, 0),       # Upper-clothes - Red            #
        #     5: (255, 0, 255),     # Skirt - Magenta                #
        #     6: (128, 0, 255),     # Pants - Purple                 #
        #     7: (255, 128, 0),     # Dress - Orange                 #
        #     8: (0, 128, 255),     # Belt - Blue-Orange             #
        #     9: (150, 75, 0),      # Left-shoe - Brown              #
        #     10: (100, 100, 100),  # Right-shoe - Grey              #
        #     11: (255, 204, 153),  # Face - Skin-tone               #
        #     12: (64, 224, 208),   # Left-leg - Turquoise           #
        #     13: (70, 130, 180),   # Right-leg - Steel blue         #
        #     14: (186, 85, 211),   # Left-arm - Medium orchid       #
        #     15: (60, 179, 113),   # Right-arm - Medium sea green   #
        #     16: (128, 128, 0),    # Bag - Olive                    #
        #     17: (255, 255, 0),    # Scarf - Yellow                 #
        # }

        # Create RGB segmentation mask
        # seg_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        # for class_id, color in class_colors.items():
        #     seg_rgb[pred_seg == class_id] = color

        # # Prepare legend patches
        # patches = []
        # for class_id, label in id2label.items():
        #     if class_id in np.unique(pred_seg):
        #         color = np.array(class_colors[class_id]) / 255.0  # Normalize to 0–1 for matplotlib
        #         patch = mpatches.Patch(color=color, label=f"{class_id}: {label}")
        #         patches.append(patch)

        # Plot original image and segmentation
        
        # plt.figure(figsize=(14, 6))

        # plt.subplot(1, 2, 1)
        # plt.title("Original Image")
        # plt.imshow(numpy_image)
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.title("Custom Colored Segmentation")
        # plt.imshow(seg_rgb)
        # plt.axis("off")

        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.tight_layout()
        # #plt.savefig("segmentation_result.png", dpi=300, bbox_inches="tight")  # change filename if needed
        # plt.show()

        return [hat_mask, sunglasses_mask, upper_clothes_mask, skirt_mask, pants_mask, dress_mask, belt_mask, left_shoe_mask, right_shoe_mask, bag_mask, scarf_mask]

