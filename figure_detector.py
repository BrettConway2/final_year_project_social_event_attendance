import numpy as np
import mediapipe as mp
import clip
import cv2
import torch
from matplotlib import pyplot as plt
from PIL import Image
from ultralyticsplus import YOLO
from ColourSelection import get_detailed_colour_kmeans, luminance
from constants import FIGURE_DETECTION_THRESHOLD, FIGURE_PIXEL_QUANTITY_THRESHOLD, MIN_MASK_SIZE
from face_data import FaceData
from clothing_detector import ClothingDetector
from face_detector import FaceDetector
from feature import Feature
from figure import Figure
from segmenter import Segmenter
from wardrobe import Wardrobe
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def visualize_palette_bar(palette, title="Palette"):

    colors = palette.colours
    counts = palette.counts
    
    normalized_colors = colors

    plt.figure(figsize=(10, 2))
    plt.bar(range(len(counts)), counts, color=normalized_colors)
    plt.title(title)
    plt.xlabel("Color Index")
    plt.ylabel("Count")
    plt.xticks(range(len(counts)))
    plt.tight_layout()
    #plt.show()



class FigureDetector:

    def __init__(self):

        self.people_detector: YOLO = YOLO('yolov8x.pt')
        self.segmenter: Segmenter = Segmenter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clothing_detector: ClothingDetector = ClothingDetector(self.segmenter, self.device, self.model, self.preprocess)
        self.face_detector: FaceDetector = FaceDetector()
        self.figure_num = 1
        self.person_count = 1
        self.photo_num = 0


   # Reset detector between events for testing
    def reset(self):
        self.figure_num = 1
        self.person_count = 1
        self.photo_num = 0
    

    # Detect figures from input image
    def detect_figures(self, image: np.ndarray, display_bound=False) -> list[tuple[tuple[int, int], tuple[int, int]]]:

        # Use google mediapipe instace segmentation for figures
        results = self.segmenter.instance_segmenter(image)

        bboxes = []
        figures = []
        masks = []

        # Process each detected figure
        for i, mask in enumerate(results[0].masks.data):

            # Skip low confidence or non-person detections
            conf = float(results[0].boxes.conf[i])
            
            cls_id = int(results[0].boxes.cls[i])
            if cls_id != 0 or mask.size == 0 or conf < FIGURE_DETECTION_THRESHOLD:
                continue


            # Get the binary mask and bounding box
            binary_mask = mask.cpu().numpy()
            bbox = results[0].boxes.xyxy[i].cpu().numpy().astype(int)

            mask = binary_mask.astype(np.uint8)

            # Fill in the gaps in the mask (to get outline only)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)

            # Resize mask
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # Apply mask
            person_segment = cv2.bitwise_and(image, image, mask=mask)

            # Make sure the mask is binary and 1-channel
            mask = mask.astype(np.uint8)
            if mask.max() == 1:
                mask *= 255  # Convert to 0 or 255

            # Create a 3-channel version of the mask
            mask_3_channels = cv2.merge([mask, mask, mask])

            ######################### BACKGROUND CODE #########################
            # # Get height/width of image
            # height, width = image.shape[:2]

            # # Generate random noise image (RGB)
            # noise_pattern = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # # Inverse mask for background (still 3-channel)
            # background_mask_3_channels = cv2.bitwise_not(mask_3_channels)

            # Apply mask to separate person and background
            person_segment = cv2.bitwise_and(image, mask_3_channels)

            ######################### BACKGROUND CODE #########################
            # background = cv2.bitwise_and(noise_pattern, background_mask_3_channels)

            # # Combine person with noisy background
            # final_result = cv2.add(person_segment, background)

            # Add bbox and figures to lists
            bboxes.append(((bbox[0], bbox[1]), (bbox[2], bbox[3])))
            figures.append(person_segment[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            masks.append(mask)

        return bboxes, figures, masks
    


    def get_figure_data(self, people_bboxes: list[tuple[int, int], tuple[int, int]], segmented_people: list[np.ndarray], masks = [], names: list[str] = []) -> list[Figure]:

        # Initialise list of figures found in photo
        figures: list[Figure] = []

        # For each figure cutout in segmented figures        
        for fig_num, ((figure_x1, figure_y1), (figure_x2, figure_y2)) in enumerate(people_bboxes):

            print((figure_x1, figure_y1), (figure_x2, figure_y2))

            # Get image and mask
            image_of_person = segmented_people[fig_num]

            #if len(masks) == 0:
            figure_mask = np.any(image_of_person != [0, 0, 0], axis=-1).astype(np.uint8)
            # else:
            #     figure_mask = masks[fig_num]
            #     figure_mask = figure_mask[figure_y1:figure_y2, figure_x1:figure_x2]


            # USEFUL code for writing new test images, requires manual matching of people!!!
            #cv2.imwrite("runtime_temp/unseen_tests/stock/name_" + str(self.photo_num) + "_" + str(figure_x1) + "_" + str(figure_y1) + "_" + str(figure_x2) + "_" + str(figure_y2) + "_" +  ".jpg", cv2.cvtColor(image_of_person, cv2.COLOR_RGB2BGR))

            # Discard figures which are too small/in the background
            if image_of_person.size <= FIGURE_PIXEL_QUANTITY_THRESHOLD:
                continue


            # MTCNN  face detection   
            face_emb, face_prob, face_img, ((x1, y1), (x2, y2)) = self.face_detector.detect_face(image_of_person)

            if face_emb != None and face_prob != None and face_img.shape != (0, 0, 0) :
                bbox = ((x1 + figure_x1, y1 + figure_y1), (x2 + figure_x1, y2 + figure_y1))
                face_data: torch.Tensor = FaceData(face_emb, face_prob, face_img, bbox)
            else:
                face_data: torch.Tensor = None
            

            # Apply mediapipe segmntation for hair, body, clothes masks
            mp_image: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_of_person.astype(np.uint8))
            category_mask: mp.python._framework_bindings.image.Image = self.segmenter.segment_figure(mp_image, True)

            # Conditions for extracting masks of each part of image
            hair_condition: np.ndarray = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == 1
            body_condition: np.ndarray = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == 2
            face_condition: np.ndarray = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == 3
            clothes_condition: np.ndarray = np.stack((category_mask.numpy_view(),) * 3, axis=-1) == 4

            # Retrieve [hat_mask, sunglasses_mask, upper_clothes_mask, skirt_mask, pants_mask, dress_mask, belt_mask, left_shoe_mask, right_shoe_mask, bag_mask, scarf_mask]
            clothes_masks = self.clothing_detector.detect_clothing_per_bodypart(image_of_person)

            # clothes_masks is now in the format [hat_mask, sunglasses_mask, upper_clothes_mask, skirt_mask, pants_mask, dress_mask, belt_mask, left_shoe_mask, right_shoe_mask, bag_mask, scarf_mask, hair, body, face, clothes(full)]
            clothes_masks.extend([hair_condition, body_condition, face_condition, clothes_condition])

            # Convert each 3D numpy attribute mask into a 2D one and mask again by figure mask to ensure it is bounded within the figure
            for n in range(len(clothes_masks)):
                clothes_masks[n] = clothes_masks[n].all(axis=-1).astype(int) & figure_mask

            # Combine 'upper clothes' and 'dress' classes due to model inconsistencies
            clothes_masks[2] = clothes_masks[5] | clothes_masks[2]

            # Turn off 'dress' mask as data has been moved into 'upper clothes'
            clothes_masks[5] = np.zeros(clothes_masks[5].shape)

            # Initialise colours and palettes lists (indicies correspond to masks in clothes masks)
            clothes_colours = [np.empty((3,)) for _ in clothes_masks]
            clothes_palettes = [[] for _ in clothes_masks]

            # Find palettes + colours of each clothing item
            for n in range(len(clothes_masks)):
                clothes_colours[n], clothes_palettes[n] = get_detailed_colour_kmeans(image_of_person, clothes_masks[n], sort_key=luminance)
                if clothes_colours[n] is not None:
                    clothes_colours[n] = clothes_colours[n][0][0]

            # Turn off flags of non-detected/below pixel threshold attributes
            clothing_flags = [np.count_nonzero(mask) > MIN_MASK_SIZE for mask in clothes_masks]

            # Convert image to BGR
            image_of_person = cv2.cvtColor(image_of_person, cv2.COLOR_RGB2BGR)

            # Set name if testing
            name = ""
            if len(names) > 0:
                name = names[fig_num]

            # Get images of clothing items where masks are valid
            clothes_masked_images = [np.clip(image_of_person * (clothes_masks[n])[:, :, None], 0, 255).astype(np.uint8) for n, f in enumerate(clothing_flags)]

            # Put the clothing cut-out images onto black backgrounds
            for n in range(len(clothes_masked_images)):
                black_pixels = np.all(clothes_masked_images[n] == [0, 0, 0], axis=-1)
                clothes_masked_images[n][black_pixels] = [255, 255, 255]
            
            # Initialise embeddings array
            clothes_embeddings = [None for _ in range(15)]

            # Get hair embedding
            for n in range(len(clothes_masks)):
                if np.count_nonzero(clothes_masks[n]) > MIN_MASK_SIZE:
                    pil_img = Image.fromarray((clothes_masked_images[n]))
                    crop_preprocessed = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.model.encode_image(crop_preprocessed)
                        embedding = embedding.cpu().numpy()
                        clothes_embeddings[n] = embedding



            


            # Define list of features which will be wardrobe attributes
            wardrobe_features = [Feature(clothes_masked_images[0], clothes_colours[0], (0.0, 0.0), clothes_masks[0], clothes_palettes[0], 0.0, 0.0, embedding=clothes_embeddings[0]),
                                Feature(clothes_masked_images[1], clothes_colours[1], (0.0, 0.0), clothes_masks[1], clothes_palettes[1], 0.0, 0.0, embedding=clothes_embeddings[1]),
                                Feature(clothes_masked_images[2], clothes_colours[2], (0.0, 0.0), clothes_masks[2], clothes_palettes[2], 0.0, 0.0, embedding=clothes_embeddings[2]),
                                Feature(clothes_masked_images[3], clothes_colours[3], (0.0, 0.0), clothes_masks[3], clothes_palettes[3], 0.0, 0.0, embedding=clothes_embeddings[3]),
                                Feature(clothes_masked_images[4], clothes_colours[4], (0.0, 0.0), clothes_masks[4], clothes_palettes[4], 0.0, 0.0, embedding=clothes_embeddings[4]),
                                Feature(clothes_masked_images[5], clothes_colours[5], (0.0, 0.0), clothes_masks[5], clothes_palettes[5], 0.0, 0.0, embedding=clothes_embeddings[5]),
                                Feature(clothes_masked_images[6], clothes_colours[6], (0.0, 0.0), clothes_masks[6], clothes_palettes[6], 0.0, 0.0, embedding=clothes_embeddings[6]),
                                Feature(clothes_masked_images[7], clothes_colours[7], (0.0, 0.0), clothes_masks[7], clothes_palettes[7], 0.0, 0.0, embedding=clothes_embeddings[7]),
                                Feature(clothes_masked_images[8], clothes_colours[8], (0.0, 0.0), clothes_masks[8], clothes_palettes[8], 0.0, 0.0, embedding=clothes_embeddings[8]),
                                Feature(clothes_masked_images[9], clothes_colours[9], (0.0, 0.0), clothes_masks[9], clothes_palettes[9], 0.0, 0.0, embedding=clothes_embeddings[9]),
                                Feature(clothes_masked_images[10], clothes_colours[10], (0.0, 0.0), clothes_masks[10], clothes_palettes[10], 0.0, 0.0, embedding=clothes_embeddings[10])]
            
            # Set features to none if they were undetected/have unsuitable masks
            wardrobe_features = [feature if clothing_flags[n] else None for n, feature in enumerate(wardrobe_features)]

            # Set wardrobe
            wardrobe = Wardrobe(wardrobe_features[0], wardrobe_features[1], 
                                wardrobe_features[2], wardrobe_features[3], 
                                wardrobe_features[4], wardrobe_features[5], 
                                wardrobe_features[6], wardrobe_features[7], 
                                wardrobe_features[8], wardrobe_features[9], 
                                wardrobe_features[10])
            

            ############# FIGURE CARD STUFF ################

            labels = ["hat", "sunglasses", "upper clothes", "skirt", "pants", "dress", "belt", "left shoe", "right shoe", "bag", "scarf", "hair", "body", "face", "all clothes"]

            valid_items = [(clothes_palettes[i], clothes_masked_images[i], labels[i], clothes_colours[i])
                        for i in range(len(clothes_palettes))
                        if clothes_palettes[i] is not None]

            num_items = len(valid_items)

            # Create a figure with 3 rows and num_items columns
            fig, axs = plt.subplots(3, num_items, figsize=(4 * num_items, 10))

            # If num_items == 1, axs won't be 2D, so we fix that
            if num_items == 1:
                axs = np.expand_dims(axs, axis=1)

            for col, (palette, masked_image, label, main_colour) in enumerate(valid_items):
                # Row 0: Person image
                axs[0, col].imshow(image_of_person[..., ::-1])
                axs[0, col].set_title("Person")
                axs[0, col].axis('off')

                # Row 1: Masked clothes
                axs[1, col].imshow(masked_image[..., ::-1])
                axs[1, col].set_title(label)
                axs[1, col].axis('off')

                # Row 2: Palette colors
                axs[2, col].set_title("Palette")
                axs[2, col].axis('off')

                # Draw main colour at the top
                main_color_rect = patches.Rectangle((0, 0), 100, 20, linewidth=1,
                                                    edgecolor='black', facecolor=np.array(main_colour))
                axs[2, col].add_patch(main_color_rect)

                # Draw the rest of the palette below main_colour
                for j, color in enumerate(palette.colours):
                    rect = patches.Rectangle((0, (j + 1) * 20), 100, 20, linewidth=1,
                                            edgecolor='none', facecolor=np.array(color))
                    axs[2, col].add_patch(rect)

                # Update axis limits
                axs[2, col].set_xlim(0, 100)
                axs[2, col].set_ylim(0, (len(palette.colours) + 1) * 20)
                axs[2, col].invert_yaxis()

            plt.tight_layout()
            #plt.show()


            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()
            w, h = int(renderer.width), int(renderer.height)

            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))

            feature_card = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            plt.close(fig) 

            ################################################



            # Define figure and add to list
            fig = Figure(image_of_person, wardrobe,
                         Feature(clothes_masked_images[13], clothes_colours[13], (0.0, 0.0), clothes_masks[13], clothes_palettes[13], 0.0, 0.0),
                         Feature(clothes_masked_images[12], clothes_colours[12], (0.0, 0.0), clothes_masks[12], clothes_palettes[12], 0.0, 0.0),
                         Feature(clothes_masked_images[11], clothes_colours[11], (0.0, 0.0), clothes_masks[11], clothes_palettes[11], 0.0, 0.0, embedding=clothes_embeddings[11]),
                         Feature(clothes_masked_images[14], clothes_colours[14], (0.0, 0.0), clothes_masks[14], clothes_palettes[14], 0.0, 0.0, embedding=clothes_embeddings[14]),
                         face_data, name=name, feature_card=feature_card )

            figures.append(fig)





        return figures



    def get_figures_from_photo(self, file_name: str, names: list[str] = []) -> list[Figure]:

        image: np.ndarray = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        
        # Detect figures from the photo
        people_bboxes, segmented_people, masks = self.detect_figures(image, False)  

        self.photo_num += 1

        return self.get_figure_data(people_bboxes, segmented_people, masks, names)


        
