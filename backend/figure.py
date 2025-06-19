from types import NoneType
from backend.ColourSelection import colour_palette_distance
from backend.constants import FACE_DIFFERENTIATION_THRESHOLD, FACE_MATCH_THRESHOLD, NN_INPUT_DIM
from backend.face_data import FaceData
from backend.feature import Feature
from backend.palette import Palette
from backend.wardrobe import Wardrobe
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchreid
import numpy as np
import cv2
import torch

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


class SimilarityNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


match_model = SimilarityNN(NN_INPUT_DIM)
match_model.load_state_dict(torch.load('backend/removed_bad_features.pth'))
match_model.eval()



model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=0,
    pretrained=True,
)
model.eval()


# Transform the inputs to match training conditions
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


def cie_dist(rgb1, rgb2):
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    color1_rgb = sRGBColor(r1, g1, b1, is_upscaled=True)
    color2_rgb = sRGBColor(r2, g2, b2, is_upscaled=True)

    # Convert to LAB colour space
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)

    # Compute Delta E (CIEDE2000) using predefined func.
    return delta_e_cie2000(color1_lab, color2_lab)



# Represents a single appearance component (a cutout of a human detected in a photo)
class Figure:

    def __init__(self, image: np.ndarray, wardobe: Wardrobe, face: Feature, body: Feature, hair: Feature, clothes: Feature, face_data: FaceData, feature_card: np.ndarray = np.empty((2, 3)), name=""):

        # Numpy image of figure
        self.image = image

        # Clothing data in form of Wardrobe object
        self.wardobe = wardobe

        # Set non clothing features
        self.face = face
        self.hair = hair
        self.body = body
        self.clothes = clothes

        # Set face data
        self.face_data = face_data

        # Set feature card image
        self.feature_card = feature_card

        # Set test label if testing
        self.name = name

        # Convert figure image to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)

        # Get re_id embedding
        img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to('cpu')  # or .cuda()

        with torch.no_grad():
            embedding = model(img_tensor)  # No return_feat needed

        items = [hair, body, clothes]
        self.count = sum(1 for item in items if item is not None) + wardobe.count



        self.reid_embedding = embedding
            

    # Get MTCNN facial comparison
    def euclidian_facial_likeness(self, other_figure: "Figure"):
    
        facial_similarity = None
        face_data_1 = self.face_data
        face_data_2 = other_figure.face_data

        # If both figures have a detected face then return dist. between their embeddings
        if face_data_1 != None and face_data_2 != None and face_data_1.embedding != None and face_data_2.embedding != None and face_data_1.img.size > 0 and face_data_2.img.size > 0:
            facial_similarity = torch.norm(face_data_1.embedding - face_data_2.embedding).item()

            return facial_similarity
    
        return None



    ### Figure comparison metric (controlled by constants.FIGURE_MATCH_THRESHOLD)
    def likeness(self, other_figure: "Figure", using_facial_data = False) -> float:

        
        # Exclude matches from any figures with clearly visible faces that are not similar enough
        if using_facial_data:

            facial_diff = self.euclidian_facial_likeness(other_figure)

            if facial_diff != None and facial_diff > FACE_DIFFERENTIATION_THRESHOLD:
                return (np.inf, {})
            
            if facial_diff != None and facial_diff < FACE_MATCH_THRESHOLD:
                return (0.0, {})
        

        (input_vector, dict) = self.get_training_vector(other_figure)

        input_vector = torch.tensor(input_vector, dtype=torch.float32)
        input_vector = input_vector.unsqueeze(0)

        with torch.no_grad():
            output = match_model(input_vector)
            pred = output.item() 


        # Turn the classification probability into a distance because of how the clustering works
        dist = 1 - pred

        # For testing the ReID baseline
        re_id_dist = np.linalg.norm(self.reid_embedding - other_figure.reid_embedding)
        #return ((re_id_dist- 23.0) / 11.7, {})

        return (dist, {})
        

        


    # Figure comparison metric (controlled by constants.FIGURE_MATCH_THRESHOLD)
    def get_training_vector(self, other_figure: "Figure", using_facial_data = False) -> float:

        dict = {}

        self_attributes = [self.wardobe.hat, 
                           self.wardobe.sunglasses,
                           self.wardobe.upper_clothes,
                           self.wardobe.skirt,
                           self.wardobe.pants,
                           self.wardobe.belt,
                           self.wardobe.left_shoe,
                           self.wardobe.right_shoe,
                           self.wardobe.bag,
                           self.wardobe.scarf,
                           self.hair,
                           self.body,
                           self.clothes]
        
        other_attributes = [other_figure.wardobe.hat, 
                           other_figure.wardobe.sunglasses,
                           other_figure.wardobe.upper_clothes,
                           other_figure.wardobe.skirt,
                           other_figure.wardobe.pants,
                           other_figure.wardobe.belt,
                           other_figure.wardobe.left_shoe,
                           other_figure.wardobe.right_shoe,
                           other_figure.wardobe.bag,
                           other_figure.wardobe.scarf,
                           other_figure.hair,
                           other_figure.body,
                           other_figure.clothes]
        
        num_attr = len(self_attributes)
        
        self_colours = [attribute.colour if type(attribute) != NoneType else None for attribute in self_attributes]
        other_colours = [attribute.colour if type(attribute) != NoneType else None for attribute in other_attributes]

        self_palettes = [attribute.colour_palette if type(attribute) != NoneType else None for attribute in self_attributes]
        other_palettes = [attribute.colour_palette if type(attribute) != NoneType else None for attribute in other_attributes]

        self_embeddings = [attribute.embedding if (type(attribute) != NoneType and type(attribute.embedding != NoneType)) else None for attribute in self_attributes]
        other_embeddings = [attribute.embedding if (type(attribute) != NoneType and type(attribute.embedding != NoneType))  else None for attribute in other_attributes]
        # labels =  ["hat", "sunglasses", "upper clothes", "skirt", "pants", "belt", "left shoe", "right shoe", "bag", "scarf", "hair", "body", "all clothes"]

        match_count = mismatch_count = missing_count = 0

        # Used for the niche data detections
        identifying_item_indices = [0, 1, 3, 5, 8, 9, 10]

        for n in range(num_attr):
            if n in (11, 12):
                continue

            if self_attributes[n] is not None and other_attributes[n] is not None and n in identifying_item_indices:
                match_count += 1    
            elif self_attributes[n] is None and other_attributes[n] is None and n in identifying_item_indices:
                missing_count += 1
            elif n in identifying_item_indices:
                mismatch_count += 1


        palettes_exist = [n for n in range(num_attr) if type(self_palettes[n]) == Palette and type(other_palettes[n]) == Palette]
        colours_exist = [n for n in range(num_attr) if type(self_colours[n]) == np.ndarray and type(other_colours[n]) == np.ndarray]
        embeddings_exist = [n for n in range(num_attr) if type(self_embeddings[n]) == np.ndarray and type(other_embeddings[n]) == np.ndarray]


        # Get re_embedding distance
        re_id_dist = np.linalg.norm(self.reid_embedding - other_figure.reid_embedding)

        colour_differences = [None for _ in range(num_attr)]
        palette_differences = [None for _ in range(num_attr)]
        embedding_differences = [None for _ in range(num_attr)]

        for n in colours_exist:
            colour_differences[n] = cie_dist(self_colours[n], other_colours[n])

        for n in palettes_exist:
            if n == 10:
                palette_differences[n] = cie_dist(self_colours[n], other_colours[n])
            else:
                palette_differences[n] = colour_palette_distance(self_palettes[n], other_palettes[n])

        for n in embeddings_exist:
            embedding_differences[n] = np.linalg.norm(self_embeddings[n] - other_embeddings[n])

        # Upper clothes colours
        if palette_differences[2] is not None:
            upper_clothes_palette_diff = (palette_differences[2] - 0.105) * 11
            dict["upper_clothes_palette_diff"] = upper_clothes_palette_diff
        else:
            upper_clothes_palette_diff = 0

        # All clothes colours
        if palette_differences[12] is not None:
            whole_clothes_palette_diff = (palette_differences[12] - 0.105) * 11
            dict["whole_clothes_palette_diff"] = whole_clothes_palette_diff
        else:
            whole_clothes_palette_diff = 0
        
        # Hair colours
        if palette_differences[10] is not None:
            hair_palette_diff = (palette_differences[10] - 0.0229) * 11
            dict["hair_palette_diff"] = hair_palette_diff
        else:
            hair_palette_diff = 0
        
        # Pants colours
        if palette_differences[4] is not None:
            pants_palette_diff = (palette_differences[4] - 0) * 1
            dict["pants_palette_diff"] = pants_palette_diff
        else:
            pants_palette_diff = 0

        # Skirt colours
        if palette_differences[3] is not None:
            skirt_palette_diff = (palette_differences[3] - 0.06) * 11
            dict["skirt_palette_diff"] = skirt_palette_diff
        else:
            skirt_palette_diff = 0

        # Hat colour
        if palette_differences[0] is not None:
            hat_palette_diff = (palette_differences[0] - 0.06)  * 1
            dict["hat_palette_diff"] = hat_palette_diff
        else:
            hat_palette_diff = 0

        # Scarf colours
        if palette_differences[9] is not None:
            scarf_palette_diff = (palette_differences[9] - 0.1) * 5.8
            dict["scarf_palette_diff"] = scarf_palette_diff
        else:
            scarf_palette_diff = 0

        # Bag colours
        if palette_differences[8] is not None:
            bag_palette_diff = (palette_differences[8] - 0.14) * 25
            dict["bag_palette_diff"] = bag_palette_diff
        else:
            bag_palette_diff = 0

        # skin colours
        if colour_differences[11] is not None:
            skin_palette_diff = (colour_differences[11] - 0.04) * 13
            dict["skin_palette_diff"] = skin_palette_diff
        else:
            skin_palette_diff = 0


        # Hair embedding - weight low
        if embedding_differences[10] is not None:
            hair_emb_diff = (embedding_differences[10] - 4.5) * 0.12
            dict["hair_emb_diff"] = hair_emb_diff
        else:
            hair_emb_diff = 0

        # Skirt embedding
        if embedding_differences[3] is not None:
            skirt_emb_diff = (embedding_differences[3] - 4.5) * 0.13
            dict["skirt_emb_diff"] = hair_emb_diff
        else:
            skirt_emb_diff = 0

        # Upper clothes embedding
        if embedding_differences[2] is not None:
            upperclothes_emb_diff = (embedding_differences[2] - 6.0) * 0.1
            dict["upperclothes_emb_diff"] = upperclothes_emb_diff
        else:
            upperclothes_emb_diff = 0

        # Whole clothes embedding
        if embedding_differences[12] is not None:
            wholeclothes_emb_diff = (embedding_differences[12] - 6.0) * 0.1
            dict["wholeclothes_emb_diff"] = wholeclothes_emb_diff
        else:
            wholeclothes_emb_diff = 0
        
        # hat embedding
        if embedding_differences[0] is not None:
            hat_emb_diff = (embedding_differences[0] - 6.0) * 0.1
            dict["hat_emb_diff"] = hat_emb_diff
        else:
            hat_emb_diff = 0


        one_is_bald = 0.0
        if self.hair is None or self.hair.colour_palette is None:
            if other_figure.hair is not None and other_figure.hair.colour_palette is not None:
                one_is_bald = 1.0
        elif other_figure.hair is None or other_figure.hair.colour_palette is None:
            if self.hair is not None and self.hair.colour_palette is not None:
                one_is_bald = 1.0


        # Niche detections
        
        one_has_hat = 0.0
        if self.wardobe.hat is None or self.wardobe.hat.colour_palette is None:
            if other_figure.wardobe.hat is not None and other_figure.wardobe.hat.colour_palette is not None:
                one_has_hat = 1.0
        elif other_figure.wardobe.hat is None or other_figure.wardobe.hat.colour_palette is None:
            if self.wardobe.hat is not None and self.wardobe.hat.colour_palette is not None:
                one_has_hat = 1.0

        ###### Not a good indicated so removed
        # one_has_sunglasses = 0.0
        # if self.wardobe.sunglasses is None or self.wardobe.sunglasses.colour_palette is None:
        #     if other_figure.wardobe.sunglasses is not None and other_figure.wardobe.sunglasses.colour_palette is not None:
        #         one_has_sunglasses = 1.0
        # elif other_figure.wardobe.sunglasses is None or other_figure.wardobe.sunglasses.colour_palette is None:
        #     if self.wardobe.sunglasses is not None and self.wardobe.sunglasses.colour_palette is not None:
        #         one_has_sunglasses = 1.0

        one_has_skirt = 0.0
        if self.wardobe.skirt is None or self.wardobe.skirt.colour_palette is None:
            if other_figure.wardobe.skirt is not None and other_figure.wardobe.skirt.colour_palette is not None:
                one_has_skirt = 1.0
        elif other_figure.wardobe.skirt is None or other_figure.wardobe.skirt.colour_palette is None:
            if self.wardobe.skirt is not None and self.wardobe.skirt.colour_palette is not None:
                one_has_skirt = 1.0

        one_has_belt = 0.0
        if self.wardobe.belt is None or self.wardobe.belt.colour_palette is None:
            if other_figure.wardobe.belt is not None and other_figure.wardobe.belt.colour_palette is not None:
                one_has_belt = 1.0
        elif other_figure.wardobe.belt is None or other_figure.wardobe.belt.colour_palette is None:
            if self.wardobe.belt is not None and self.wardobe.belt.colour_palette is not None:
                one_has_belt = 1.0
                
        one_has_bag = 0.0
        if self.wardobe.bag is None or self.wardobe.bag.colour_palette is None:
            if other_figure.wardobe.bag is not None and other_figure.wardobe.bag.colour_palette is not None:
                one_has_bag = 1.0
        elif other_figure.wardobe.bag is None or other_figure.wardobe.bag.colour_palette is None:
            if self.wardobe.bag is not None and self.wardobe.bag.colour_palette is not None:
                one_has_bag = 1.0

        one_has_scarf = 0.0
        if self.wardobe.scarf is None or self.wardobe.scarf.colour_palette is None:
            if other_figure.wardobe.scarf is not None and other_figure.wardobe.scarf.colour_palette is not None:
                one_has_scarf = 1.0
        elif other_figure.wardobe.scarf is None or other_figure.wardobe.scarf.colour_palette is None:
            if self.wardobe.scarf is not None and self.wardobe.scarf.colour_palette is not None:
                one_has_scarf = 1.0

        # dict["one_is_bald"] = one_is_bald
        # dict["one_has_hat"] = one_has_hat
        # dict["one_has_skirt"] = one_has_skirt
        # dict["one_has_belt"] = one_has_belt
        # dict["one_has_bag"] = one_has_bag
        # dict["one_has_scarf"] = one_has_scarf


        reid_diff = (re_id_dist -23.0) / 11.7

        #vector = [upper_clothes_palette_diff, whole_clothes_palette_diff, hair_palette_diff, pants_palette_diff, skirt_palette_diff, hat_palette_diff, scarf_palette_diff, bag_palette_diff, skin_palette_diff, hair_emb_diff, skirt_emb_diff, upperclothes_emb_diff, wholeclothes_emb_diff, hat_emb_diff, one_has_hat, one_has_skirt, one_has_belt, one_has_bag, one_has_scarf, reid_diff]
        #vector = [upper_clothes_palette_diff, whole_clothes_palette_diff, hair_palette_diff, pants_palette_diff, skirt_palette_diff, hat_palette_diff, scarf_palette_diff, bag_palette_diff, skin_palette_diff, hair_emb_diff, skirt_emb_diff, upperclothes_emb_diff, wholeclothes_emb_diff, hat_emb_diff, one_has_hat, one_has_skirt, one_has_belt, one_has_bag, one_has_scarf, 0]
        # neutralising the niche gives better performance vvv
        vector = [upper_clothes_palette_diff, whole_clothes_palette_diff, hair_palette_diff, pants_palette_diff, skirt_palette_diff, hat_palette_diff, scarf_palette_diff, bag_palette_diff, skin_palette_diff, hair_emb_diff, skirt_emb_diff, upperclothes_emb_diff, wholeclothes_emb_diff, hat_emb_diff, reid_diff]
        
        return (np.array(vector), dict)



    
    ### Old code, manual combination, heuristic weights, no NN
    def initial_maunal_likeness(self, other_figure: "Figure", using_facial_data = True) -> float:

        dict = {}


        re_id_dist_default = 24.0
        whole_clothes_palette_dist_default = 0.055
        whole_clothes_embedding_dist_default = 6.6
        hair_embedding_dist_default = 5.8
        hair_palette_dist_default = 0.036
        body_palette_dist_default = 0.044
        hat_palette_dist_default = 0.10
        sunglasses_palette_dist_default = 0.060
        upper_clothes_palette_dist_default = 0.061
        skirt_palette_dist_default = 0.049
        pants_palette_dist_default = 0.052
        belt_palette_dist_default = 0.020
        shoe_palette_dist_default = 0.059
        bag_palette_dist_default = 0.013
        scarf_palette_dist_default = 0.020
        hat_embedding_dist_default = 4.5
        sunglasses_embedding_dist_default = 4.5
        upper_clothes_embedding_dist_default = 6.4
        skirt_embedding_dist_default = 5.3
        pants_embedding_dist_default = 6.3
        belt_embedding_dist_default = 3.6
        shoe_embedding_dist_default = 2.5
        bag_embedding_dist_default = 5.6
        scarf_embedding_dist_default = 4.8

        default_mismatch_count = 1.61
        default_reciprocol_match_count = 1 / 2.49
        

        # Exclude matches from any figures with clearly visible faces that are not similar enough
        if using_facial_data:

            facial_diff = self.euclidian_facial_likeness(other_figure)

            if facial_diff != None and facial_diff > FACE_DIFFERENTIATION_THRESHOLD:
                return (np.inf, {})

        self_attributes = [self.wardobe.hat, 
                           self.wardobe.sunglasses,
                           self.wardobe.upper_clothes,
                           self.wardobe.skirt,
                           self.wardobe.pants,
                           self.wardobe.belt,
                           self.wardobe.left_shoe,
                           self.wardobe.right_shoe,
                           self.wardobe.bag,
                           self.wardobe.scarf,
                           self.hair,
                           self.body,
                           self.clothes]
        
        other_attributes = [other_figure.wardobe.hat, 
                           other_figure.wardobe.sunglasses,
                           other_figure.wardobe.upper_clothes,
                           other_figure.wardobe.skirt,
                           other_figure.wardobe.pants,
                           other_figure.wardobe.belt,
                           other_figure.wardobe.left_shoe,
                           other_figure.wardobe.right_shoe,
                           other_figure.wardobe.bag,
                           other_figure.wardobe.scarf,
                           other_figure.hair,
                           other_figure.body,
                           other_figure.clothes]
        
        num_attr = len(self_attributes)
        
        self_colours = [attribute.colour if type(attribute) != NoneType else None for attribute in self_attributes]
        other_colours = [attribute.colour if type(attribute) != NoneType else None for attribute in other_attributes]

        self_palettes = [attribute.colour_palette if type(attribute) != NoneType else None for attribute in self_attributes]
        other_palettes = [attribute.colour_palette if type(attribute) != NoneType else None for attribute in other_attributes]

        self_embeddings = [attribute.embedding if (type(attribute) != NoneType and type(attribute.embedding != NoneType)) else None for attribute in self_attributes]
        other_embeddings = [attribute.embedding if (type(attribute) != NoneType and type(attribute.embedding != NoneType))  else None for attribute in other_attributes]

        labels =  ["hat", "sunglasses", "upper clothes", "skirt", "pants", "belt", "left shoe", "right shoe", "bag", "scarf", "hair", "body", "all clothes"]

        match_count = mismatch_count = missing_count = 0

        default_palettes = [hat_palette_dist_default, sunglasses_palette_dist_default, upper_clothes_palette_dist_default, 
                            skirt_palette_dist_default, pants_palette_dist_default, belt_palette_dist_default, 
                            shoe_palette_dist_default, shoe_palette_dist_default, bag_palette_dist_default, 
                            scarf_palette_dist_default, hair_palette_dist_default, body_palette_dist_default, 
                            whole_clothes_palette_dist_default]
        
        default_embeddings = [hat_embedding_dist_default, sunglasses_embedding_dist_default, upper_clothes_embedding_dist_default, 
                              skirt_embedding_dist_default, pants_embedding_dist_default, belt_embedding_dist_default,
                              shoe_embedding_dist_default, shoe_embedding_dist_default, bag_embedding_dist_default, 
                              scarf_embedding_dist_default, hair_embedding_dist_default, None, whole_clothes_embedding_dist_default]


        for n in range(num_attr):
            if n in (11, 12):
                continue
            if self_attributes[n] is not None and other_attributes[n] is not None:
                match_count += 1
            elif self_attributes[n] is None and other_attributes[n] is None:
                missing_count += 1
            else:
                mismatch_count += 1


        palettes_exist = [n for n in range(num_attr) if type(self_palettes[n]) == Palette and type(other_palettes[n]) == Palette]
        colours_exist = [n for n in range(num_attr) if type(self_colours[n]) == np.ndarray and type(other_colours[n]) == np.ndarray]
        embeddings_exist = [n for n in range(num_attr) if type(self_embeddings[n]) == np.ndarray and type(other_embeddings[n]) == np.ndarray]


        # dict["items missing in one"] = mismatch_count
        # dict["items present in both"] = match_count
        # dict["items missing in both"] = missing_count

        # Get re_embedding distance
        re_id_dist = np.linalg.norm(self.reid_embedding - other_figure.reid_embedding)

        colour_differences = [None for _ in range(num_attr)]
        palette_differences = [None for _ in range(num_attr)]
        embedding_differences = [None for _ in range(num_attr)]

        for n in colours_exist:
            colour_differences[n] = cie_dist(self_colours[n], other_colours[n])

        for n in palettes_exist:
            palette_differences[n] = colour_palette_distance(self_palettes[n], other_palettes[n])

        for n in embeddings_exist:
            embedding_differences[n] = np.linalg.norm(self_embeddings[n] - other_embeddings[n])

        # for n in range(num_attr):
        #     if colour_differences[n] is not None:
        #         dict[labels[n] + " cie single colour diff"] = colour_differences[n]
            
        #     if embedding_differences[n] is not None:
        #         dict[labels[n] + " embedding diff"] = embedding_differences[n]

        #     if palette_differences[n] is not None:
        #         dict[labels[n] + "cie  palette diff (luminance sorted)"] = palette_differences[n]

        # dict["reid embedding dist"] = re_id_dist

            

        metric = 0
        attribute_count = 0

        metric += re_id_dist / re_id_dist_default
        attribute_count += 1

        metric += (mismatch_count) / default_mismatch_count
        attribute_count += 1

        metric += (1.0 / match_count) / default_reciprocol_match_count
        attribute_count += 1

        for n in range(num_attr):
            if palette_differences[n] != None and default_palettes[n] != None:
                metric += palette_differences[n] / default_palettes[n]
                attribute_count += 1

            if embedding_differences[n] != None and default_embeddings[n] != None:
                metric += embedding_differences[n] / default_embeddings[n]
                attribute_count += 1

        if attribute_count > 0:
            metric = metric / attribute_count
        else:
            metric = np.inf

        return (metric, {})



       
