import colorsys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import rgb_to_hsv
from constants import KMEANS_CLUSTERS, MIN_MASK_SIZE
from palette import Palette


from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import webcolors



basic_colour = {
    "aliceblue": ["alice", "blue"],
    "antiquewhite": ["antique", "white"],
    "aqua": ["aqua"],
    "aquamarine": ["aqua", "marine"],
    "azure": ["azure"],
    "beige": ["beige"],
    "bisque": ["bisque"],
    "black": ["black"],
    "blanchedalmond": ["blanched", "almond"],
    "blue": ["blue"],
    "blueviolet": ["blue", "violet"],
    "brown": ["brown"],
    "burlywood": ["burly", "wood"],
    "cadetblue": ["cadet", "blue"],
    "chartreuse": ["chartreuse"],
    "chocolate": ["chocolate"],
    "coral": ["coral"],
    "cornflowerblue": ["cornflower", "blue"],
    "cornsilk": ["corn", "silk"],
    "crimson": ["crimson"],
    "cyan": ["cyan"],
    "darkblue": ["dark", "blue"],
    "darkcyan": ["dark", "cyan"],
    "darkgoldenrod": ["dark", "goldenrod"],
    "darkgray": ["dark", "gray"],
    "darkgrey": ["dark", "grey"],
    "darkgreen": ["dark", "green"],
    "darkkhaki": ["dark", "khaki"],
    "darkmagenta": ["dark", "magenta"],
    "darkolivegreen": ["dark", "olive", "green"],
    "darkorange": ["dark", "orange"],
    "darkorchid": ["dark", "orchid"],
    "darkred": ["dark", "red"],
    "darksalmon": ["dark", "salmon"],
    "darkseagreen": ["dark", "sea", "green"],
    "darkslateblue": ["dark", "slate", "blue"],
    "darkslategray": ["dark", "slate", "gray"],
    "darkslategrey": ["dark", "slate", "grey"],
    "darkturquoise": ["dark", "turquoise"],
    "darkviolet": ["dark", "violet"],
    "deeppink": ["deep", "pink"],
    "deepskyblue": ["deep", "sky", "blue"],
    "dimgray": ["dim", "gray"],
    "dimgrey": ["dim", "grey"],
    "dodgerblue": ["dodger", "blue"],
    "firebrick": ["fire", "brick"],
    "floralwhite": ["floral", "white"],
    "forestgreen": ["forest", "green"],
    "fuchsia": ["fuchsia"],
    "gainsboro": ["gainsboro"],
    "ghostwhite": ["ghost", "white"],
    "gold": ["gold"],
    "goldenrod": ["golden", "rod"],
    "gray": ["gray"],
    "grey": ["grey"],
    "green": ["green"],
    "greenyellow": ["green", "yellow"],
    "honeydew": ["honey", "dew"],
    "hotpink": ["hot", "pink"],
    "indianred": ["indian", "red"],
    "indigo": ["indigo"],
    "ivory": ["ivory"],
    "khaki": ["khaki"],
    "lavender": ["lavender"],
    "lavenderblush": ["lavender", "blush"],
    "lawngreen": ["lawn", "green"],
    "lemonchiffon": ["lemon", "chiffon"],
    "lightblue": ["light", "blue"],
    "lightcoral": ["light", "coral"],
    "lightcyan": ["light", "cyan"],
    "lightgoldenrodyellow": ["light", "goldenrod", "yellow"],
    "lightgray": ["light", "gray"],
    "lightgrey": ["light", "grey"],
    "lightgreen": ["light", "green"],
    "lightpink": ["light", "pink"],
    "lightsalmon": ["light", "salmon"],
    "lightseagreen": ["light", "sea", "green"],
    "lightskyblue": ["light", "sky", "blue"],
    "lightslategray": ["light", "slate", "gray"],
    "lightslategrey": ["light", "slate", "grey"],
    "lightsteelblue": ["light", "steel", "blue"],
    "lightyellow": ["light", "yellow"],
    "lime": ["lime"],
    "limegreen": ["lime", "green"],
    "linen": ["linen"],
    "magenta": ["magenta"],
    "maroon": ["maroon"],
    "mediumaquamarine": ["medium", "aqua", "marine"],
    "mediumblue": ["medium", "blue"],
    "mediumorchid": ["medium", "orchid"],
    "mediumpurple": ["medium", "purple"],
    "mediumseagreen": ["medium", "sea", "green"],
    "mediumslateblue": ["medium", "slate", "blue"],
    "mediumspringgreen": ["medium", "spring", "green"],
    "mediumturquoise": ["medium", "turquoise"],
    "mediumvioletred": ["medium", "violet", "red"],
    "midnightblue": ["midnight", "blue"],
    "mintcream": ["mint", "cream"],
    "mistyrose": ["misty", "rose"],
    "moccasin": ["moccasin"],
    "navajowhite": ["navajo", "white"],
    "navy": ["navy"],
    "oldlace": ["old", "lace"],
    "olive": ["olive"],
    "olivedrab": ["olive", "drab"],
    "orange": ["orange"],
    "orangered": ["orange", "red"],
    "orchid": ["orchid"],
    "palegoldenrod": ["pale", "goldenrod"],
    "palegreen": ["pale", "green"],
    "paleturquoise": ["pale", "turquoise"],
    "palevioletred": ["pale", "violet", "red"],
    "papayawhip": ["papaya", "whip"],
    "peachpuff": ["peach", "puff"],
    "peru": ["peru"],
    "pink": ["pink"],
    "plum": ["plum"],
    "powderblue": ["powder", "blue"],
    "purple": ["purple"],
    "red": ["red"],
    "rosybrown": ["rosy", "brown"],
    "royalblue": ["royal", "blue"],
    "saddlebrown": ["saddle", "brown"],
    "salmon": ["salmon"],
    "sandybrown": ["sandy", "brown"],
    "seagreen": ["sea", "green"],
    "seashell": ["sea", "shell"],
    "sienna": ["sienna"],
    "silver": ["silver"],
    "skyblue": ["sky", "blue"],
    "slateblue": ["slate", "blue"],
    "slategray": ["slate", "gray"],
    "slategrey": ["slate", "grey"],
    "snow": ["snow"],
    "springgreen": ["spring", "green"],
    "steelblue": ["steel", "blue"],
    "tan": ["tan"],
    "teal": ["teal"],
    "thistle": ["thistle"],
    "tomato": ["tomato"],
    "turquoise": ["turquoise"],
    "violet": ["violet"],
    "wheat": ["wheat"],
    "white": ["white"],
    "whitesmoke": ["white", "smoke"],
    "yellow": ["yellow"],
    "yellowgreen": ["yellow", "green"],
    "midnightblue": ["midnight", "blue"]
}





def rgb_to_name(rgb):
    if rgb is None:
        return None
    

    def closest_color(requested_color):

        min_colors = {}
        for name, hex in webcolors.CSS3_NAMES_TO_HEX.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]
    
    colour = tuple(int(c * 255) for c in rgb)


    return closest_color(colour)




def cie_dist(rgb1, rgb2):

    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    color1_rgb = sRGBColor(r1, g1, b1, is_upscaled=True)
    color2_rgb = sRGBColor(r2, g2, b2, is_upscaled=True)

    # Convert to LAB
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)

    # Compute Delta E (CIEDE2000)
    return delta_e_cie2000(color1_lab, color2_lab)



# Get Luminance for sorting palettes
def luminance(rgb):
    return np.sqrt((0.299 * (rgb[0] ^ 2)) + (0.587 * (rgb[1] ^ 2)) + (0.114 * (rgb[2] ^ 2)))


# Get Hue for sorting palettes
def hue(rgb):
    hsv = rgb_to_hsv(rgb)
    return hsv[0]


# Returns string colour, RGB colour, and RGB Colour palette
def get_detailed_colour_kmeans(image: np.ndarray, mask: np.ndarray = None, k: int = KMEANS_CLUSTERS, sort_key = hue) -> tuple[np.ndarray, Palette]:

    # Discard masks which are too small
    if mask is None or np.count_nonzero(mask) < MIN_MASK_SIZE:
        mask = np.ones(image.shape, dtype=np.uint8)
        return None, None
    
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Consider masked pixels
    consider = hsv_image[mask > 0]

    # Decide on k value
    samples = len(consider)
    if samples == 0:
        return None, None
    elif samples < k:
        num_clusters = max(1, samples) 
    else:
        num_clusters = k
    
    # Flatten
    pixels = consider.reshape(-1, 3) 

    # Do kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    
    try:
        kmeans.fit(pixels)
    except ValueError as e:
        return None, None
    
    # Retrieve kmeans centres
    cluster_centers = kmeans.cluster_centers_

    # Retrieve cluster labels and counts
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # Get cluster with highest count
    dominant_cluster_index = labels[np.argmax(counts)]
    dominant_colour = cluster_centers[dominant_cluster_index]
    
    # Retrieve cluster with highest count as RGB
    rgb_colour = cv2.cvtColor(np.array([[dominant_colour]], dtype=np.uint8), cv2.COLOR_HSV2RGB) / 255.0
    
    # Convert palette to RGB
    rgb_palette = cv2.cvtColor(cluster_centers.astype("uint8")[np.newaxis, :, :], cv2.COLOR_HSV2RGB)[0]
    color_count_pairs = list(zip(rgb_palette, counts))

    ################# Sort by Count #################
    #color_count_pairs.sort(key=lambda x: x[1], reverse=True)

    # Sort by luminance
    color_count_pairs.sort(key=lambda x: luminance(x[0]))

    sorted_colors, sorted_counts = zip(*color_count_pairs)
    sorted_colors = np.array(sorted_colors) / 255.0


    # Find the counts as their fraction of the pixels measured (normlise them for more consistent comparison)
    sorted_counts = np.array(sorted_counts)
    count_norm = np.linalg.norm(sorted_counts)
    if count_norm != 0:
        sorted_counts = sorted_counts / count_norm
    else:
        sorted_counts = np.zeros(sorted_counts.shape)

    # Final check
    if not len(sorted_colors) == 5:
        return None, None

    return rgb_colour, Palette((sorted_colors), sorted_counts)

# Convert RGB to HSV
def rgb_to_hsv(rgb_array):

    # Normalize
    rgb_array = rgb_array / 255.0
    
    hsv_array = np.apply_along_axis(
        lambda x: colorsys.rgb_to_hsv(x[0], x[1], x[2]), 
        axis=-1, 
        arr=rgb_array
    )
    
    return np.array(hsv_array)


# Convert to LCS
def rgb_to_lcs(rgb, base=""):

    if type(rgb) != np.ndarray:
        return None

    rgb = np.clip(rgb * 255, 1, 255)
    log_rgb = np.log(rgb)

    # base R
    if base == 0:
        c1 = log_rgb[1] - log_rgb[0] 
        c2 = log_rgb[2] - log_rgb[0]
    # Base G
    elif base == 1:
        c1 = log_rgb[0] - log_rgb[1]
        c2 = log_rgb[2] - log_rgb[1] 
    # Base B
    elif base == 2:
        c1 = log_rgb[0] - log_rgb[2]
        c2 = log_rgb[1] - log_rgb[2]
    
    # Gemoteric
    else:
        g = np.mean(log_rgb)
        rho = log_rgb - g
        return rho

    return np.array([c1, c2])


def colour_palette_distance(palette_1: Palette, palette_2: Palette):

    if palette_1 == None or palette_2 == None:
        # return average colour distance - shouldn't fall to this, checks are done before
        return  0.2

    # Extract color vectors and counts
    colors_1 = palette_1.colours
    colors_2 = palette_2.colours

    ######################## LCS CODE ########################
    # # LCS geometric
    # lcs_colours_1 = np.array([rgb_to_lcs(colour, 2) for colour in colors_1])
    # lcs_colours_2 = np.array([rgb_to_lcs(colour, 2) for colour in colors_2])
    # lcs_dist_1 = np.linalg.norm(lcs_colours_1[0] - lcs_colours_2[0] )
    # lcs_dist_2 = np.linalg.norm(lcs_colours_1[1] - lcs_colours_2[1] )
    # lcs_dist_3 = np.linalg.norm(lcs_colours_1[2] - lcs_colours_2[2] )
    # lcs_dist_4 = np.linalg.norm(lcs_colours_1[3] - lcs_colours_2[3] )
    # lcs_dist_5 = np.linalg.norm(lcs_colours_1[4] - lcs_colours_2[4] )
    # average_lcs_dist = (lcs_dist_1 + lcs_dist_2 + lcs_dist_3 + lcs_dist_4 + lcs_dist_5) / 5.0
    # #average_lcs_dist = lcs_dist_1

    # lcs_dist = np.linalg.norm(average_lcs_dist)


    dist_1 = cie_dist(colors_1[0], colors_2[0] )
    dist_2 = cie_dist(colors_1[1], colors_2[1] )
    dist_3 = cie_dist(colors_1[2], colors_2[2] )
    dist_4 = cie_dist(colors_1[3], colors_2[3] )
    dist_5 = cie_dist(colors_1[4], colors_2[4] )
    average_dist = (dist_1 + dist_2 + dist_3 + dist_4 + dist_5) / 5.0


    ######################## HSV CODE ########################

    # def rgb_to_hsv_np(rgb):
    # # Normalize RGB values to [0, 1] for colorsys
    # r, g, b = rgb
    # return np.array(colorsys.rgb_to_hsv(r, g, b))


    # hsv_colours_1 = np.array([rgb_to_hsv(colour) for colour in colors_1])
    # hsv_colours_2 = np.array([rgb_to_hsv(colour) for colour in colors_2])

    # # Compute pairwise Euclidean distances
    # dists = [
    #     np.linalg.norm(hsv_colours_1[i] - hsv_colours_2[i])
    #     for i in range(5)
    # ]
    # #average_hsv_dist = np.mean(dists)
    # average_hsv_dist = dists[0]

    return average_dist




