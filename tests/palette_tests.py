import os
import sys
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from ColourSelection import colour_palette_distance
from palette import Palette


PALETTE_SIZE = 5


def display_palette(palette: Palette):
    colour_palette = palette.colours
    counts = palette.counts

    sorted_indices = np.argsort(counts)[::-1]
    sorted_colours = colour_palette[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    plt.figure(figsize=(10, 2))

    plt.bar(
        range(len(sorted_counts)),  # X-axis positions
        sorted_counts,  # Heights of the bars (counts)
        color=[tuple(color / 255) for color in sorted_colours],  # Normalized RGB colors
        width=1
    )
    
    # Hide X and Y ticks for a cleaner look
    plt.xticks([])  
    plt.yticks([])  
    
    # Title for the plot
    plt.title("Colour Palette")

    # Show the plot
    #plt.show()

daniel_clothing_1 = np.array([[12, 29, 52],
                              [47, 44, 73],
                              [157, 151, 173],
                              [20, 26, 60],
                              [10, 13, 29]])
daniel_count_1 = np.array([63637, 11216, 2243, 26207, 6876])
daniel_count_1 = daniel_count_1 / np.linalg.norm(daniel_count_1)
daniel_1 = Palette(daniel_clothing_1, daniel_count_1)


daniel_clothing_2 = np.array([[87, 27, 51],
                              [21, 24, 59],
                              [172, 170, 192],
                              [13, 19, 53],
                              [54, 48, 77]])
daniel_count_2 = np.array([31040, 36567, 4330, 34062, 7142])
daniel_count_2 = daniel_count_2 / np.linalg.norm(daniel_count_2)
daniel_2 = Palette(daniel_clothing_2, daniel_count_2)

brett_clothing_1 = np.array([[47, 72, 104],
                             [76, 77, 104],
                             [17, 28, 67],
                             [54, 47, 72],
                             [24, 24, 55]])
brett_count_1 = np.array([10705, 9043, 14739, 10318, 9072])
brett_count_1 = brett_count_1 / np.linalg.norm(brett_count_1)
brett_1 = Palette(brett_clothing_1, brett_count_1)

brett_clothing_2 = np.array([[21, 32, 65],
                             [53, 98, 97],
                             [46, 40, 63],
                             [26, 28, 56],
                             [80, 72, 92]])
brett_count_2 = np.array([14342, 5751, 11673, 17286, 8402])
brett_count_2 = brett_count_2 / np.linalg.norm(brett_count_2)
brett_2 = Palette(brett_clothing_2, brett_count_2)

joe_clothing_1 = np.array([[10, 11, 26],
                           [27, 19, 29],
                           [136, 99, 106],
                           [13, 12, 24],
                           [112, 109, 131]])
joe_count_1 = np.array([42696, 10420, 488, 30296, 1880])
joe_count_1 = joe_count_1 / np.linalg.norm(joe_count_1)
joe_1 = Palette(joe_clothing_1, joe_count_1)


joe_clothing_2 = np.array([[15, 19, 41],
                           [43, 39, 58],
                           [157, 154, 184],
                           [13, 19, 43],
                           [23, 23, 45]])
joe_count_2 = np.array([61575, 5166, 949, 42140, 18550])
joe_count_2 = joe_count_2 / np.linalg.norm(joe_count_2)
joe_2 = Palette(joe_clothing_2, joe_count_2)

alan_clothing_1 = np.array([[77, 24, 41],
                   [47, 24, 49],
                   [136, 82, 71],
                   [14, 14, 36],
                   [51, 20, 42]])
alan_count_1 = np.array([51099, 11981, 2014, 12341, 32426])
alan_count_1 = alan_count_1 / np.linalg.norm(alan_count_1)
alan_1 = Palette(alan_clothing_1, alan_count_1)

alan_clothing_2 = np.array([[70, 24, 55],
                   [129, 127, 157],
                   [24, 38, 68],
                   [88, 27, 55],
                   [115, 39, 57]])
alan_count_2 = np.array([37431, 6542, 15943, 129499, 25466])
alan_count_2 = alan_count_2 / np.linalg.norm(alan_count_2)
alan_2 = Palette(alan_clothing_2, alan_count_2)



daphne_clothing_1 = np.array([[117,  49, 163],
                              [ 96,  20, 108],
                              [144,  84, 196],
                              [ 24,  26,  60],
                              [ 41,   3,  40]])
daphne_count_1 = np.array([ 0.94109,   0.22267,    0.18911,    0.14833,   0.083617])
daphne_1 = Palette(daphne_clothing_1, daphne_count_1)

daphne_clothing_2 = np.array([[120, 46, 108],
 [151,  67, 134],
 [ 80,  34,  75],
 [  6,   1,  15],
 [ 21,   7,  27]])
daphne_count_2 = np.array([16107, 10380,  4596,  1854,  1516])
daphne_count_2 = daphne_count_2 / np.linalg.norm(daphne_count_2)
daphne_2 = Palette(daphne_clothing_2, daphne_count_2)

# Olivia's clothing palettes
olivia_clothing_1 = np.array([[200, 100, 50],
                               [180, 90, 40],
                               [120, 60, 30],
                               [90, 45, 20],
                               [60, 30, 15]])
olivia_count_1 = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
olivia_1 = Palette(olivia_clothing_1, olivia_count_1)

olivia_clothing_2 = np.array([[195, 95, 45],
                               [175, 85, 35],
                               [125, 65, 35],
                               [85, 40, 18],
                               [55, 28, 14]])
olivia_count_2 = np.array([10000, 4000, 3000, 2000, 1000])
olivia_count_2 = olivia_count_2 / np.linalg.norm(olivia_count_2)
olivia_2 = Palette(olivia_clothing_2, olivia_count_2)


# Marcus' clothing palettes
marcus_clothing_1 = np.array([[30, 140, 200],
                               [25, 110, 160],
                               [20, 90, 130],
                               [15, 70, 100],
                               [10, 50, 80]])
marcus_count_1 = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
marcus_1 = Palette(marcus_clothing_1, marcus_count_1)

marcus_clothing_2 = np.array([[28, 135, 190],
                               [23, 105, 150],
                               [18, 85, 120],
                               [12, 65, 95],
                               [8, 45, 75]])
marcus_count_2 = np.array([8000, 5000, 4000, 2000, 1000])
marcus_count_2 = marcus_count_2 / np.linalg.norm(marcus_count_2)
marcus_2 = Palette(marcus_clothing_2, marcus_count_2)


# Ethan's clothing palettes
ethan_clothing_1 = np.array([[250, 200, 100],
                              [230, 180, 90],
                              [210, 160, 80],
                              [190, 140, 70],
                              [170, 120, 60]])
ethan_count_1 = np.array([0.45, 0.25, 0.15, 0.1, 0.05])
ethan_1 = Palette(ethan_clothing_1, ethan_count_1)

ethan_clothing_2 = np.array([[245, 195, 95],
                              [225, 175, 85],
                              [215, 155, 75],
                              [185, 135, 65],
                              [165, 115, 55]])
ethan_count_2 = np.array([12000, 7000, 4000, 3000, 1000])
ethan_count_2 = ethan_count_2 / np.linalg.norm(ethan_count_2)
ethan_2 = Palette(ethan_clothing_2, ethan_count_2)


# Sophia's clothing palettes
sophia_clothing_1 = np.array([[90, 200, 50],
                               [70, 180, 40],
                               [50, 160, 30],
                               [30, 140, 20],
                               [10, 120, 10]])
sophia_count_1 = np.array([0.35, 0.3, 0.2, 0.1, 0.05])
sophia_1 = Palette(sophia_clothing_1, sophia_count_1)

sophia_clothing_2 = np.array([[85, 195, 48],
                               [65, 175, 38],
                               [48, 155, 28],
                               [28, 135, 18],
                               [8, 115, 8]])
sophia_count_2 = np.array([9000, 8000, 6000, 3000, 1000])
sophia_count_2 = sophia_count_2 / np.linalg.norm(sophia_count_2)
sophia_2 = Palette(sophia_clothing_2, sophia_count_2)




jake_clothing_1 = np.array([[74, 61, 45],
                            [27, 40, 58],
                            [128, 100, 76],
                            [25, 21, 26],
                            [2, 21, 8]])
jake_count_1 = np.array([0.73332, 0.61731, 0.23808, 0.12101, 0.0992])
jake_1 = Palette(jake_clothing_1, jake_count_1)

jake_clothing_2 = np.array([[39, 34, 26],
                               [39, 37, 35],
                               [120, 101, 77],
                               [48, 37, 21],
                               [40, 33, 39]])
jake_count_2 = np.array([ 0.65072, 0.49671 , 0.48448, 0.29376, 0.093946])
jake_count_2 = jake_count_2 / np.linalg.norm(jake_count_2)
jake_2 = Palette(jake_clothing_2, jake_count_2)



twin_clothing_1 = np.array([[97, 152, 188],
                            [69, 121, 160],
                            [42, 80, 114],
                            [81, 106, 112],
                            [10, 46, 66]])
twin_count_1 = np.array([0.76096, 0.48133, 0.31425, 0.2155, 0.2099])
twin_count_1 = twin_count_1 / np.linalg.norm(twin_count_1)
twin_1 = Palette(twin_clothing_1, twin_count_1)


twin_clothing_2 = np.array([[50, 152, 190],
                            [69, 121, 160],
                            [42, 80, 180],
                            [81, 94, 107],
                            [5, 46, 66]])
twin_count_2 = np.array([0.76096, 0.5, 0.31425, 0.2155, 0.1])
twin_count_2 = twin_count_2 / np.linalg.norm(twin_count_2)
twin_2 = Palette(twin_clothing_2, twin_count_2)


import numpy as np

all_palettes = [daniel_1, daniel_2, brett_1, brett_2, joe_1, joe_2, alan_1, alan_2, daphne_1, daphne_2,
                olivia_1, olivia_2, marcus_1, marcus_2, ethan_1, ethan_2, sophia_1, sophia_2, jake_1, jake_2, twin_1, twin_2]

palettes = {daniel_1: daniel_2, 
            brett_1: brett_2, 
            joe_1 :  joe_2, 
            alan_1 : alan_2,
            daphne_1: daphne_2,
            olivia_1: olivia_2,
            marcus_1: marcus_2,
            ethan_1: ethan_2,
            sophia_1: sophia_2,
            jake_1: jake_2,
            twin_1: twin_2}

num_people = len(all_palettes) // 2  # Integer division

same_scores = []
diff_scores = []

for palette in palettes.keys():
    # Compute similarity for the same person
    same = palettes[palette]
    same_score = colour_palette_distance(palette, same)
    same_scores.append(same_score)

    # display_palette(palette)
    # display_palette(same)

    # Compute similarity for different people
    for different_palette in all_palettes:
        if different_palette != palette and different_palette != same:
            diff_score = colour_palette_distance(palette, different_palette)
            diff_scores.append(diff_score)

# Convert lists to numpy arrays
same_scores = np.array(same_scores)
diff_scores = np.array(diff_scores)

# Compute statistics
same_mean = np.mean(same_scores)
same_min = np.min(same_scores)
same_max = np.max(same_scores)
same_sd = np.std(same_scores)

diff_mean = np.mean(diff_scores)
diff_min = np.min(diff_scores)
diff_max = np.max(diff_scores)
diff_sd = np.std(diff_scores)

# Output results
print("Same Person - " + str([same_mean, same_min, same_max, same_sd]))
print("Different People - :" +  str([diff_mean, diff_min, diff_max, diff_sd]))





