from matplotlib import pyplot as plt
import numpy as np


alan = [ {'hair_colour': np.array([103, 64, 48]), 'face_colour': np.array([171, 129, 138]), 'body_colour': np.array([155, 123, 135]), 'clothing_colour': np.array([87, 27, 57])},
         {'hair_colour': np.array([166, 119, 111]), 'face_colour': np.array([214, 157, 151]), 'body_colour': np.array([195, 161, 156]), 'clothing_colour': np.array([74, 30, 44])},
         {'hair_colour': np.array([81, 44, 39]), 'face_colour': np.array([140, 79, 69]), 'body_colour': np.array([136, 85, 81]), 'clothing_colour': np.array([75, 23, 40])}
       ]

backgroundman = [ {'hair_colour': np.array([5, 6, 15]), 'face_colour': np.array([121, 66, 99]), 'body_colour': np.array([84, 44, 84]), 'clothing_colour': None},
                  {'hair_colour': None, 'face_colour': np.array([202, 135, 114]), 'body_colour': np.array([164, 140, 143]), 'clothing_colour': np.array([53, 49, 44])},
                  {'hair_colour': None, 'face_colour': np.array([248, 182, 159]), 'body_colour': np.array([136, 77, 91]), 'clothing_colour': np.array([60, 50, 47])}
                ]

barb = [ {'hair_colour': np.array([54, 43, 33]), 'face_colour': np.array([117, 82, 70]), 'body_colour': np.array([92, 62, 49]), 'clothing_colour': np.array([68, 67, 63])},
         {'hair_colour': None, 'face_colour': None, 'body_colour': np.array([193, 159, 147]), 'clothing_colour': np.array([149, 152, 162])}
       ]

blueguy = [ {'hair_colour': np.array([0, 0, 0]), 'face_colour': np.array([4, 9, 31]), 'body_colour': np.array([1, 1, 27]), 'clothing_colour': None},
            {'hair_colour': np.array([0, 0, 0]), 'face_colour': None, 'body_colour': None, 'clothing_colour': None},
            {'hair_colour': np.array([26, 47, 183]), 'face_colour': np.array([132, 162, 207]), 'body_colour': np.array([173, 143, 130]), 'clothing_colour': None},
            {'hair_colour': None, 'face_colour': None, 'body_colour': np.array([122, 93, 129]), 'clothing_colour': None}
          ]

brett = [ {'hair_colour': np.array([124, 103, 99]), 'face_colour': np.array([204, 167, 177]), 'body_colour': np.array([207, 179, 190]), 'clothing_colour': np.array([56, 74, 116])},
          {'hair_colour': np.array([132, 91, 76]), 'face_colour': np.array([195, 134, 129]), 'body_colour': np.array([158, 130, 142]), 'clothing_colour': None},
          {'hair_colour': np.array([138, 86, 56]), 'face_colour': np.array([160, 115, 126]), 'body_colour': np.array([154, 128, 143]), 'clothing_colour': np.array([41, 64, 94])},
          {'hair_colour': np.array([87, 44, 31]), 'face_colour': np.array([166, 119, 127]), 'body_colour': np.array([152, 121, 134]), 'clothing_colour': np.array([28, 28, 55])}
        ]

cat = [ {'hair_colour': np.array([122, 94, 73]), 'face_colour': np.array([159, 103, 89]), 'body_colour': np.array([107, 73, 58]), 'clothing_colour': np.array([15, 17, 16])},
        {'hair_colour': np.array([133, 86, 50]), 'face_colour': np.array([148, 106, 113]), 'body_colour': np.array([150, 100, 91]), 'clothing_colour': np.array([10, 18, 33])},
        {'hair_colour': np.array([158, 103, 67]), 'face_colour': np.array([225, 164, 146]), 'body_colour': np.array([175, 137, 144]), 'clothing_colour': np.array([22, 31, 53])},
        {'hair_colour': np.array([97, 83, 95]), 'face_colour': np.array([195, 157, 167]), 'body_colour': np.array([137, 126, 140]), 'clothing_colour': np.array([18, 38, 69])}
      ]

dan = [ {'hair_colour': np.array([63, 38, 46]), 'face_colour': np.array([169, 114, 101]), 'body_colour': np.array([159, 120, 133]), 'clothing_colour': np.array([16, 20, 55])},
        {'hair_colour': np.array([67, 41, 38]), 'face_colour': np.array([173, 115, 97]), 'body_colour': np.array([147, 111, 121]), 'clothing_colour': np.array([11, 19, 51])},
        {'hair_colour': np.array([73, 58, 64]), 'face_colour': np.array([190, 135, 133]), 'body_colour': np.array([206, 164, 172]), 'clothing_colour': np.array([49, 56, 84])},
        {'hair_colour': np.array([73, 69, 62]), 'face_colour': np.array([180, 112, 98]), 'body_colour': np.array([160, 104, 98]), 'clothing_colour': np.array([28, 30, 34])},
        {'hair_colour': np.array([84, 51, 47]), 'face_colour': np.array([175, 110, 103]), 'body_colour': np.array([197, 140, 128]), 'clothing_colour': np.array([30, 31, 76])}
      ]

david = [ {'hair_colour': np.array([20, 15, 14]), 'face_colour': np.array([147, 104, 78]), 'body_colour': np.array([161, 124, 105]), 'clothing_colour': np.array([48, 34, 26])},
          {'hair_colour': None, 'face_colour': None, 'body_colour': np.array([187, 157, 147]), 'clothing_colour': None}
        ]

dianne = [ {'hair_colour': np.array([137, 109, 66]), 'face_colour': np.array([216, 182, 160]), 'body_colour': np.array([104, 60, 31]), 'clothing_colour': np.array([56, 41, 30])},
           {'hair_colour': np.array([200, 188, 184]), 'face_colour': None, 'body_colour': None, 'clothing_colour': np.array([183, 166, 160])},
           {'hair_colour': None, 'face_colour': None, 'body_colour': None, 'clothing_colour': np.array([0, 0, 0])}
         ]

ed = [ {'hair_colour': np.array([31, 19, 12]), 'face_colour': np.array([136, 94, 70]), 'body_colour': np.array([205, 166, 141]), 'clothing_colour': np.array([30, 31, 27])},
       {'hair_colour': np.array([44, 45, 37]), 'face_colour': np.array([87, 70, 61]), 'body_colour': np.array([59, 50, 37]), 'clothing_colour': np.array([141, 154, 147])},
       {'hair_colour': np.array([46, 39, 34]), 'face_colour': np.array([92, 68, 64]), 'body_colour': np.array([112, 101, 94]), 'clothing_colour': np.array([108, 112, 117])},
       {'hair_colour': None, 'face_colour': None, 'body_colour': np.array([125, 109, 101]), 'clothing_colour': np.array([142, 137, 133])}
     ]

gemma = [ {'hair_colour': np.array([23, 17, 24]), 'face_colour': np.array([141, 92, 103]), 'body_colour': np.array([161, 116, 125]), 'clothing_colour': np.array([180, 175, 191])},
          {'hair_colour': np.array([24, 18, 28]), 'face_colour': np.array([143, 96, 108]), 'body_colour': np.array([152, 109, 123]), 'clothing_colour': np.array([173, 169, 190])},
          {'hair_colour': np.array([38, 35, 37]), 'face_colour': np.array([171, 117, 121]), 'body_colour': np.array([174, 133, 141]), 'clothing_colour': np.array([192, 200, 203])}
        ]

greenguy = [ {'hair_colour': np.array([0, 0, 0]), 'face_colour': None, 'body_colour': None, 'clothing_colour': None},
             {'hair_colour': None, 'face_colour': None, 'body_colour': None, 'clothing_colour': None}
           ]

greenlady = [ {'hair_colour': np.array([144, 110, 97]), 'face_colour': np.array([232, 164, 154]), 'body_colour': np.array([45, 16, 9]), 'clothing_colour': np.array([78, 76, 46])},
              {'hair_colour': np.array([186, 138, 96]), 'face_colour': np.array([247, 181, 157]), 'body_colour': np.array([250, 198, 168]), 'clothing_colour': None},
              {'hair_colour': np.array([217, 187, 142]), 'face_colour': np.array([122, 87, 126]), 'body_colour': np.array([100, 104, 109]), 'clothing_colour': np.array([94, 129, 168])},
              {'hair_colour': np.array([221, 171, 121]), 'face_colour': np.array([249, 177, 151]), 'body_colour': np.array([227, 142, 120]), 'clothing_colour': np.array([22, 20, 15])},
              {'hair_colour': np.array([63, 71, 76]), 'face_colour': np.array([145, 102, 146]), 'body_colour': np.array([105, 96, 142]), 'clothing_colour': np.array([0, 0, 0])}
            ]

greylady = [ {'hair_colour': np.array([1, 1, 13]), 'face_colour': np.array([148, 107, 116]), 'body_colour': np.array([130, 103, 121]), 'clothing_colour': np.array([95, 117, 163])},
             {'hair_colour': np.array([14, 2, 0]), 'face_colour': np.array([245, 178, 144]), 'body_colour': np.array([148, 102, 119]), 'clothing_colour': np.array([40, 31, 27])},
             {'hair_colour': np.array([3, 3, 3]), 'face_colour': np.array([190, 111, 83]), 'body_colour': np.array([167, 115, 81]), 'clothing_colour': np.array([167, 166, 160])},
             {'hair_colour': np.array([7, 2, 0]), 'face_colour': np.array([247, 170, 142]), 'body_colour': np.array([133, 79, 100]), 'clothing_colour': np.array([35, 27, 25])},
             {'hair_colour': None, 'face_colour': np.array([242, 163, 134]), 'body_colour': np.array([120, 59, 77]), 'clothing_colour': np.array([34, 26, 23])}
           ]

hatlady = [ {'hair_colour': np.array([13, 12, 12]), 'face_colour': np.array([238, 156, 120]), 'body_colour': np.array([241, 153, 109]), 'clothing_colour': np.array([32, 33, 24])},
            {'hair_colour': None, 'face_colour': np.array([250, 191, 162]), 'body_colour': np.array([249, 191, 147]), 'clothing_colour': np.array([96, 82, 78])}
          ]

huw = [ {'hair_colour': np.array([113, 94, 106]), 'face_colour': np.array([184, 131, 115]), 'body_colour': np.array([131, 101, 114]), 'clothing_colour': np.array([192, 197, 225])}
      ]

joe = [ {'hair_colour': np.array([119, 78, 63]), 'face_colour': np.array([154, 107, 117]), 'body_colour': np.array([137, 107, 123]), 'clothing_colour': np.array([15, 20, 42])},
        {'hair_colour': np.array([126, 79, 59]), 'face_colour': np.array([141, 84, 70]), 'body_colour': np.array([121, 89, 95]), 'clothing_colour': np.array([9, 11, 25])},
        {'hair_colour': np.array([128, 105, 95]), 'face_colour': np.array([179, 120, 105]), 'body_colour': np.array([161, 111, 93]), 'clothing_colour': np.array([22, 24, 23])},
        {'hair_colour': np.array([130, 73, 44]), 'face_colour': np.array([175, 117, 113]), 'body_colour': np.array([163, 125, 131]), 'clothing_colour': np.array([26, 26, 53])},
        {'hair_colour': np.array([45, 35, 38]), 'face_colour': np.array([123, 90, 98]), 'body_colour': np.array([168, 134, 141]), 'clothing_colour': np.array([20, 20, 25])}
      ]

lastguy = [ {'hair_colour': np.array([181, 125, 97]), 'face_colour': np.array([247, 157, 119]), 'body_colour': np.array([246, 152, 105]), 'clothing_colour': np.array([52, 42, 38])},
            {'hair_colour': np.array([45, 34, 32]), 'face_colour': np.array([221, 136, 100]), 'body_colour': np.array([220, 121, 91]), 'clothing_colour': np.array([0, 0, 0])}
          ]

leslie = [ {'hair_colour': None, 'face_colour': None, 'body_colour': None, 'clothing_colour': None}
         ]

liz = [ {'hair_colour': np.array([125, 117, 97]), 'face_colour': np.array([140, 114, 99]), 'body_colour': np.array([113, 84, 74]), 'clothing_colour': np.array([42, 76, 110])},
        {'hair_colour': np.array([148, 132, 103]), 'face_colour': np.array([169, 136, 117]), 'body_colour': np.array([133, 101, 87]), 'clothing_colour': np.array([39, 68, 88])},
        {'hair_colour': np.array([183, 171, 154]), 'face_colour': np.array([194, 169, 156]), 'body_colour': np.array([157, 131, 123]), 'clothing_colour': np.array([206, 227, 230])},
        {'hair_colour': np.array([238, 231, 224]), 'face_colour': np.array([207, 175, 180]), 'body_colour': np.array([194, 159, 170]), 'clothing_colour': np.array([133, 165, 202])}
      ]

neonyellowguy = [ {'hair_colour': np.array([15, 11, 1]), 'face_colour': np.array([228, 131, 89]), 'body_colour': np.array([215, 138, 94]), 'clothing_colour': np.array([196, 217, 7])},
                  {'hair_colour': np.array([3, 3, 3]), 'face_colour': np.array([39, 12, 3]), 'body_colour': np.array([183, 109, 67]), 'clothing_colour': None},
                  {'hair_colour': np.array([47, 28, 20]), 'face_colour': np.array([243, 158, 115]), 'body_colour': np.array([237, 156, 121]), 'clothing_colour': np.array([33, 27, 25])}
                ]

rachel = [ {'hair_colour': np.array([27, 13, 21]), 'face_colour': np.array([129, 82, 92]), 'body_colour': np.array([145, 111, 119]), 'clothing_colour': np.array([0, 0, 0])},
           {'hair_colour': np.array([38, 33, 38]), 'face_colour': np.array([155, 120, 137]), 'body_colour': np.array([90, 84, 93]), 'clothing_colour': np.array([1, 1, 1])},
           {'hair_colour': np.array([40, 25, 36]), 'face_colour': np.array([172, 124, 134]), 'body_colour': np.array([177, 140, 151]), 'clothing_colour': np.array([174, 165, 189])},
           {'hair_colour': np.array([42, 25, 35]), 'face_colour': np.array([164, 126, 141]), 'body_colour': np.array([122, 100, 121]), 'clothing_colour': None},
           {'hair_colour': np.array([61, 45, 39]), 'face_colour': np.array([189, 129, 113]), 'body_colour': np.array([181, 131, 112]), 'clothing_colour': np.array([44, 48, 47])}
         ]
         
rebecca = [ {'hair_colour': np.array([27, 27, 29]), 'face_colour': np.array([164, 123, 135]), 'body_colour': np.array([212, 159, 153]), 'clothing_colour': np.array([0, 0, 0])},
            {'hair_colour': np.array([29, 18, 30]), 'face_colour': np.array([161, 104, 93]), 'body_colour': np.array([188, 130, 102]), 'clothing_colour': None},
            {'hair_colour': np.array([40, 37, 65]), 'face_colour': np.array([144, 102, 117]), 'body_colour': np.array([109, 91, 113]), 'clothing_colour': np.array([123, 111, 129])},
            {'hair_colour': np.array([55, 42, 66]), 'face_colour': np.array([166, 116, 124]), 'body_colour': np.array([135, 100, 111]), 'clothing_colour': None}
          ]

shaun = [ {'hair_colour': np.array([138, 117, 95]), 'face_colour': np.array([176, 136, 117]), 'body_colour': np.array([215, 181, 164]), 'clothing_colour': np.array([7, 8, 11])},
          {'hair_colour': np.array([174, 164, 145]), 'face_colour': np.array([106, 90, 86]), 'body_colour': np.array([96, 76, 70]), 'clothing_colour': np.array([146, 183, 201])},
          {'hair_colour': np.array([205, 183, 162]), 'face_colour': np.array([216, 190, 179]), 'body_colour': np.array([112, 81, 74]), 'clothing_colour': np.array([214, 222, 225])},
          {'hair_colour': np.array([223, 201, 190]), 'face_colour': np.array([194, 160, 168]), 'body_colour': np.array([167, 131, 138]), 'clothing_colour': np.array([217, 214, 227])},
          {'hair_colour': np.array([63, 44, 31]), 'face_colour': np.array([134, 94, 79]), 'body_colour': np.array([143, 108, 88]), 'clothing_colour': None},
          {'hair_colour': np.array([86, 70, 60]), 'face_colour': np.array([108, 69, 78]), 'body_colour': np.array([131, 110, 128]), 'clothing_colour': None}
        ]

sue = [ {'hair_colour': np.array([176, 155, 148]), 'face_colour': np.array([188, 151, 161]), 'body_colour': np.array([176, 144, 154]), 'clothing_colour': np.array([22, 27, 56])}
      ]

tom = [ {'hair_colour': np.array([30, 27, 31]), 'face_colour': np.array([196, 141, 152]), 'body_colour': np.array([150, 108, 118]), 'clothing_colour': None}
      ]

# people = {"alan": alan, "backgroundman": backgroundman, "barb": barb, "blueguy": blueguy, "brett": brett, "cat": cat, "dan": dan, "david": david, 
#           "dianne": dianne, "ed": ed, "gemma": gemma, "greenguy": greenguy, "greenlady": greenlady, "hatlady": hatlady, "huw": huw, "joe": joe, 
#           "lastguy": lastguy, "leslie": leslie, "liz": liz, "neonyellowguy": neonyellowguy, "rachel": rachel, "rebecca": rebecca, "shaun": shaun}

people = {"alan": alan, "brett": brett, "cat": cat, "dan": dan, "david": david, 
          "ed": ed, "gemma": gemma, "greenguy": greenguy, "greenlady": greenlady, "hatlady": hatlady, "joe": joe, 
          "lastguy": lastguy, "leslie": leslie, "liz": liz, "neonyellowguy": neonyellowguy, "rachel": rachel, "rebecca": rebecca, "shaun": shaun}



import numpy as np
import matplotlib.pyplot as plt

def visualize_all(all_character_data, titles):
    num_people = len(all_character_data)
    num_entries_per_person = [len(data) for data in all_character_data]
    max_entries = max(num_entries_per_person)

    fig, axes = plt.subplots(num_people, max_entries, figsize=(4*max_entries, 4*num_people))

    if num_people == 1:
        axes = [axes]
    if max_entries == 1:
        axes = [[ax] for ax in axes]

    for row_idx, character_data in enumerate(all_character_data):
        for col_idx in range(max_entries):
            ax = axes[row_idx][col_idx]
            if col_idx < len(character_data):
                entry = character_data[col_idx]
                colours = []
                labels = []
                for key in ['hair_colour', 'face_colour', 'body_colour', 'clothing_colour']:
                    colour = entry.get(key)
                    if colour is not None:
                        colours.append(colour/255)
                        labels.append(key.replace('_', ' '))

                for i, colour in enumerate(colours):
                    rect = plt.Rectangle((i, 0), 1, 1, facecolor=colour)
                    ax.add_patch(rect)
                    ax.text(i+0.5, -0.4, labels[i], ha='center', va='top', fontsize=8)

                ax.set_xlim(0, len(colours))
                ax.set_ylim(0, 1.5)
                ax.axis('off')
                if col_idx == 0:
                    ax.set_ylabel(titles[row_idx], rotation=0, labelpad=40, va='center', fontsize=12)
            else:
                ax.axis('off')  # hide empty plots if uneven

    plt.tight_layout()
    #plt.show()



def match_colour(character_list):

    matching_on_attr = 'clothing_colour'

    if character_list[0][matching_on_attr] is not None:
        target_colour = character_list[0][matching_on_attr].astype(float)
    else:
        return character_list

    adjusted_list = []
    for entry in character_list:
        current_colour = entry.get(matching_on_attr)
        
        if current_colour is not None:
            current_colour = current_colour.astype(float)
            safe_current = np.where(current_colour == 0, 1, current_colour)
            factor = target_colour / safe_current
        else:
            factor = np.ones(3)  # No change

        adjusted_entry = {}
        for key in ['hair_colour', 'face_colour', 'body_colour', 'clothing_colour']:
            colour = entry.get(key)
            if colour is not None:
                adjusted_colour = (colour.astype(float) * factor).clip(0, 255).astype(np.uint8)
                adjusted_entry[key] = adjusted_colour
            else:
                adjusted_entry[key] = None
        
        adjusted_list.append(adjusted_entry)

    return adjusted_list



# Main code
all_adjusted = []
titles = []

for person_name, current_person in people.items():
    adjusted = match_colour(current_person)
    all_adjusted.append(adjusted)
    titles.append(person_name)

visualize_all(all_adjusted, titles)