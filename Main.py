import os
import shutil

from matplotlib import pyplot as plt
import numpy as np
import torch
from appearance import Appearance
from constants import FACE_MATCH_THRESHOLD, FIGURE_MATCH_THRESHOLD
from figure_detector import FigureDetector
from person import Person
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2



# Calculate the 'Elements Like Me' F1 metric for clustering solutions
def elm_metric(ground_truth_names: list[str], predicted_names: list[str]) -> float:

    f1s = []

    flat_gt_names = [name for cluster in ground_truth_names for name in cluster]
    flat_pred_names = [name for cluster in predicted_names for name in cluster]

    assert(sorted(flat_pred_names) == sorted(flat_gt_names))

    for e in flat_gt_names:

        predicted_cluster = next((cluster for cluster in predicted_names if e in cluster), [])
        fpe = set(predicted_cluster) - {e}

        ground_truth_cluster = next((cluster for cluster in ground_truth_names if e in cluster), [])
        fte = set(ground_truth_cluster) - {e}

        if not fte and not fpe:
            F1 = 1.0  # Special case: if both only contain e

        else:
            tp = len(fte & fpe)
            fp = len(fpe - fte)
            fn = len(fte - fpe)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)

    f1s = np.array(f1s)

    return np.mean(f1s)


    

# Helper for displaying a clustering solution (formatted as a list of People)
def show_people_figures(people):
    num_people = len(people)
    max_appearances = max(len(p.appearances) for p in people)

    # Get shapes of valid images
    shapes = []
    for person in people:
        for appearance in person.appearances:
            img = appearance.figure.image
            if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                shapes.append(img.shape)

    # Default shape if no valid images
    if shapes:
        avg_shape = np.mean(shapes, axis=0).astype(int)
    else:
        avg_shape = np.array([100, 100, 3])  # fallback
    avg_height, avg_width = avg_shape[0], avg_shape[1]

    inches_per_img = 3
    fig_width = max_appearances * inches_per_img
    fig_height = num_people * inches_per_img

    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(num_people, max_appearances),
                     axes_pad=0.2,  # adjust spacing between images
                     direction="row",  # horizontal layout
                     cbar_location="right")

    def resize_or_pad(img):
        """ Resize images to match avg_shape, or pad with white if necessary. """
        h, w = img.shape[:2]
        
        if (h, w) != (avg_height, avg_width):
            # Resize to average size
            if h / w > avg_height / avg_width:
                img = cv2.resize(img, (int(w * avg_height / h), avg_height))
            else:
                img = cv2.resize(img, (avg_width, int(h * avg_width / w)))

            # Add padding to match the aspect ratio
            top = bottom = (avg_height - img.shape[0]) // 2
            left = right = (avg_width - img.shape[1]) // 2

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img_rgb

    images = []
    descriptions = []
    for person in people:
        row = []

        for appearance in person.appearances:
            img = appearance.figure.image
            if img is not None and isinstance(img, np.ndarray):
                row.append((resize_or_pad(img), ""))
            else:
                # Empty image if none exists
                blank = np.ones((avg_height, avg_width, 3), dtype=np.uint8) * 255
                row.append((blank, None))

        while len(row) < max_appearances:
            blank = np.ones((avg_height, avg_width, 3), dtype=np.uint8) * 255
            row.append((blank, None))

        images.extend(row)
        descriptions.append(person.get_description())

    for i, (ax, (img, title)) in enumerate(zip(grid, images)):
        ax.imshow(img.astype(np.uint8), aspect='auto')
        ax.axis('off')

        if title:
            ax.set_title(title, fontsize=8)

        if i % max_appearances == 0:
            person_index = i // max_appearances
            ax.annotate(f"Person {i // max_appearances + 1} ",
                        xy=(-7, 0.5),
                        xycoords='axes fraction',
                        ha='center', va='center',
                        fontsize=10, color='black', fontweight='bold')   # descriptions[person_index]

    plt.tight_layout(pad=1.0)  # Added padding to avoid overlap
    plt.subplots_adjust(right=0.95, left=0.05, top=0.95, bottom=0.05)  # Fine-tune margin spacing
    plt.show()

    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    w, h = int(renderer.width), int(renderer.height)

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))

    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    save_path = "runtime_temp/figures/ALL_PEOPLE.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr_image)


def merge_faces(event_people) -> list[Person]:
    merged_people: list[Person] = []
    matched_flags = [False] * len(event_people)

    for person_num, person in enumerate(event_people):
        if matched_flags[person_num]:
            continue

        new_person = Person(person.appearances[:])  # make a copy
        matched_flags[person_num] = True

        for other_person_num, other_person in enumerate(event_people):
            if matched_flags[other_person_num] or other_person_num == person_num:
                continue

            person_1_appearances = new_person.appearances  # important: use updated list
            person_2_appearances = other_person.appearances

            photo_nums_1 = {a.photo_num for a in person_1_appearances}
            photo_nums_2 = {a.photo_num for a in person_2_appearances}

            if photo_nums_1.isdisjoint(photo_nums_2):

                if new_person.face_embedding is not None and other_person.face_embedding is not None:
                    dist = torch.norm(new_person.face_embedding - other_person.face_embedding).item()
                else:
                    dist = np.inf

                if dist < FACE_MATCH_THRESHOLD:
                    # Avoid adding duplicates
                    existing_photos = {a.photo_num for a in new_person.appearances}
                    for app in other_person.appearances:
                        if app.photo_num not in existing_photos:
                            app.match_type = str("FM")
                            new_person.add_appearance(app)

                    matched_flags[other_person_num] = True

        merged_people.append(new_person)

    return merged_people



# normal/non centroid clustering (ALL LINKAGE)
def merge_noface_biometrics(event_people, using_facial_data = False, testing_dicts=None) -> list[Person]:
    merged_people: list[Person] = []

    testing =  testing_dicts != None

    done = False
    collected_data = False


    same_people = []
    diff_people = []

    # if testing:
    #     print("TESTING")
    #     figs = [person.appearances[0].figure for person in event_people]
        
    #     for n1, fig_1 in enumerate(figs):
    #         for n2, fig_2 in enumerate(figs):
    #             if n1 != n2:
    #                 dist, dist_dic = fig_1.likeness(fig_2, using_facial_data)
    #                 if [c for c in fig_1.name if c.isalpha()] == [c for c in fig_2.name if c.isalpha()]:
                        
    #                     same_people.append(dist_dic)
    #                 else:
    #                     diff_people.append(dist_dic)
    #                     # if dist_dic.get("whole clothes embedding")  == 0.0:
    #                     #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    #                     #     axs[0].imshow(fig_1.clothes.image, cmap='gray')
    #                     #     axs[0].set_title('Image 1')
    #                     #     axs[0].axis('off')

    #                     #     axs[1].imshow(fig_2.clothes.image, cmap='gray')
    #                     #     axs[1].set_title('Image 2')
    #                     #     axs[1].axis('off')

    #                     #     plt.tight_layout()
    #                     #     plt.show()



    def calc_dist(person_num, person, other_person_num, other_person, collected_data):

        new_person = Person(person.appearances[:])  # make a copy

        if other_person_num == person_num:
            return float('inf'), collected_data

        person_1_appearances = new_person.appearances  # important: use updated list
        person_2_appearances = other_person.appearances

        photo_nums_1 = {a.photo_num for a in person_1_appearances}
        photo_nums_2 = {a.photo_num for a in person_2_appearances}

        if photo_nums_1.isdisjoint(photo_nums_2):
            figs1 = [a.figure for a in person_1_appearances]
            figs2 = [a.figure for a in person_2_appearances]

            distances = [
                fig1.likeness(fig2, using_facial_data)[0]
                for fig1 in figs1
                for fig2 in figs2
                if fig1.likeness(fig2, using_facial_data)[0] is not None
            ]

            
            if distances:
                return np.mean(np.array(distances)), collected_data

        return float('inf'), collected_data
    
    while not done:
        distances = np.full((len(event_people), len(event_people)), float('inf'))

        
        for person_num, person in enumerate(event_people):
                for other_person_num, other_person in enumerate(event_people):
                    distances[person_num][other_person_num], collected_data = calc_dist(person_num, person, other_person_num, other_person, collected_data)
                    

        
        smallest_dist = np.min(distances)
        print(smallest_dist)
        (i, j) = np.unravel_index(np.argmin(distances), distances.shape)

        if smallest_dist < FIGURE_MATCH_THRESHOLD:

            if i < j:
                event_people[i].appearances.extend(event_people[j].appearances)
                event_people.pop(j)
            else:
                event_people[j].appearances.extend(event_people[i].appearances)
                event_people.pop(i)
            
            
            #show_people_figures(event_people)
        else:
            done = True

    return same_people, diff_people


# distance centroid based clustering
def merge_noface_biometrics_distcentroid(event_people, using_facial_data = False, testing_dicts=None) -> list[Person]:
    merged_people: list[Person] = []

    centroid_indices = [0 for _ in event_people]

    testing =  testing_dicts != None

    done = False
    collected_data = False


    same_people = []
    diff_people = []

    if testing:
        print("TESTING")
        figs = [person.appearances[0].figure for person in event_people]
        
        for n1, fig_1 in enumerate(figs):
            for n2, fig_2 in enumerate(figs):
                if n1 != n2:
                    dist, dist_dic = fig_1.likeness(fig_2, using_facial_data)
                    if [c for c in fig_1.name if c.isalpha()] == [c for c in fig_2.name if c.isalpha()]:
                        
                        same_people.append(dist_dic)
                    else:
                        diff_people.append(dist_dic)
                        # if dist_dic.get("whole clothes embedding")  == 0.0:
                        #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                        #     axs[0].imshow(fig_1.clothes.image, cmap='gray')
                        #     axs[0].set_title('Image 1')
                        #     axs[0].axis('off')

                        #     axs[1].imshow(fig_2.clothes.image, cmap='gray')
                        #     axs[1].set_title('Image 2')
                        #     axs[1].axis('off')

                        #     plt.tight_layout()
                        #     plt.show()

    def update_centroids():
        for person_num, person in enumerate(event_people):
            figs = [a.figure for a in person.appearances]
            n = len(figs)

            if n == 0:
                continue

            dist_matrix = np.full((n, n), float('inf'))

            for i in range(n):
                for j in range(i + 1, n):
                    dist = figs[i].likeness(figs[j], using_facial_data)[0]
                    if dist is not None:
                        dist_matrix[i][j] = dist
                        dist_matrix[j][i] = dist

            total_dists = np.sum(dist_matrix, axis=1)

            best_idx = np.argmin(total_dists)

            centroid_indices[person_num] = figs[best_idx]


    def calc_dist(person_num, person, other_person_num, other_person, collected_data):

        new_person = Person(person.appearances[:]) 

        if other_person_num == person_num:
            return float('inf'), collected_data

        person_1_appearances = new_person.appearances 
        person_2_appearances = other_person.appearances

        photo_nums_1 = {a.photo_num for a in person_1_appearances}
        photo_nums_2 = {a.photo_num for a in person_2_appearances}

        if photo_nums_1.isdisjoint(photo_nums_2):
            dist = centroid_indices[person_num].likeness(centroid_indices[other_person_num])
            
            if dist:
                return dist

        return float('inf'), collected_data
    
    while not done:
        update_centroids()
        distances = np.full((len(event_people), len(event_people)), float('inf'))
        

        for i, person in enumerate(event_people):
            for j in range(i + 1, len(event_people)): 
                dist, collected_data = calc_dist(i, person, j, event_people[j], collected_data)
                distances[i][j] = dist
                distances[j][i] = dist

        smallest_dist = np.min(distances)
        print(smallest_dist)
        (i, j) = np.unravel_index(np.argmin(distances), distances.shape)

        if smallest_dist < FIGURE_MATCH_THRESHOLD:
            if i < j:
                event_people[i].appearances.extend(event_people[j].appearances)
                event_people.pop(j)
            else:
                event_people[j].appearances.extend(event_people[i].appearances)
                event_people.pop(i)
            # show_people_figures(event_people)
        else:
            done = True

    return same_people, diff_people



# visibility centroid based clustering
def merge_noface_biometrics_visbased(event_people, using_facial_data = False, testing_dicts=None) -> list[Person]:
    merged_people: list[Person] = []

    testing =  testing_dicts != None

    done = False
    collected_data = False


    same_people = []
    diff_people = []

    if testing:
        print("TESTING")
        figs = [person.appearances[0].figure for person in event_people]
        
        for n1, fig_1 in enumerate(figs):
            for n2, fig_2 in enumerate(figs):
                if n1 != n2:
                    dist, dist_dic = fig_1.likeness(fig_2, using_facial_data)
                    if [c for c in fig_1.name if c.isalpha()] == [c for c in fig_2.name if c.isalpha()]:
                        
                        same_people.append(dist_dic)
                    else:
                        diff_people.append(dist_dic)
                        # if dist_dic.get("whole clothes embedding")  == 0.0:
                        #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                        #     axs[0].imshow(fig_1.clothes.image, cmap='gray')
                        #     axs[0].set_title('Image 1')
                        #     axs[0].axis('off')

                        #     axs[1].imshow(fig_2.clothes.image, cmap='gray')
                        #     axs[1].set_title('Image 2')
                        #     axs[1].axis('off')

                        #     plt.tight_layout()
                        #     plt.show()


    def calc_dist(person_num, person, other_person_num, other_person, collected_data):

        new_person = Person(person.appearances[:]) 

        if other_person_num == person_num:
            return float('inf')

        person_1_appearances = new_person.appearances 
        person_2_appearances = other_person.appearances

        photo_nums_1 = {a.photo_num for a in person_1_appearances}
        photo_nums_2 = {a.photo_num for a in person_2_appearances}

        if photo_nums_1.isdisjoint(photo_nums_2):
            if person.face_embedding != None and other_person.face_embedding != None:
                dist = torch.norm(person.face_embedding - other_person.face_embedding).item()

            else:
                (dist, _) = (person.centroid).likeness(other_person.centroid)
            
            if dist:
                return dist

        return float('inf')
    
    while not done:
        for person in event_people:
            person.update_centroid()

        distances = np.full((len(event_people), len(event_people)), float('inf'))
        

        for i, person in enumerate(event_people):
            for j in range(i + 1, len(event_people)): 
                dist = calc_dist(i, person, j, event_people[j], collected_data)
                print(dist)
                distances[i][j] = dist
                distances[j][i] = dist

        smallest_dist = np.min(distances)

        (i, j) = np.unravel_index(np.argmin(distances), distances.shape)

        if smallest_dist < FIGURE_MATCH_THRESHOLD:
            if i < j:
                event_people[i].appearances.extend(event_people[j].appearances)
                event_people.pop(j)
            else:
                event_people[j].appearances.extend(event_people[i].appearances)
                event_people.pop(i)
            # show_people_figures(event_people)
        else:
            done = True

    return same_people, diff_people





def sort_people(people: list[Person]):
    return sorted([
        sorted((a.photo_num, a.match_type) for a in person.appearances)
        for person in people
    ])




## Takes event photos from runtime_temp/event and clusters them, output in matplotlib. outputs also saved to figures folder
if __name__ == "__main__":


    using_facial_detection = True

    # Initialise figure detector
    figure_detector = FigureDetector()

    # All photos from the "event" folder
    event_image_names = os.listdir("runtime_temp/event")


    if os.path.isdir("runtime_temp/figures"):
        shutil.rmtree("runtime_temp/figures")

    event_people: list[Appearance] = []


    for photo_num, event_image_name in enumerate(event_image_names):
        figures = figure_detector.get_figures_from_photo("runtime_temp/event/" + event_image_name)
       
        people = [Person([Appearance(fig, photo_num + 1)]) for fig in figures]
        event_people.extend(people)
    

    # Gather face matches
    if using_facial_detection:
        new_event_people = merge_faces(event_people)

        while sort_people(new_event_people) != sort_people(event_people):
            event_people = new_event_people
            new_event_people = merge_faces(event_people)


    # Gather non-face matches
    merge_noface_biometrics(event_people, using_facial_detection)


    # saving the images
    for person_num, person in enumerate(event_people):

        for appearance in person.appearances:
            # Save features/ numpy representation
            save_path = "runtime_temp/figures/person_" + str(person_num) + "/person_" + str(person_num)  + "_photo_" + str(appearance.photo_num) + "_features.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, appearance.figure.feature_card)

            # Save person image
            save_path = "runtime_temp/figures/person_" + str(person_num) + "/person_" + str(person_num)  + "_photo_" + str(appearance.photo_num) + ".jpg"
            cv2.imwrite(save_path, appearance.figure.image)


    show_people_figures(event_people)

