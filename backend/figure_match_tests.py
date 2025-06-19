from collections import defaultdict
from matplotlib import pyplot as plt
from backend.main import elm_metric, merge_faces, merge_noface_biometrics, show_people_figures, sort_people
from appearance import Appearance
from figure_detector import FigureDetector
from person import Person
import numpy as np
import cv2
import os
import shutil


# Compute stats on potential classification attributes
def compute_stats(dicts_list):

    # Gather values for each key in input dict list
    values_by_key = defaultdict(list)

    for d in dicts_list:
        for key, value in d.items():
            values_by_key[key].append(value)

    # Compute stats per attribute (key)
    stats = {}
    for key, values in values_by_key.items():
        arr = np.array([value for value in values if value != float('inf')])

        stats[key] = {
            'mean': np.mean(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'std': np.std(arr)
        }

    return stats




# NOTE that these tests take pre-processed figure images produced by the instance segmented
# Stored in figure_test_sets with aming convention: personName_photoNum_bx1_bx2_by1_by2
# Where bx1 etc are bbx coordinates in the source image

if __name__ == "__main__":

    # Toggled to turn matching on face on/off
    using_facial_data = True

    # Init. fig detector
    figure_detector = FigureDetector()

    #test_path = 'runtime_temp/final_tests'
    test_path = 'runtime_temp/unseen_tests'

    # Get all event names (folders in figure test sets directory)
    event_names = figure_file_names = os.listdir(test_path)           #event_names = figure_file_names = os.listdir("runtime_temp/figure_test_sets")
    num_events = len(event_names)

    # Init. total F1 to 0 to be averaged at the end
    f1_tot = 0

    # Init list of same people comparisons across all events
    # (for stat recording, will be a list of dictionaries returned by figure.likeness)
    same_people = []
    diff_people = []

    # Apply system to each test event, calculate ELM F1 metric and get stats
    for event in event_names:
        # if event != "barbie":
        #     continue

        print("\n----------------------------------\n")
        print("Processing Event " + event + "\n")

        # All photos from the relevant event folder
        path = test_path + "/" + event
        figure_file_names = os.listdir(path)

        # Init lists for processing/testing
        figure_images = []
        bboxes_for_each_image = []
        names_for_each_bbox = []
        figure_names = []

        # Get photo names (each correspond to a figur cutout photo) and num. photos from current event
        split_file_names = [file_name.split('_') for file_name in figure_file_names]
        num_photos = max([int(file_name[1]) for file_name in split_file_names])
        all_names_set = sorted(list(set([file_name[0] for file_name in split_file_names])))

        ground_truth_num_people = len(all_names_set)
        ground_truth_names = []

        # List of lists for each person, for actual appearances
        # Is populated using the filename setup of the test event
        for _ in range(ground_truth_num_people):
            ground_truth_names.append([])

        # Initialise as lists of lists each corresponding to a photo in the event
        for _ in range(num_photos):
            figure_images.append([])
            bboxes_for_each_image.append([])
            names_for_each_bbox.append([])
            figure_names.append([])

        # For each figure photo we initialise the testing list contents
        for n, file_name in enumerate(split_file_names):

            # Add figure to correct photo in list of lists as ground truth allocation
            ground_truth_names[all_names_set.index(file_name[0])].append(file_name[0] + file_name[1])

            # Get photo num as index (0 onward)
            photo_num = int(file_name[1]) - 1

            # Add name to names list of lists where lists correspond to event photos
            names_for_each_bbox[photo_num].append(file_name[0])

            # Add bbox to corresponding index of bbox list
            bboxes_for_each_image[photo_num].append(((int(file_name[2]), int(file_name[3])), (int(file_name[4]), int(file_name[5]))))

            # Add 
            figure_names[photo_num].append(file_name[0] + file_name[1])

            # Read figure image and add it to corresponding figure images list (index is the photo number)
            image: np.ndarray = cv2.cvtColor(cv2.imread(path + "/" + figure_file_names[n]), cv2.COLOR_BGR2RGB)
            figure_images[photo_num].append(image)


        # Initialise people
        people: list[Person] = []

        # For each event photo we get the figure data and add a person to the list of people
        for photo_num in range(num_photos):

            

            # Get data and pass figure name as a testing label
            figures = figure_detector.get_figure_data(bboxes_for_each_image[photo_num], figure_images[photo_num], names = figure_names[photo_num])

            # We create a new person and also pass through the name as the test label
            people.extend([Person([Appearance(fig, photo_num + 1, test_label=names_for_each_bbox[photo_num][n])]) for n, fig in enumerate(figures)])

        ####### CODE FOR CHECKING IMAGES MATCH UP W/ LABELS
        # for person in people:
        #     print("\n\n")
        #     print("name: " + str(person.appearances[0].test_label))
        #     print("photo: " + str(person.appearances[0].photo_num))
        #     plt.imshow(person.appearances[0].figure.image)
        #     plt.axis('off')
        #     plt.show()


        event_people = people

        # Clear working dir
        if os.path.isdir("runtime_temp/figures"):
            shutil.rmtree("runtime_temp/figures")

        # Cluster on MTCNN facial matches if facial matching is active
        if using_facial_data:

            # Gather face matches
            new_event_people = merge_faces(event_people)

            # Cluster/merge until no more changes can be made
            while sort_people(new_event_people) != sort_people(event_people):
                event_people = new_event_people
                new_event_people = merge_faces(event_people)
        

        # lists of the dictionaries of same and diff comparison data
        testing_dicts = (same_people, diff_people)
        
        # Gather non-face matches
        new_same_people, new_diff_people = merge_noface_biometrics(event_people, using_facial_data, testing_dicts)

        # Extend comparison data with new results
        same_people.extend(new_same_people)
        diff_people.extend(new_diff_people)
    
        
        # Initialise list for predicted clusters/attendees
        predicted_names = []

        # Create a list of name lists corresponding to predicted attendees to compare to ground truth attendees
        for person in event_people:
            names = [appearance.figure.name for appearance in person.appearances]
            predicted_names.append(names)
        
        # Calculate the ELM F1 of the predicted clustering
        f1 = elm_metric(ground_truth_names, predicted_names)
        print(f1)

        # Add F1 to total
        f1_tot += f1 


        # # Compute stats for attribute dictionaries
        # same_stats = compute_stats(same_people)
        # diff_stats = compute_stats(diff_people)

        # # Output stats
        # print("same people stats: ")
        
        # for key, value in same_stats.items():
        #     print(f"{key}:")
        #     for stat_name, stat_value in value.items():
        #         print(f"    {stat_name}: {stat_value}")

        
        # print("diff people stats: ")
        
        # for key, value in diff_stats.items():
        #     print(f"{key}:")
        #     for stat_name, stat_value in value.items():
        #         print(f"    {stat_name}: {stat_value}")
            

        # Reset detector
        figure_detector.reset()
 
        # Show the predicted clustering
        #show_people_figures(event_people)

    # Print average F1 across test set
    print("Average F1: " + str(f1_tot / num_events))

    # Compute stats for attribute dictionaries
    same_stats = compute_stats(same_people)
    diff_stats = compute_stats(diff_people)

    # Output stats
    print("same people stats: ")
    
    for key, value in same_stats.items():
        print(f"{key}:")
        for stat_name, stat_value in value.items():
            print(f"    {stat_name}: {stat_value}")

    
    print("diff people stats: ")
    
    for key, value in diff_stats.items():
        print(f"{key}:")
        for stat_name, stat_value in value.items():
            print(f"    {stat_name}: {stat_value}")
        
