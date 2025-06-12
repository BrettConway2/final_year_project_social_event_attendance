import os
from matplotlib import pyplot as plt
import numpy as np
from constants import NN_INPUT_DIM
from figure_detector import FigureDetector
from person import Person
import cv2
from figure import Figure
import csv

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from random import sample, shuffle




def train_nn(event_people, using_facial_data = False):


    def calc_dist(person_num, person, other_person_num, other_person, collected_data):

        new_person = Person(person.appearances[:]) 

        if other_person_num == person_num:
            return float('inf'), collected_data

        person_1_appearances = new_person.appearances
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
                return sum(distances) / len(distances), collected_data

        return float('inf'), collected_data
    




def sort_people(people: list[Person]):
    return sorted([
        sorted((a.photo_num, a.match_type) for a in person.appearances)
        for person in people
    ])



class SimilarityDataset(Dataset):
    def __init__(self, data):
        self.vectors = [torch.tensor(vec, dtype=torch.float32) for vec, _ in data]
        self.labels = [torch.tensor(lbl, dtype=torch.float32) for _, lbl in data]

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]




class SimilarityNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 128),
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


### Old version for 5 logit voting attempt
# 5_logit_nn.pth
# class SimilarityNN(nn.Module):
#     def __init__(self, input_dim):
#         super(SimilarityNN, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(5, 32),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.BatchNorm1d(16),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.classifier(x)



if __name__ == "__main__":


    ##################################################### This section is for saving the data from one of the test sets
    figure_detector = FigureDetector()
    
    event_names = figure_file_names = os.listdir("runtime_temp/all_tests_train")
    num_events = len(event_names)

    data = []

    # Geting the data, uncomment if new method

    for event in event_names:
        print("\n----------------------------------\n")
        print("Processing Event " + event + "\n")

        # All photos from the "event" folder
        path = "runtime_temp/all_tests_train/" + event
        figure_file_names = os.listdir(path)

        figure_images = []
        bboxes_for_each_image = []
        names_for_each_bbox = []
        figure_names = []
        
        split_file_names = [file_name.split('_') for file_name in figure_file_names]
        num_photos = max([int(file_name[1]) for file_name in split_file_names])
        all_names_set = sorted(list(set([file_name[0] for file_name in split_file_names])))

        ground_truth_num_people = len(all_names_set)
        ground_truth_names = []

        for _ in range(ground_truth_num_people):
            ground_truth_names.append([])

        for _ in range(num_photos):
            figure_images.append([])
            bboxes_for_each_image.append([])
            names_for_each_bbox.append([])
            figure_names.append([])

        for n, file_name in enumerate(split_file_names):

            ground_truth_names[all_names_set.index(file_name[0])].append(file_name[0] + file_name[1])

            photo_num = int(file_name[1]) - 1
            names_for_each_bbox[photo_num].append(file_name[0])
            bboxes_for_each_image[photo_num].append(((int(file_name[2]), int(file_name[3])), (int(file_name[4]), int(file_name[5]))))

            figure_names[photo_num].append(file_name[0] + file_name[1])

            image: np.ndarray = cv2.cvtColor(cv2.imread(path + "/" + figure_file_names[n]), cv2.COLOR_BGR2RGB)
            figure_images[photo_num].append(image)

        labelled_figures: tuple[Figure, str] = []

        for photo_num in range(num_photos):

            figures = figure_detector.get_figure_data(bboxes_for_each_image[photo_num], figure_images[photo_num], figure_names[photo_num])
            labelled_figures.extend([(fig, names_for_each_bbox[photo_num][n]) for n, fig in enumerate(figures)])


        for n1, (fig_1, name_1) in enumerate(labelled_figures):
            for n2, (fig_2, name_2) in enumerate(labelled_figures):
                (vec, dict) = fig_1.get_training_vector(fig_2)

                if n1 != n2:
                    if name_1 == name_2:
                        data.append((vec, 1))

                    else:
                        data.append((vec, 0))

    
    with open('all_data_12_06.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for vec, label in data:
            writer.writerow(list(vec) + [label])
    
 
    ########################################################################## This section is for partitioning data, ratio is variable of +ve to -ve samples

    # Load data
    loaded_data = []
    with open('all_data_final.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vec = list(map(float, row[:-1]))
            label = int((row[-1]))
            loaded_data.append((vec, label))



    # DOWNSAMPLING:
    # Separate 0 and 1 samples
    zeros = [item for item in loaded_data if item[1] == 0]
    ones = [item for item in loaded_data if item[1] == 1]

    
    num_zeros = len(zeros)
    num_ones = len(ones)

    # RATIO of +ve to -ve
    (parts_zeros, parts_ones) = (4.0, 4.0)

    if num_zeros > (parts_zeros / parts_ones) * num_ones:
        num_zeros_target = int((parts_zeros / parts_ones) * num_ones)
        zeros = sample(zeros, num_zeros_target)
    elif num_ones > (parts_ones / parts_zeros) * num_zeros:
        num_ones_target = int((parts_ones / parts_zeros) * num_zeros)
        ones = sample(ones, num_ones_target)

    balanced_data = zeros + ones



    shuffle(balanced_data)

    # Step 3: Split each class into 80/20
    def train_test_split(data, split=0.8):
        split_idx = int(len(data) * split)
        return data[:split_idx], data[split_idx:]

    zeros_train, zeros_test = train_test_split(zeros)
    ones_train, ones_test = train_test_split(ones)

    train_data = zeros_train + ones_train
    test_data = zeros_test + ones_test

    shuffle(train_data)
    shuffle(test_data)

    # Step 4: Save to CSV
    def save_csv(data, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for vec, label in data:
                writer.writerow(vec + [label])

    save_csv(train_data, 'train_data_latest.csv')
    save_csv(test_data, 'test_data_latest.csv')


    ################################################### This section is for testing on the test set (unseen)



    test_data = []
    with open('test_data_latest.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vec = list(map(float, row[:-1]))
            label = int(row[-1])
            test_data.append((vec, label))

    
    match_model = SimilarityNN(NN_INPUT_DIM)
    match_model.load_state_dict(torch.load('backup.pth'))
    match_model.eval()  # set to evaluation mode

    fp = 0
    tp = 0
    tn = 0
    fn = 0

    for (vector, gt) in test_data:

        input_vector = torch.tensor(vector, dtype=torch.float32)
        input_vector = input_vector.unsqueeze(0) 


        with torch.no_grad():  # Turn off gradients for inference
            output = match_model(input_vector)
            prediction = output.item()  # Get scalar from tensor

            print(prediction)


            if prediction > ( 0.5) and gt == 1:
                tp += 1
            elif prediction > (0.5) and gt == 0:
                fp += 1
            elif prediction <= (0.5) and gt == 0:
                tn += 1
            else:
                fn += 1

    a = (tp + tn) / (fp + fn + tp + tn)
            

    
    print("tp: " + str(tp) + " fp: " + str(fp) + " tn: " + str(tn) + "fn: " + str(fn) + " accuracy: " + str(a))

##############################################


# The section below was experiments with ensembles, did not amount to much
############### ENSEMBLE CREATION



    #     # Load data
    # loaded_data = []
    # with open('all_data_latest.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         vec = list(map(float, row[:-1]))
    #         label = int((row[-1]))
    #         loaded_data.append((vec, label))



    # (parts_zeros, parts_ones) = (4.0, 3.0)
    # for n in range(1):
    #     zeros = [item for item in loaded_data if item[1] == 0]
    #     ones = [item for item in loaded_data if item[1] == 1]
        
    #     num_zeros = len(zeros)
    #     num_ones = len(ones)

    # # TRAINING
    #     # Downsample
    #     # Downsample to 4:3 ratio
    #     if num_zeros > (parts_zeros / parts_ones) * num_ones:
    #         num_zeros_target = int((parts_zeros / parts_ones) * num_ones)
    #         zeros = sample(zeros, num_zeros_target)
    #     elif num_ones > (parts_ones / parts_zeros) * num_zeros:
    #         num_ones_target = int((parts_ones / parts_zeros) * num_zeros)
    #         ones = sample(ones, num_ones_target)
    #         # Combine
    #     balanced_data = zeros + ones

    #     shuffle(balanced_data)

    #     # Step 3: Split each class into 80/20
    #     def train_test_split(data, split=0.8):
    #         split_idx = int(len(data) * split)
    #         return data[:split_idx], data[split_idx:]

    #     zeros_train, zeros_test = train_test_split(zeros)
    #     ones_train, ones_test = train_test_split(ones)

    #     train_data = zeros_train + ones_train
    #     test_data = zeros_test + ones_test

    #     shuffle(train_data)
    #     shuffle(test_data)


    #     train_dataset = SimilarityDataset(train_data)
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    #     input_dim = len(train_data[0][0])
    #     model = SimilarityNN(input_dim)

    #     criterion = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.0007)

    #     num_epochs = 500

    #     epochs = list(range(10, num_epochs + 1, 10))

    #     for epoch in range(num_epochs):
    #         model.train()
    #         total_loss = 0
    #         for vectors, labels in train_loader:
    #             outputs = model(vectors).squeeze()
    #             loss = criterion(outputs, labels)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
                
    #             total_loss += loss.item()
            
    #         print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")


    #         if (epoch + 1) in epochs:
    #             model.eval()  # set to evaluation mode

    #             fp = 0
    #             tp = 0
    #             tn = 0
    #             fn = 0

    #             for (vector, gt) in test_data:

    #                 input_vector = torch.tensor(vector, dtype=torch.float32)
    #                 input_vector = input_vector.unsqueeze(0) 


    #                 with torch.no_grad():  # Turn off gradients for inference
    #                     output = model(input_vector)
    #                     prediction = output.item()  # Get scalar from tensor


    #                     if prediction > (1 - FIGURE_MATCH_THRESHOLD) and gt == 1:
    #                         tp += 1
    #                     elif prediction > (1 - FIGURE_MATCH_THRESHOLD) and gt == 0:
    #                         fp += 1
    #                     elif prediction <= (1 - FIGURE_MATCH_THRESHOLD) and gt == 0:
    #                         tn += 1
    #                     else:
    #                         fn += 1
    #             f1 = (tp) / (tp + 0.5 * (fp + fn))
    #             print("tp: " + str(tp) + " fp: " + str(fp) + " tn: " + str(tn) + "fn: " + str(fn) + "   F1: " + str(f1))

    #             if f1 > 0.75:
    #                 print(f"Early stopping model {n}, F1: {f1:.5f}")
    #                 break  # Break out of epoch loop

    #             model.train()
    #     torch.save(model.state_dict(), 'nn5_newestensemble_' + str(n + 5) + '.pth')




########################## PLOTTING

    
    #     # Load data
    # loaded_data = []
    # with open('all_data_latest.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         vec = list(map(float, row[:-1]))
    #         label = int(row[-1])
    #         loaded_data.append((vec, label))

    # num_runs = 5
    # num_epochs = 500
    # epochs_to_record = list(range(10, num_epochs + 1, 10))

    # # Store F1s for each epoch across runs
    # f1_scores_across_runs = {epoch: [] for epoch in epochs_to_record}

    # for n in range(num_runs):
    #     print(f"\n=== Run {n+1} ===")

    #     # Reload and resample data each run
    #     zeros = [item for item in loaded_data if item[1] == 0]
    #     ones = [item for item in loaded_data if item[1] == 1]

    #     # Downsample
    #     if len(zeros) < len(ones):
    #         ones = sample(ones, len(zeros))
    #     else:
    #         zeros = sample(zeros, len(ones))

    #     # Split each class
    #     def train_test_split(data, split=0.8):
    #         split_idx = int(len(data) * split)
    #         return data[:split_idx], data[split_idx:]

    #     zeros_train, zeros_test = train_test_split(zeros)
    #     ones_train, ones_test = train_test_split(ones)

    #     train_data = zeros_train + ones_train
    #     test_data = zeros_test + ones_test

    #     shuffle(train_data)
    #     shuffle(test_data)

    #     # DataLoader
    #     train_dataset = SimilarityDataset(train_data)
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    #     input_dim = len(train_data[0][0])
    #     model = SimilarityNN(input_dim)

    #     criterion = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.0007)

    #     for epoch in range(num_epochs):
    #         model.train()
    #         total_loss = 0
    #         for vectors, labels in train_loader:
    #             outputs = model(vectors).squeeze()
    #             loss = criterion(outputs, labels.float())  # Ensure labels are float

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
            
    #         print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    #         if (epoch + 1) in epochs_to_record:
    #             model.eval()

    #             fp = tp = tn = fn = 0

    #             for (vector, gt) in test_data:
    #                 input_vector = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

    #                 with torch.no_grad():
    #                     output = model(input_vector)
    #                     prediction = output.item()

    #                 if prediction > 0.5 and gt == 1:
    #                     tp += 1
    #                 elif prediction > 0.5 and gt == 0:
    #                     fp += 1
    #                 elif prediction <= 0.5 and gt == 0:
    #                     tn += 1
    #                 else:
    #                     fn += 1

    #             f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) != 0 else 0
    #             f1_scores_across_runs[epoch + 1].append(f1)
    #             print(f"Epoch {epoch+1}: TP={tp}, FP={fp}, TN={tn}, FN={fn}, F1={f1:.4f}")

    # # Compute average F1 at each epoch
    # avg_f1_per_epoch = {epoch: np.mean(f1s) for epoch, f1s in f1_scores_across_runs.items()}
    # epochs = sorted(avg_f1_per_epoch.keys())
    # avg_f1s = [avg_f1_per_epoch[e] for e in epochs]

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, avg_f1s, marker='o')
    # plt.title("Average F1 Score per Epoch over 10 Runs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Average F1 Score")
    # plt.grid(True)
    # plt.show()


##################################### This section was used for the training run figure



    num_runs = 10
    num_epochs = 150
    all_losses = []

    loaded_data = []
    with open('all_data_12_06.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vec = list(map(float, row[:-1]))
            label = int((row[-1]))
            loaded_data.append((vec, label))

    for run in range(num_runs):
            
        
        # DOWNSAMPLING
        zeros = [item for item in loaded_data if item[1] == 0]
        ones = [item for item in loaded_data if item[1] == 1]

        
        num_zeros = len(zeros)
        num_ones = len(ones)

        # We leave ratio equal as this didn't work out
        (parts_zeros, parts_ones) = (4.0, 4.0)


        if num_zeros > (parts_zeros / parts_ones) * num_ones:
            num_zeros_target = int((parts_zeros / parts_ones) * num_ones)
            zeros = sample(zeros, num_zeros_target)
        elif num_ones > (parts_ones / parts_zeros) * num_zeros:
            num_ones_target = int((parts_ones / parts_zeros) * num_zeros)
            ones = sample(ones, num_ones_target)

        balanced_data = zeros + ones

        shuffle(balanced_data)


        # Partition into the 80:20 split
        def train_test_split(data, split=0.8):
            split_idx = int(len(data) * split)
            return data[:split_idx], data[split_idx:]

        zeros_train, zeros_test = train_test_split(zeros)
        ones_train, ones_test = train_test_split(ones)

        train_data = zeros_train + ones_train
        test_data = zeros_test + ones_test
        shuffle(train_data)


        train_dataset = SimilarityDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


        model = SimilarityNN(NN_INPUT_DIM)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

        losses = []

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for vectors, labels in train_loader:
                outputs = model(vectors).squeeze()
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Run {run+1} Epoch {epoch+1} Loss: {avg_loss:.4f}")

        all_losses.append(losses)

    

    # Data analysis below
    all_losses = np.array(all_losses)  # (10 runs, num_epochs)
    mean_loss = np.mean(all_losses, axis=0)
    std_loss = np.std(all_losses, axis=0)


    # Plot mean and st. dev.
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, mean_loss, label='Mean Loss')
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3, label='Standard Deviation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean and Standard Deviation BCE Loss Over 10 Training Runs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





# ######################################### For main training (plotting the lose over a single traning run)


    loaded_data = []
    with open('all_data_12_06.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vec = list(map(float, row[:-1]))
            label = int((row[-1]))
            loaded_data.append((vec, label))



    # DOWNSAMPLING
    zeros = [item for item in loaded_data if item[1] == 0]
    ones = [item for item in loaded_data if item[1] == 1]

    num_zeros = len(zeros)
    num_ones = len(ones)

    # Ratio is balanced
    (parts_zeros, parts_ones) = (4.0, 4.0)

    if num_zeros > (parts_zeros / parts_ones) * num_ones:
        num_zeros_target = int((parts_zeros / parts_ones) * num_ones)
        zeros = sample(zeros, num_zeros_target)
    elif num_ones > (parts_ones / parts_zeros) * num_zeros:
        num_ones_target = int((parts_ones / parts_zeros) * num_zeros)
        ones = sample(ones, num_ones_target)

    balanced_data = zeros + ones
    shuffle(balanced_data)

    # Split into 80:20 partitions
    def train_test_split(data, split=0.8):
        split_idx = int(len(data) * split)
        return data[:split_idx], data[split_idx:]

    zeros_train, zeros_test = train_test_split(zeros)
    ones_train, ones_test = train_test_split(ones)

    train_data = zeros_train + ones_train
    test_data = zeros_test + ones_test

    shuffle(train_data)
    shuffle(test_data)

    def save_csv(data, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for vec, label in data:
                writer.writerow(vec + [label])

    save_csv(train_data, 'train_data_latest.csv')
    save_csv(test_data, 'test_data_latest.csv')

    train_data = []
    with open('train_data_latest.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vec = list(map(float, row[:-1]))
            label = int((row[-1]))
            train_data.append((vec, label))



    class SimilarityDataset(Dataset):
        def __init__(self, data):
            self.vectors = [torch.tensor(v, dtype=torch.float32) for v, _ in data]
            self.labels = [torch.tensor(l, dtype=torch.float32) for _, l in data]

        def __len__(self):
            return len(self.vectors)

        def __getitem__(self, idx):
            return self.vectors[idx], self.labels[idx]



    train_dataset = SimilarityDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = len(train_data[0][0])
    model = SimilarityNN(input_dim)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    num_epochs = 120
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for vectors, labels in train_loader:
            outputs = model(vectors).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            model.eval()

            fp = 0
            tp = 0
            tn = 0
            fn = 0

            for (vector, gt) in test_data:

                input_vector = torch.tensor(vector, dtype=torch.float32)
                input_vector = input_vector.unsqueeze(0) 


                with torch.no_grad():
                    output = model(input_vector)
                    prediction = output.item()


                    if prediction > 0.5 and gt == 1:
                        tp += 1
                    elif prediction > 0.5 and gt == 0:
                        fp += 1
                    elif prediction <= 0.5 and gt == 0:
                        tn += 1
                    else:
                        fn += 1
            
            f1 = (tp) / (tp + 0.5 * (fp + fn))
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            print("tp: " + str(tp) + " fp: " + str(fp) + " tn: " + str(tn) + "fn: " + str(fn) + "   accuracy: " + str(accuracy))

            model.train()
    
    
    # Plot the loss
    torch.save(model.state_dict(), 'random.pth')
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()
