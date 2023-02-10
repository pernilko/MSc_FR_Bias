import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def evaluate_performance(model, test_data_loader):

    # no gradients needed
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            print("labels: ", labels)
            outputs = model(images)
            print("outputs:", outputs)
            _, predictions = torch.max(outputs, 1)
            print("Predict: ", predictions)

            class_id, true_positives, false_positives, true_negatives, false_negatives = performance_measure(labels, predictions)
            print("class_id: ", class_id, "TP: ", true_positives, "FP: ", false_positives, "TN: ", true_negatives, "FN: ", false_negatives)

            far, frr = calculate_far_frr(false_positives, false_negatives, true_negatives, true_positives)
            print ("FAR: ", far, "FRR: ", frr)
    
    

def calculate_far_frr(false_positives, false_negatives, true_negatives, true_positives):
    far = false_positives/float(false_positives + true_negatives) 
    frr = false_negatives/float(false_negatives + true_positives)
    return far, frr

def performance_measure(actual_labels, predicted_labels):
    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []
    class_id = set(actual_labels).union(set(predicted_labels))

    for index ,_id in enumerate(class_id):
        true_positives.append(0)
        false_positives.append(0)
        true_negatives.append(0)
        false_negatives.append(0)
        for i in range(len(predicted_labels)):
            if actual_labels[i] == predicted_labels[i] == _id:
                true_positives[index] += 1
            if predicted_labels[i] == _id and actual_labels[i] != predicted_labels[i]:
                false_positives[index] += 1
            if actual_labels[i] == predicted_labels[i] != _id:
                true_negatives[index] += 1
            if predicted_labels[i] != _id and actual_labels[i] != predicted_labels[i]:
                false_negatives[index] += 1

    return(class_id, true_positives, false_positives, true_negatives, false_negatives)

def create_distribution_plot(df : pd.DataFrame, output_filename : str):
    # need to take in a dataFrame which contains three columns named "mated", "agevsage" and "non-mated".
    # Each row must contain one identity with values pertaining to the three columns, 
    # i.e. the identitiy's similarity score for a mated sample, sim score compared to mated but older, and non-mated.

    similarity_id_1 = [0.89, 0.66, 0.001]
    similarity_id_2 = [0.9, 0.56, 0.1]
    similarity_id_3 = [0.83, 0.75, 0.2]
    similarity_id_4 = [0.86, 0.67, 0.02]
    df = pd.DataFrame(np.array((similarity_id_1, similarity_id_2, similarity_id_3, similarity_id_4)),
                   columns=['mated', '20vs50', 'non-mated'])
    print(df.head())
    sns.displot(df, kind="kde")
    plt.savefig('plot_test.png')

#create_distribution_plot()