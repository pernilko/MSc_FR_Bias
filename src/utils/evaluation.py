import torch

def evaluate_performance(model, test_data_loader):

    # no gradients needed
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            print("labels: ", labels)
            outputs = model(images)
            print("outputs:", outputs)
            _, predictions = torch.max(outputs)
            print("Predict: ", predictions)

            true_positives, false_positives, true_negatives, false_negatives = performance_measure(labels, predictions)
            print("TP: ", true_positives, "FP: ", false_positives, "TN: ", true_negatives, "FN: ", false_negatives)

            far, frr = calculate_far_frr(false_positives, false_negatives, true_negatives, true_positives)
            print ("FAR: ", far, "FRR: ", frr)
    
    

def calculate_far_frr(false_positives, false_negatives, true_negatives, true_positives):
    far = false_positives/float(false_positives + true_negatives) 
    frr = false_negatives/float(false_negatives + true_positives)
    return far, frr

def performance_measure(actual_labels, predicted_labels):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(len(predicted_labels)): 
        if actual_labels[i]==predicted_labels[i]==1:
           true_positives += 1
        if predicted_labels[i]==1 and actual_labels[i]!=predicted_labels[i]:
           false_positives += 1
        if actual_labels[i]==predicted_labels[i]==0:
           true_negatives += 1
        if predicted_labels[i]==0 and actual_labels[i]!=predicted_labels[i]:
           false_negatives += 1

    return(true_positives, false_positives, true_negatives, false_negatives)