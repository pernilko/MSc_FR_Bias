import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
import math

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

            far, frr = calculate_far_frr(np.sum(false_positives), np.sum(false_negatives), np.sum(true_negatives), np.sum(true_positives))
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

'''
def calculate_similarity_score(orginal_img, img_to_compare):
    similarity_score = 1- np.linalg.norm(orginal_img.detach().numpy()-img_to_compare.detach().numpy())
    return similarity_score
'''

def calculate_similarity_score(embeddings1, embeddings2, distance_type='Cosine'):
    if distance_type=='Euclidian':
        # Euclidian distance
        embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=0, keepdims=True)
        embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=0, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),0)
    elif distance_type=='Cosine':
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=0)
        norm = np.linalg.norm(embeddings1, axis=0) * np.linalg.norm(embeddings2, axis=0)
        similarity = dot/norm
        similarity = min(1,similarity)
        #dist=1-similarity
    else:
        raise 'Undefined distance metric %d' # distance_metric 
    return similarity

def test(test_data_loader, model : iresnet50):

    sim_scores = []
    for i, data in enumerate(test_data_loader):
            vinputs, vlabels = data
            voutputs = model(vinputs)

            # find identities
            identity_start_indicies = []
            identities = []
            for i in range(0, len(voutputs)):
                if vlabels[i] not in identities:
                    #print(vlabels[i])
                    identity_start_indicies.append(i)
                    identities.append(vlabels[i])
            #print("identity indicies: ", identity_start_indicies)
            #print("identities: ", identities )

            
            for identity_index in identity_start_indicies:
                sim_score_identity = []
                sim_score_mated = []
                sim_score_non_mated = []
                sim_score_age_mated = []
                for mated_img in range(identity_index + 1,identity_index + 4):
                    if vlabels[identity_index] == vlabels[mated_img] and mated_img < identity_index + 3:
                        print("mated match")
                        output1 = voutputs[identity_index]
                        output2 = voutputs[mated_img]
                        distance = calculate_similarity_score(output1, output2)
                        print("mated dist: " ,distance)
                        sim_score_mated.append(distance)
                        #sim_score_identity.append(distance)
                    if vlabels[identity_index] == vlabels[mated_img] and mated_img >= identity_index + 3:
                        print("age-mated match")
                        output1 = voutputs[identity_index]
                        output2 = voutputs[mated_img]
                        distance = calculate_similarity_score(output1, output2)
                        sim_score_age_mated.append(distance)
                        #sim_score_identity.append(distance)
                for non_mated_img in identity_start_indicies:
                    if vlabels[non_mated_img] != vlabels[identity_index]:
                        print("non-mated match")
                        output1 = voutputs[identity_index]
                        output2 = voutputs[non_mated_img]
                        distance = calculate_similarity_score(output1, output2)
                        sim_score_non_mated.append(distance)
                        #sim_score_identity.append(distance)
                
                sim_score_identity.append(np.mean(sim_score_mated))
                sim_score_identity.append(np.mean(sim_score_age_mated))
                sim_score_identity.append(np.mean(sim_score_non_mated))
                sim_scores.append(sim_score_identity)
            '''
            sim_scores = []
            counter = 1
            for i in range(0,len(vinputs)):
                sim_score_identity = []
                sim_score_mated = []
                sim_score_non_mated =[]
                sim_score_age_mated = []
                for j in range(i+1, len(vinputs)):
                    if vlabels[i] == vlabels[j] and counter < 3: #mated
                        print("mated match", "counter: ", counter)
                        output1 = voutputs[i]
                        output2 = voutputs[j]
                        distance = calculate_similarity_score(output1, output2)
                        t : bool = np.isnan(distance)
                        if t:
                            print("label 1: ", vlabels[i], "label 2: ", vlabels[j], "out 1: ", output1, "out 2: ", output2)
                        sim_score_mated.append(distance)

                    if vlabels[i] != vlabels[j]: # non-mated
                        print("non-mated match", "counter: ", counter)
                        output1 = voutputs[i]
                        output2 = voutputs[j]
                        distance = calculate_similarity_score(output1, output2)
                        t : bool = np.isnan(distance)
                        if t:
                            print("label 1: ", vlabels[i], "label 2: ", vlabels[j], "out 1: ", output1, "out 2: ", output2)
                        sim_score_non_mated.append(distance)
                    if vlabels[i] == vlabels[j] and j > 2: #age-mated
                        print("age-mated match", "counter: ", counter) 
                        output1 = voutputs[i]
                        output2 = voutputs[j]
                        distance = calculate_similarity_score(output1, output2)
                        t : bool = np.isnan(distance)
                        if t:
                            print("label 1: ", vlabels[i], "label 2: ", vlabels[j], "out 1: ", output1, "out 2: ", output2)
                        sim_score_age_mated.append(distance)
                    
                    if counter == 4:
                        counter = 1
                    else:
                        counter = counter + 1
                sim_score_identity.append(np.mean(sim_score_mated))
                sim_score_identity.append(np.mean(sim_score_age_mated))
                sim_score_identity.append(np.mean(sim_score_non_mated))
                sim_scores.append(sim_score_identity)
    '''
    print("sim scores mated: ", sim_scores, "end sim scores")
    return sim_scores

def create_dataframe(similarity_scores):
    df = pd.DataFrame(similarity_scores, columns=['mated', '20vs65', 'non-mated'])
    return df
        
        
    # for each identity:
        # find sim score for mated sample
        # find  sim score for 20 vs 50 year old
        # find avg? sim score for non-mated sample
    #df = pd.DataFrame(sim_scores, columns=['mated', '20vs50', 'non-mated'])

    #return df

def create_distribution_plot(df : pd.DataFrame, output_filename : str):
    # need to take in a dataFrame which contains three columns named "mated", "agevsage" and "non-mated".
    # Each row must contain one identity with values pertaining to the three columns, 
    # i.e. the identitiy's similarity score for a mated sample, sim score compared to mated but older, and non-mated.

    '''
    similarity_id_1 = [0.89, 0.66, 0.001]
    similarity_id_2 = [0.9, 0.56, 0.1]
    similarity_id_3 = [0.83, 0.75, 0.2]
    similarity_id_4 = [0.86, 0.67, 0.02]
    df = pd.DataFrame(np.array((similarity_id_1, similarity_id_2, similarity_id_3, similarity_id_4)),
                   columns=['mated', '20vs50', 'non-mated'])
    '''
    
    #print(df.head())
    sns.displot(df, kind="kde")
    plt.savefig(output_filename)

#create_distribution_plot()