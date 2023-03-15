import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
from data.data_preprocessing import load_test_dataset
from torch.utils.data import DataLoader
import math
import os
from torchvision import transforms
from data.data_preprocessing import orgranize_fgnet_dataset

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
    embeddings1 = embeddings1.detach().numpy()
    embeddings2 = embeddings2.detach().numpy()
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

def compute_similarity_scores_for_test_dataset(test_data_loader, model : iresnet50):

    sim_scores = []
    for i, data in enumerate(test_data_loader):
            vinputs, vlabels = data
            voutputs = model(vinputs)

            # find identities
            identity_start_indicies = []
            identities = []
            for i in range(0, len(voutputs)):
                if vlabels[i] not in identities:
                    identity_start_indicies.append(i)
                    identities.append(vlabels[i])
            
            for identity_index in identity_start_indicies:
                sim_score_identity = []
                sim_score_mated = []
                sim_score_non_mated = []
                sim_score_age_mated = []
                for mated_img in range(identity_index + 1,identity_index + 4):
                    if vlabels[identity_index] == vlabels[mated_img] and mated_img < identity_index + 3:
                        #print("mated match")
                        output1 = voutputs[identity_index]
                        output2 = voutputs[mated_img]
                        distance = calculate_similarity_score(output1, output2)
                        sim_score_mated.append(distance)
                    
                    if vlabels[identity_index] == vlabels[mated_img] and mated_img >= identity_index + 3:
                        #print("age-mated match")
                        output1 = voutputs[identity_index]
                        output2 = voutputs[mated_img]
                        distance = calculate_similarity_score(output1, output2)
                        sim_score_age_mated.append(distance)
                        
                for non_mated_img in identity_start_indicies:
                    if vlabels[non_mated_img] != vlabels[identity_index]:
                        #print("non-mated match")
                        output1 = voutputs[identity_index]
                        output2 = voutputs[non_mated_img]
                        distance = calculate_similarity_score(output1, output2)
                        sim_score_non_mated.append(distance)
                        
                sim_score_identity.append(np.mean(sim_score_mated))
                sim_score_identity.append(np.mean(sim_score_age_mated))
                sim_score_identity.append(np.mean(sim_score_non_mated))
                sim_scores.append(sim_score_identity)

    return sim_scores

def get_filename(data_loader : DataLoader):
    filenames = []
    filename_class = data_loader.dataset.imgs
    for i in filename_class:
        fname = i[0].split('/')[3]
        class_name_pair = []
        class_name_pair.append(i[1])
        class_name_pair.append(fname.split('.')[0])
        filenames.append(class_name_pair)

    return filenames

def get_filenames_by_identity(id : int, filenames : list):
    filenames_id = []
    for i in filenames:
        if i[0] == id:
            filenames_id.append(i[1])
    return filenames_id

def get_filenames_by_batch(batch_size : int, batch_number : int, filenames : list):
    num_of_filenames = len(filenames)
    imgs_per_batch = batch_size
    batch_of_filenames = filenames[batch_number*imgs_per_batch:batch_number*imgs_per_batch+imgs_per_batch]
    
    return batch_of_filenames


def compute_sim_scores_fg_net(test_data_loader : DataLoader, model : iresnet50, outdir_plot : str):
    filenames = get_filename(test_data_loader)
    
    sim_scores = []
    for i, data in enumerate(test_data_loader):
        vinputs, vlabels = data
        voutputs = model(vinputs)
        batch_of_filenames = get_filenames_by_batch(len(vlabels), i, filenames)
        identities_idx  = np.array(torch.unique(torch.tensor(vlabels)))


        for idx in identities_idx: # for each identity
            age_mated_young_outputs = []
            age_mated_old_outputs = []
            mated_old_outputs = []
            mated_middle_outputs = []
            mated_young_outputs = []
            non_mated_outputs = []
            non_mated_template_outputs = []


            identity_sim_score = []

            for index in range(0,len(voutputs)): #for each index in outputs
               
                #print("len vout: ", len(voutputs))
                #print(len(batch_of_filenames))
                current_age = batch_of_filenames[index]
                current_age = batch_of_filenames[index][1].split('_')[1]
                current_age = int(current_age.translate({ord(i): None for i in 'abcdefghijklmnopqrstuvwxyz'}))
             
                if vlabels[index] == idx and (current_age >= 20 and current_age <= 30):  # age-mated young
                    #print("young age mated", current_age)
                    mated_young = voutputs[index]
                    age_mated_young_outputs.append(mated_young)
                if vlabels[index] == idx and (current_age >= 50 and current_age <= 70): #age-mated old
                    #print("old age mated ", current_age)
                    mated_old = voutputs[index]
                    age_mated_old_outputs.append(mated_old)
                if vlabels[index] == idx and (current_age >= 10 and current_age <= 15): #mated young
                    #print("young mated ", current_age)
                    young_out = voutputs[index]
                    mated_young_outputs.append(young_out)
                if vlabels[index] == idx and (current_age >= 20 and current_age <=40): #mated middle
                    #print("middle mated ", current_age)
                    middle_out = voutputs[index]
                    mated_middle_outputs.append(middle_out)
                if vlabels[index] == idx and (current_age >= 45 and current_age <= 65): # mated old
                    #print("old mated ", current_age, ", id: ", idx)
                    old_out = voutputs[index]
                    mated_old_outputs.append(old_out)
                if vlabels[index] == idx and (current_age >= 20 and current_age <= 30):
                    template_output = voutputs[index]
                    non_mated_template_outputs.append(template_output)
                if vlabels[index] != idx and (current_age >= 20 and current_age <= 30):
                    #print("non-mated ",current_age)
                    non_mated_out = voutputs[index]
                    non_mated_outputs.append(non_mated_out)

            
            identity_mated_young = calculate_mated_sim_scores(mated_young_outputs)
            identity_mated_middle = calculate_mated_sim_scores(mated_middle_outputs)
            identity_mated_old = calculate_mated_sim_scores(mated_old_outputs)
            identity_age_mated = calculate_age_mated_sim_scores(age_mated_young_outputs, age_mated_old_outputs)
            identity_non_mated = calculate_non_mated_sim_scores(non_mated_template_outputs, non_mated_outputs)

            identity_sim_score.append(np.mean(identity_mated_young)) # mated
            identity_sim_score.append(np.mean(identity_mated_middle)) # mated
            identity_sim_score.append(np.mean(identity_mated_old)) # mated
            identity_sim_score.append(np.mean(identity_age_mated)) # age-mated
            identity_sim_score.append(np.mean(identity_non_mated)) # non-mated

            sim_scores.append(identity_sim_score)
    print(sim_scores)

    df = create_dataframe_fg_net(sim_scores)
    create_distribution_plot(df, outdir_plot)

    return sim_scores

def calculate_age_mated_sim_scores(young_outputs, old_outputs):
    age_mated_sim_scores = []
    for young_output in young_outputs:
        for old_output in old_outputs:
            sim_score = calculate_similarity_score(young_output, old_output)
            age_mated_sim_scores.append(sim_score)
    return age_mated_sim_scores

def calculate_mated_sim_scores(outputs):
    mated_sim_scores = []
    for i in range(len(outputs)-1):
        sim_score = calculate_similarity_score(outputs[i], outputs[i+1])
        mated_sim_scores.append(sim_score)
    return mated_sim_scores

def calculate_non_mated_sim_scores(template_outputs, non_mated_outputs):
    non_mated_sim_scores = []
    for template in template_outputs:
        for non_mated in non_mated_outputs:
            sim_score = calculate_similarity_score(template, non_mated)
            non_mated_sim_scores.append(sim_score)
    return non_mated_sim_scores

def create_dataframe_fg_net(sim_scores):
    df =  pd.DataFrame(sim_scores, columns=['mated young', 'mated middle', 'mated old', 'YoungvsOld', 'non-mated'])
    return df

def create_dataframe(similarity_scores):
    df = pd.DataFrame(similarity_scores, columns=['mated', '20vs65', 'non-mated'])
    return df

def create_distribution_plot(df : pd.DataFrame, output_path : str):
    # need to take in a dataFrame which contains three columns named "mated", "agevsage" and "non-mated".
    # Each row must contain one identity with values pertaining to the three columns, 
    # i.e. the identitiy's similarity score for a mated sample, sim score compared to mated but older, and non-mated.

    os.makedirs(output_path, exist_ok=True)
    sns.displot(df, kind="kde")
    plt.savefig(f"{output_path}/distribution_plot.png")

def G(x, y):
    n = len(y)
    g = ((n)/(n-1))
    return g

# FMNR
def B(threshold):
    b = 0
    return b

# FMR
def A(threshold : float):
    a = 0
    return a

def GARBE(A, B, alpha = 0.5):
    return alpha*A + (1-alpha)*B

'''


tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((98, 98))])
outdir_plot = "plots/test/"
data_loader = load_test_dataset("datasets/fgnet/", 1002, tsfm)
model = iresnet50()
compute_sim_scores_fg_net(data_loader, model, outdir_plot)
'''

