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
from PIL import Image
import operator


'''
Method for loading the bare ArcFace model

Parameters: 
    filename (string) : the path for the file containing the pre-trained ArcFace model
    device (torch.device) : specifies whether to use cpu or cuda
Return: 
    model () : the pre-trained ArcFace model
'''
def load_arc_face_model(filename : str, device : torch.device):
    loaded_model = torch.load(filename,  map_location = device)
    model = iresnet50()
    model.load_state_dict(loaded_model, strict=True)

    return model

def get_arcface_sim_scores(model_filename : str, test_data_loader):
    device = torch.device('cuda:0')
    # Fetching arcface model
    model = load_arc_face_model(model_filename, device)
    model.train(False)

    sim_scores = compute_sim_scores_fg_net(test_data_loader, model, "", 0, False)

    return sim_scores


"""
Computes the false non-match rates and false match rates for various thresholds

Input: Mated comparison scores & Non-Mated comparison scores    
    - gscores: mated comparison scores
    - iscores: non-mated comparison scores
    - ds_scores: False means that input scores are similarity scores 
Output: Thresholds with corresponding FMRs and FNMRs
"""
def calculate_roc(gscores, iscores, ds_scores=False, rates=True):

    if isinstance(gscores, list):
        gscores = np.array(gscores, dtype=np.float64)

    if isinstance(iscores, list):
        iscores = np.array(iscores, dtype=np.float64)

    if gscores.dtype == np.int:
        gscores = np.float64(gscores)

    if iscores.dtype == np.int:
        iscores = np.float64(iscores)

    if ds_scores:
        gscores = gscores * -1
        iscores = iscores * -1

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    gscores = zip(gscores, [1] * gscores_number)
    iscores = zip(iscores, [0] * iscores_number)

    gscores = list(gscores)
    iscores = list(iscores)

    scores = np.array(sorted(gscores + iscores, key=operator.itemgetter(0)))
    cumul = np.cumsum(scores[:, 1])

    thresholds, u_indices = np.unique(scores[:, 0], return_index=True)

    fnm = cumul[u_indices] - scores[u_indices][:, 1]
    fm = iscores_number - (u_indices - fnm)

    if rates:
        fnm_rates = fnm / gscores_number
        fm_rates = fm / iscores_number
    else:
        fnm_rates = fnm
        fm_rates = fm

    if ds_scores:
        return thresholds * -1, fm_rates, fnm_rates

    return thresholds, fm_rates, fnm_rates

"""
Computes the false non-match rates and false match rates for various thresholds

Input: Mated comparison scores & Non-Mated comparison scores
    - gscores: mated comparison scores
    - iscores: non-mated comparison scores
    - ds_scores: False means that input scores are similarity scores
    - threshold: threshold value for classification
Output: FMR and FNMR for the given threshold
"""
def calculate_fmr_fnmr_with_threshold(gscores, iscores, threshold, ds_scores=False):
    gscores_number = len(gscores)
    iscores_number = len(iscores)

    fnm = 0
    for score in gscores:
        if score < threshold:
            fnm += 1

    fm = 0
    for score in iscores:
        if score >= threshold:
            fm += 1

    fnmr = fnm / gscores_number
    fmr = fm / iscores_number

    if ds_scores:
        return fmr, fnmr

    return fnmr, fmr

def calculate_fmr_fnmr_test(mated_scores, non_mated_scores, threshold):
    # Convert similarity scores to binary match/non-match labels based on threshold
    mated_labels = np.where(mated_scores >= threshold, 1, 0)
    non_mated_labels = np.where(non_mated_scores >= threshold, 1, 0)

    # Handle NaN values by setting them to 0.5 (i.e., neutral threshold)
    mated_labels[np.isnan(mated_labels)] = 0.5
    non_mated_labels[np.isnan(non_mated_labels)] = 0.5

    # Calculate FMR and FNMR
    fmr = np.nanmean(non_mated_labels) # False Match Rate = Non-Match Labels / Total Non-Match Scores
    fnmr = 1 - np.nanmean(mated_labels) # False Non-Match Rate = 1 - Match Labels / Total Match Scores

    return fmr, fnmr


# fmrs: list with the different fmrs
# fnmrs: list with the different fnmrs                   [young_fmr, old_fmr] and [young_fnmr, old_fnmr]          
# fmrs and fnmrs should have the same order: e.g. fmrs = [female_fmr, male_fmr] and fnmrs = [female_fnmr, male_fnmr]
# alpha: weighting alpha=1: only fmr; alpha=0: only fnmr; alpha=0.5: fnmr and fmr same weight
def gini_aggregation_rate(fmrs, fnmrs, alpha=0.5):
    x_fmr = fmrs
    x_fnmrs = fnmrs
    n = len(x_fmr)
    sum = 0
    
    for xi in x_fmr:
        for xj in x_fmr:
            sum += np.abs(xi - xj)

    g_fmr = (n / (n - 1)) * (sum / (2 * np.power(n, 2) * np.mean(x_fmr)))

    # false non matches
    n = len(x_fnmrs)
    sum = 0
    for xi in x_fnmrs:
        for xj in x_fnmrs:
            sum += np.abs(xi - xj)

    g_fnmr = (n / (n - 1)) * (sum / (2 * np.power(n, 2) * np.mean(x_fnmrs)))
    
    garbe = alpha * g_fmr + (1 - alpha) * g_fnmr
    return garbe


def evaluate_fairness(model, test_data_loader : DataLoader, experiment_name : str, current_epoch_num : int):
    filenames = get_filename(test_data_loader)
    output_dir = f"experiments/{experiment_name}/garbe"
    os.makedirs(output_dir, exist_ok=True)

    young_mated_sim_score = []
    old_mated_sim_score = []
    young_non_mated_sim_score = []
    old_non_mated_sim_score = []
    for i, data in enumerate(test_data_loader):
        vinputs, vlabels = data
        voutputs = model(vinputs)
        batch_of_filenames = get_filenames_by_batch(len(vlabels), i, filenames)
        identities_idx  = np.array(torch.unique(torch.tensor(vlabels)))


        for idx in identities_idx: # for each identity
            non_mated_young_list = []
            mated_young_list = []
            mated_old_list = []
            non_mated_old_list = []

            for index in range(0,len(voutputs)): #for each index in outputs
                current_age = batch_of_filenames[index]
                current_age = batch_of_filenames[index][1].split('_')[1]
                current_age = int(current_age.translate({ord(i): None for i in 'abcdefghijklmnopqrstuvwxyz'}))
             
                if vlabels[index] == idx and (current_age >= 5 and current_age <= 20):
                    #print("young mated", current_age)
                    mated_young = voutputs[index]
                    mated_young_list.append(mated_young)
                if vlabels[index] != idx and (current_age >= 5 and current_age <= 20): 
                    #print("young non-mated ", current_age)
                    non_mated_young = voutputs[index]
                    non_mated_young_list.append(non_mated_young)
                if vlabels[index] == idx and (current_age >= 45 and current_age <= 70):
                    mated_old = voutputs[index]
                    mated_old_list.append(mated_old)
                if vlabels[index] != idx and (current_age >= 45 and current_age <= 70):
                    non_mated_old = voutputs[index]
                    non_mated_old_list.append(non_mated_old)

            identity_non_mated_young = calculate_non_mated_sim_scores(mated_young_list, non_mated_young_list)
            identity_non_mated_old = calculate_non_mated_sim_scores(mated_old_list, non_mated_old_list)
            identity_mated_young = calculate_mated_sim_scores(mated_young_list)
            identity_mated_old = calculate_mated_sim_scores(mated_old_list)

            young_mated_sim_score.append(np.mean(identity_mated_young))
            old_mated_sim_score.append(np.mean(identity_mated_old))
            young_non_mated_sim_score.append(np.mean(identity_non_mated_young))
            old_non_mated_sim_score.append(np.mean(identity_non_mated_old))

    tresholds_young, fmr_young, fnmr_young = calculate_roc(young_mated_sim_score, young_non_mated_sim_score)
    tresholds_old, fmr_old, fnmr_old = calculate_roc(old_mated_sim_score, old_non_mated_sim_score)

    f = open(f"{output_dir}/garbe_metrics_.txt", "a")
    f.write(f"Epoch {current_epoch_num}: \n")
    f.close()
    thresholds_test = np.arange(0, 1, step=0.025)
    garbe_values_test = []
    thresholds_test_non_nan = []
    for t in thresholds_test:
        young_fmr_test, young_fnmr_test = calculate_fmr_fnmr_test(young_mated_sim_score, young_non_mated_sim_score, t)
        old_fmr_test, old_fnmr_test = calculate_fmr_fnmr_test(old_mated_sim_score, old_non_mated_sim_score, t)
        #print(f"Threshold = {t}: FMR Young = {young_fmr_test}, FNMR Young = {young_fnmr_test}")
        #print(f"Threshold = {t}: FMR Old = {old_fmr_test}, FNMR Old = {old_fnmr_test}")
        fmrs_test = [young_fmr_test, old_fmr_test]
        fnmrs_test = [young_fnmr_test, old_fnmr_test]
        garbe_test = gini_aggregation_rate(fmrs_test, fnmrs_test)
        #print(f"GARBE({t}) = {garbe_test}")
        f = open(f"{output_dir}/garbe_metrics_.txt", "a")
        f.write(f"Threshold = {t}: FMR Young = {young_fmr_test}, FNMR Young = {young_fnmr_test}\n")
        f.write(f"Threshold = {t}: FMR Old = {old_fmr_test}, FNMR Old = {old_fnmr_test}\n")
        f.write(f"GARBE({t}) = {garbe_test}\n")
        f.close()
        if np.isnan(garbe_test) == False:
            thresholds_test_non_nan.append(t)
            garbe_values_test.append(garbe_test)
        
    
    plt.figure()
    plt.scatter(thresholds_test_non_nan, garbe_values_test)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.xlabel("Threshold")
    plt.ylabel("GARBE")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/garbe_plot_{current_epoch_num}.pdf")

    '''
    f = open(f"{output_dir}/garbe_metrics_.txt", "a")
    f.write(f"Epoch {current_epoch_num}: \n")
    f.close()
    threshold_values = []
    garbe_values = []
    for t_y in range(len(tresholds_young)):
        if math.isnan(tresholds_young[t_y])== False:
            for t_o in range(len(tresholds_old)):
                if math.isnan(tresholds_old[t_o])== False:
                    if round(tresholds_young[t_y], 2) == round(tresholds_old[t_o], 2):
                        fmrs = [fmr_young[t_y], fmr_old[t_o]]
                        fnmrs = [fnmr_young[t_y], fnmr_old[t_o]]
                        garbe = gini_aggregation_rate(fmrs,fnmrs)
                        threshold_values.append(round(tresholds_young[t_y], 2))
                        garbe_values.append(garbe)
                        #print(f"GARBE for threshold = {round(tresholds_young[t_y], 3)}, {round(tresholds_old[t_o], 3)}: {garbe}")
                        f = open(f"{output_dir}/garbe_metrics_.txt", "a")
                        f.write(f"GARBE({round(tresholds_young[t_y], 2)}) = {garbe}\n")
                        f.close()

    plt.figure()
    plt.scatter(threshold_values, garbe_values)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.xlabel("Threshold")
    plt.ylabel("GARBE")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/garbe_plot_{current_epoch_num}.pdf")
    '''
    return garbe_test


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
Method calculates similarity score between two embeddings using either Euclidian or Cosine distance
Parameters:
    embeddings1 () : the first embedding
    embeddings2 () : the second embedding
    distance_type (str) : the distance measure to be used. Default is Cosine.
Return:
    similarity () : the similarity of the two embeddings
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

'''
Method for computing the similarity score for the cusp generated dataset (only for testing purposes)

Parameters:
    test_data_loader (DataLoader) : dataloader containing the test dataset
    model (iresnet50) : model that the test dataset is to be test on
Return:
    sim_scores (list) : similarity scores for the test dataset
'''
def compute_similarity_scores_for_test_dataset(test_data_loader : DataLoader, model : iresnet50):

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

'''
Method for getting all filenames from the dataloader

Parameters:
    data_loader (DataLoader) : dataloader containing the dataset
Return:
    filenames (list) : all filenames in the dataloader
'''
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

'''
Method for getting the filenames for the current batch

Parameters:
    batch_size (int) : the size of the batches
    batch_number (int) : the number of the current batch
    filenames (list) : the list of all filenames in the dataloader
Return:
    batch_of_filenames (list) : the list of filenames in the current batch
'''
def get_filenames_by_batch(batch_size : int, batch_number : int, filenames : list):
    num_of_filenames = len(filenames)
    imgs_per_batch = batch_size
    batch_of_filenames = filenames[batch_number*imgs_per_batch:batch_number*imgs_per_batch+imgs_per_batch]
    
    return batch_of_filenames

'''
Method computes similarity scores for the FG-Net dataset.
A distribution plot is created from the similarity scores.

Parameters:
    test_data_loader (DataLoader) : dataloader containing the test dataset
    model (iresnet50) : the model used to predict outputs on the test dataset
    outdir_plot (str) : the path for where the distribution plots are to be saved
Return:
    sim_scores (list) : the similarity scores

'''
def compute_sim_scores_fg_net(test_data_loader : DataLoader, model, outdir_plot : str, epoch_num : int, plot=True):
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
    if plot:
        df = create_dataframe_fg_net(sim_scores)
        create_distribution_plot(df, outdir_plot, epoch_num)
        create_arcface_vs_finetuned_plot(sim_scores, test_data_loader, outdir_plot, epoch_num)

    return sim_scores

'''
Method for calculating similarity scores for age-mated embeddings

Parameters:
    young_outputs () : the embeddings of young images of the identity
    old_outputs () : the embeddings of old images of the identity
Return:
    age_mated_sim_scores (list) : the simlarity scores for the age-mated embeddings of one identity

'''
def calculate_age_mated_sim_scores(young_outputs, old_outputs):
    age_mated_sim_scores = []
    for young_output in young_outputs:
        for old_output in old_outputs:
            sim_score = calculate_similarity_score(young_output, old_output)
            age_mated_sim_scores.append(sim_score)
    return age_mated_sim_scores

'''
Method for calculating the similarity scores of mated samples

Parameters:
    outputs () : the embeddings of mated samples
Return:
    mated_sim_scores (list): the similarity scores of the mated samples
'''
def calculate_mated_sim_scores(outputs):
    mated_sim_scores = []
    for i in range(len(outputs)-1):
        sim_score = calculate_similarity_score(outputs[i], outputs[i+1])
        mated_sim_scores.append(sim_score)
    return mated_sim_scores

'''
Method for calculating the similarity score of non-mated samples

Parameters:
    template_outputs () : the embeddings of the current identity
    non_mated_outputs (): the embeddings of non-mated samples
Return:
    non_mated_sim_scores (list) : similarity scores for non-mated samples 
'''
def calculate_non_mated_sim_scores(template_outputs, non_mated_outputs):
    non_mated_sim_scores = []
    for template in template_outputs:
        for non_mated in non_mated_outputs:
            sim_score = calculate_similarity_score(template, non_mated)
            non_mated_sim_scores.append(sim_score)
    return non_mated_sim_scores

'''
Method creates dataframe for the FG-Net similarity scores

Parameters:
    sim_scores (list) : the similarity scores for the FG-Net dataset
Return:
    df (DataFrame) : the dataframe containing the similarity scores with five columns
'''
def create_dataframe_fg_net(sim_scores):
    df =  pd.DataFrame(sim_scores, columns=['mated young', 'mated middle', 'mated old', 'YoungvsOld', 'non-mated'])
    return df

'''
Method creates dataframe for similarity scores

Parameters:
    similarity_scores (list) : similarity scores for the test dataset
Return:
    df (DataFrame) : the dataframe containing the similarity scores with three columns

'''
def create_dataframe(similarity_scores):
    df = pd.DataFrame(similarity_scores, columns=['mated', '20vs65', 'non-mated'])
    return df

'''
Method creates a distribution plot

Parameters:
    df (DataFrame) : the dataframe containing the data that is to be plotted
    output_path (str) : the path where the distribution plot should be saved
Return:
    None.
'''
def create_distribution_plot(df : pd.DataFrame, output_path : str, epoch_num : int):
    # need to take in a dataFrame which contains three columns named "mated", "agevsage" and "non-mated".
    # Each row must contain one identity with values pertaining to the three columns, 
    # i.e. the identitiy's similarity score for a mated sample, sim score compared to mated but older, and non-mated.

    path = f"{output_path}/epoch_{epoch_num}"
    os.makedirs(path, exist_ok=True)
    sns.displot(df, kind="kde")
    plt.grid(visible=True)
    plt.xlabel("Similarity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{path}/distribution_plot.pdf")
    plt.close()

def create_arcface_vs_finetuned_plot(sim_scores, test_data_loader, outdir_plot : str, epoch_num : int):
    arcface_sim_scores = get_arcface_sim_scores("models/backbone.pth", test_data_loader)
    dfs = []

    for i in range(len(sim_scores[0])):
        df_list = []
        for row_ft, row_af in zip(sim_scores, arcface_sim_scores):
            new_row = []
            new_row.append(row_ft[i])
            new_row.append(row_af[i])
            df_list.append(new_row)
        df = create_dataframe_finetuned_vs_arcface(df_list)
        dfs.append(df)
        
    create_subplots(dfs, outdir_plot, epoch_num)

def create_dataframe_finetuned_vs_arcface(sim_scores):
    df = pd.DataFrame(sim_scores, columns=['Fine-tuned ArcFace', 'ArcFace'])
    return df

def create_subplots(dfs, outdir_plot, epoch_num):
    path = f"{outdir_plot}/epoch_{epoch_num}"
    os.makedirs(path, exist_ok=True)
    fig,axs = plt.subplots(1, len(dfs))

    labels = ['Mated Young', 'Mated Middle', 'Mated Old', 'Young vs Old', 'Non-mated']
    # For every [input,step...]
    for ax,df,l in zip(axs, dfs, labels):
        sns.displot(df, kind="kde")
        plt.grid(visible=True)
        plt.xlabel("Similarity")
        plt.ylabel("Density")
        plt.title(l)
        plt.tight_layout()
        plt.savefig(f"{path}/arcfaceVsFineTuned_{l}_plot.pdf")
        plt.close()


'''
tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((98, 98))])
outdir_plot = "plots/test/"
data_loader = load_test_dataset("datasets/fgnet/", 128, tsfm)
model = iresnet50()
#compute_sim_scores_fg_net(data_loader, model, outdir_plot,0)

#test2(model, data_loader)
'''




