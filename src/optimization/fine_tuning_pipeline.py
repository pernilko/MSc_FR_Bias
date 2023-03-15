import sys
folder_root = '/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src'
sys.path.append(folder_root)
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
import eval.evaluation as evaluation
import torch;
from torch.nn import Module as m
from data.data_preprocessing import load_dataset, load_test_dataset, orgranize_fgnet_dataset
from torchvision import transforms, datasets
import os
import numpy as np
from models.insightface2.recognition.arcface_torch.losses import CombinedMarginLoss
from models.insightface2.recognition.arcface_torch.partial_fc_v2 import PartialFC_V2
from pytorch_metric_learning import losses
import argparse


'''
Method for loading the pretrained ArcFace model

Parameters:
    filename (string) : path for the file containing the pre-trained ArcFace model. Should be .pt or .pth file 
    device (torch.device) : specifies whether to use cpu or cuda
Return: 
    model (Iresnet)
'''
def load_pretrained_model(filename : str, device : torch.device):
    loaded_model = torch.load(filename,  map_location = device) #torch.device('cpu')
    model = iresnet50()
    model.load_state_dict(loaded_model, strict=True)

    return model

'''
Method freezes specified layers, the remaining layers are unfrozen.
Input: frozenParams -> list of strings containing the parameter keys of the parameters to freeze
Input: frozenLayers -> list of strings containing the partial parameter keys of layers to be frozen
Return: model
'''
def unfreeze_model_layers(frozenParams: list, frozenLayers, model : m):

    # Freeze all layer
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers
    for name, param in model.named_parameters():
        does_not_require_grad = not param.requires_grad
        param_not_in_name = all((not x == name) for x in frozenParams )
        layer_not_in_name =  all((not x in name) for x in frozenLayers)

        if (does_not_require_grad and ( param_not_in_name and layer_not_in_name)):
            param.requires_grad = True 

    return model


def train_one_epoch(training_data_loader, optimizer, model, loss_function, batch_size):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_data_loader):
        inputs, labels = data

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == (batch_size-1):
            last_loss = running_loss / batch_size # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

def train_model(number_of_epochs : int, model, learning_rate, momentum, training_data_loader, validation_data_loader, batch_size, test_data_loader, dist_plot_path : str):
    epoch_number = 0
    best_vloss = 1_000_000.
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    #margin_loss = CombinedMarginLoss(64, 1.0, 0.5, 0.0)

    uniq_labels = []
    for i, data in enumerate(training_data_loader):
        inps, labels = data
        labels_batch = torch.unique(labels).tolist()
        for lab in labels_batch:
            if lab not in uniq_labels:
                uniq_labels.append(lab)

    print("labels: ", uniq_labels)
    num_of_classes = len(uniq_labels)
    print("num of classes: ", num_of_classes)

    loss_fn = losses.ArcFaceLoss(num_of_classes, 512, margin=28.6, scale=64)
    optimizer = torch.optim.SGD(loss_fn.parameters(), lr=learning_rate, momentum=momentum)
    
    #module_partial_fc = PartialFC_V2(margin_loss, 512, num_of_classes,1, False)
    #module_partial_fc.train().cuda()


    for epoch in range(number_of_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_data_loader, optimizer, model, loss_fn, batch_size)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_data_loader):
            vinputs, vlabels = vdata
            #print("inp len", len(vinputs))
            voutputs = model(vinputs)
            #print("out len", len(voutputs))
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}'.format(epoch_number)
            torch.save(model.state_dict(), model_path)

       
        if epoch + 1 == 10:    
            #sim_scores = evaluation.compute_similarity_scores_for_test_dataset(test_data_loader, model)
            #df = evaluation.create_dataframe(sim_scores)
            #evaluation.create_distribution_plot(df, dist_plot_path)
            sim_scores = evaluation.compute_sim_scores_fg_net(test_data_loader, model, dist_plot_path)
        
        epoch_number += 1
    return model


'''
Method fine-tunes a pre-trained model with new data
Input: 
Input:
Return 
'''
def fine_tuning_pipeline(filename : str, device : torch.device, frozenParams: list, frozenLayers, model : m, path : str, name_of_fine_tuned_model : str, test_images_path : str, dist_plot_path : str, orgranize_fgnet : bool):

    # Fetching pretrained model and unfreezing some layers
    model = load_pretrained_model(filename, device)
    model = unfreeze_model_layers(frozenParams, frozenLayers, model)


    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(98)
    ])
    batch_size = 20
    # Load training and validation dataset
    training_data_loader, validation_data_loader = load_dataset(path, batch_size, tsfm)
    if orgranize_fgnet:
        orgranize_fgnet_dataset(test_images_path)

    tfsm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((98,98))
    ])
    
    test_data_loader = load_test_dataset(test_images_path, 1002, tfsm_test)

    # Train the unfrozen layers
    fine_tuned_model = train_model(10, model, 0.01, 0.09, training_data_loader, validation_data_loader, batch_size, test_data_loader, dist_plot_path)
    #print("fine-tuned model: ", fine_tuned_model)
    os.makedirs("models/fine_tuned_models/", exist_ok=True)
    torch.save(fine_tuned_model, "models/fine_tuned_models/" + name_of_fine_tuned_model)


    

'''
Main run
'''

frozenParams = ['conv1.weight', 'bn1.weight', 'bn1.bias',  'prelu.weight']
frozenLayers = ['layer1', 'layer2']
module : torch.nn.Module = iresnet50()
input_images_path = "datasets/cusp_generated_v2/"
name_of_fine_tuned_model = "fine_tuned_model_1.pt"
test_images_path = "datasets/fgnet/"
device = torch.device('cuda:0')
plot_out_path = "plots/cusp/"

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', type=str, default=input_images_path, help="path to input images")
parser.add_argument('--dist_plot_path', type=str, default=plot_out_path, help="output path for distribution plots")
parser.add_argument('--organize_fgnet', type=bool, default='0', help="output path for distribution plots" )
parser.add_argument('--test_data_path', type=str, default='datasets/fgnet/', help="path to test images")

args = parser.parse_args()
print("Input img path: " + args.input_img_path)
print("Dist plot path: ", args.dist_plot_path)
print("Orgranize FG-NET dataset: " + str(args.organize_fgnet))
print("Test imgs path: ", args.test_data_path)

fine_tuning_pipeline("models/backbone.pth", device, frozenParams, frozenLayers,
                      module, args.input_img_path, name_of_fine_tuned_model, args.test_data_path, args.dist_plot_path, args.organize_fgnet)


