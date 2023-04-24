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
from torch.utils.data import DataLoader


'''
Method for loading the pretrained ArcFace model

Parameters:
    filename (string) : path for the file containing the pre-trained ArcFace model. Should be .pt or .pth file 
    device (torch.device) : specifies whether to use cpu or cuda
Return: 
    model (Iresnet)
'''
def load_pretrained_model(filename : str, device : torch.device):
    loaded_model = torch.load(filename,  map_location = device)
    model = iresnet50()
    model.load_state_dict(loaded_model, strict=True)

    return model

'''
Method freezes specified layers, the remaining layers are unfrozen.
Parameters:
    frozenParams (list) : list of strings containing the parameter keys of the parameters to freeze
    frozenLayers (list) : list of strings containing the partial parameter keys of layers to be frozen
    model (Module) : the module for the model
Return:
    model () : the model with the appropriate 
'''
def unfreeze_model_layers(frozenParams: list, frozenLayers : list, model : iresnet50):

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

'''
Method for training one epoch of the model
Parameters:
    training_data_loader (DataLoader) : data loader containing the training data
    optimizer () : the optimizer to be used during training
    model () : the model that is being trained
    loss_function () : the loss function
    batch_size (int) : the size of the batch
Return:
    last_loss (float) : the last loss of the epoch
'''
def train_one_epoch(training_data_loader : DataLoader, optimizer, model, loss_function, batch_size : int):
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

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == (batch_size-1):
            last_loss = running_loss / batch_size # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

'''
Method for training the model
Parameters:
    number_of_epochs (int) : number of epochs to train the model
    model () : the model to be trained
    learning_rate (float) : the rate at which the model learns
    momentum (float) : 
    training_data_loader (DataLoader) : data loader containing the training data
    validation_data_loader (DataLoader) : data loader containing the validation data
    batch_size (int) : the size of the batches
    test_data_loader (DataLoader) :  data loader containing the test data
    dist_plot_path (str) : the output path of the distribution plot
    opt (str) : string indicating which optimizer to use
Return:
    model () : the trained model
'''
def train_model(number_of_epochs : int, model, learning_rate : float, momentum : float, training_data_loader : DataLoader, validation_data_loader : DataLoader, batch_size : int, test_data_loader : DataLoader, dist_plot_path : str, opt : str, experiment_name : str, test_data_loader_loss):
    epoch_number = 0
    best_vloss = 1_000_000.
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #loss_fn = torch.nn.CrossEntropyLoss()
   

    uniq_labels = []
    for i, data in enumerate(training_data_loader):
        inps, labels = data
        labels_batch = torch.unique(labels).tolist()
        for lab in labels_batch:
            if lab not in uniq_labels:
                uniq_labels.append(lab)

    num_of_classes = len(uniq_labels)

    loss_fn = losses.ArcFaceLoss(num_of_classes, 512, margin=28.6, scale=64)
    weight_decay = 5e-4

    if opt == 'sgd':
        #optimizer = torch.optim.SGD(loss_fn.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": loss_fn.parameters()}],
        lr=learning_rate,
        momentum=momentum, weight_decay=weight_decay)
    elif opt == 'adamW':
        optimizer = torch.optim.AdamW(
            [{"params": model.parameters()}, {"params": loss_fn.parameters()}],
            lr=learning_rate,
            weight_decay=weight_decay)
    else:
        raise Exception(f"Selected optimizer ({opt}) is not supported")

    for epoch in range(number_of_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        with torch.enable_grad():
            avg_loss = train_one_epoch(training_data_loader, optimizer, model, loss_fn, batch_size)

        # We don't need gradients on to do reporting
        model.train(False)
        with torch.no_grad():

            # validation on val dataset
            running_vloss = 0.0
            for i, vdata in enumerate(validation_data_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)

            # log validation loss
            os.makedirs(f"experiments/{experiment_name}/", exist_ok=True)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            #validation on test dataset
            running_tloss = 0.0
            for i, tdata in enumerate(test_data_loader_loss):
                tinputs, tlabels = tdata
                toutputs = model(tinputs)
                tloss = loss_fn(toutputs, tlabels)
                running_tloss += tloss

            avg_tloss = running_tloss / (i + 1)
            print('LOSS train {}, valid {}, test {}'.format(avg_loss, avg_vloss, avg_tloss))
            f = open(f"experiments/{experiment_name}/validation_loss_results.txt", "a")
            f.write('LOSS train {}, valid {}, test {}\n'.format(avg_loss, avg_vloss, avg_tloss))
            f.close()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss

                dir = f"experiments/{experiment_name}/models/finetuned_state_dict/"
                os.makedirs(dir, exist_ok=True)
                for file in os.scandir(dir):
                    os.remove(file.path) # remove previously saved models to save memory
                
                model_path = f"{dir}model_{epoch_number}"
                torch.save(model.state_dict(), model_path)

        if (epoch + 1) == 1:
            epoch_num_current = epoch + 1
            sim_scores = evaluation.compute_sim_scores_fg_net(test_data_loader, model, dist_plot_path, epoch_num_current)
            garbe = evaluation.evaluate_fairness(model, test_data_loader, experiment_name, epoch_num_current)

        if (epoch + 1) % 10 == 0:    
            #sim_scores = evaluation.compute_similarity_scores_for_test_dataset(test_data_loader, model)
            #df = evaluation.create_dataframe(sim_scores)
            #evaluation.create_distribution_plot(df, dist_plot_path)
            print("Starting evaluation")
            current_epoch_num = epoch + 1
            sim_scores = evaluation.compute_sim_scores_fg_net(test_data_loader, model, dist_plot_path, current_epoch_num)
            garbe = evaluation.evaluate_fairness(model, test_data_loader, experiment_name, current_epoch_num)
        
        epoch_number += 1
    return model


'''
Method fine-tunes a pre-trained model with new data
Parameters:
    filename (str) : the path of the pre-trained model
    device (torch.device): the device to be used. CPU or CUDA.
    frozenParams (list) : the params to be frozen in the pre-trained model
    frozenLayers (list) : the layers to be frozen in the pre-trained model
    model (Module) : the model module for the pretrained model
    path (str) : the path for the training/validation data
    name_of_fine_tuned_model (str) : the name of the file to save the fine-tuned model to
    test_images_path (str) : the path for the test data
    dist_plot_path (str) : the path for where to save the distribution plot
    orgranize_fgnet (bool) : whether to orgranize the FG-NET dataset into identity folders (only required when initially downloading the dataset)
    lr (float) : learning rate
    momentum (float) : momentum for sgd optimizer
    epochs (int) : number of epochs
    batch_size (int) : batch size
    opt (str) : string indicating optimizer to use. Either sgd or adamW
Return:
    None.
'''
def fine_tuning_pipeline(filename : str, device : torch.device, frozenParams: list, frozenLayers, path : str, name_of_fine_tuned_model : str, test_images_path : str, dist_plot_path : str, orgranize_fgnet : bool, lr : float, momentum : float, epochs : int, batch_size : int, opt : str, experiment_name : str):

    # write training settings to file
    os.makedirs(f"experiments/{experiment_name}/", exist_ok=True)
    f = open(f"experiments/{experiment_name}/training_settings.txt", "w")
    f.write(f"Frozen Layers: {frozenLayers}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Number of epochs: {epochs}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Optimizer: {opt}\n")
    if opt == 'sgd':
        f.write(f"Momentum: {momentum}\n")
    f.write(f"Training dataset: {path}\n")
    f.write(f"Test dataset: {test_images_path}\n")
    f.close()

    # Fetching pretrained model and unfreezing some layers
    model = load_pretrained_model(filename, device)
    model = unfreeze_model_layers(frozenParams, frozenLayers, model)

    ''' Number of frozen and unfrozen params
    frozen = 0
    unfrozen = 0
    for param in model.parameters():
        if param.requires_grad:
            unfrozen += 1
        else:
            frozen += 1
    print("Frozen: ", frozen)
    print("Unfrozen: ", unfrozen)
    return
    '''

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize((112,112))
    ])

    # Load training and validation dataset
    training_data_loader, validation_data_loader = load_dataset(path, batch_size, tsfm)
    if orgranize_fgnet:
        orgranize_fgnet_dataset(test_images_path)

    tfsm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize((112,112))
    ])
    
    test_data_loader_loss = load_test_dataset(test_images_path, batch_size, tfsm_test)
    test_data_loader = load_test_dataset(test_images_path, 1002, tfsm_test)

    # Train the unfrozen layers
    fine_tuned_model = train_model(epochs, model, lr, momentum, training_data_loader,
                                    validation_data_loader, batch_size, test_data_loader, dist_plot_path, opt, experiment_name, test_data_loader_loss)
    
    dir_output_fined_tuned_models = f"experiments/{experiment_name}/fine_tuned_model/"
    os.makedirs(dir_output_fined_tuned_models, exist_ok=True)
    torch.save(fine_tuned_model, dir_output_fined_tuned_models + name_of_fine_tuned_model)


def GetListOfFrozenLayers(layers : str):
    frozenLayers = []
    layers = layers.split(",")
    for layer in layers:
        l = layer.strip()
        if int(l) == 1:
            frozenLayers.append('layer1')
        elif int(l) == 2:
            frozenLayers.append('layer2')
        elif int(l) == 3:
            frozenLayers.append('layer3')
        else:
            raise Exception(f"Cannot freeze layer with value {l}")
    return frozenLayers

'''
Main run of fine-tuning pipeline
'''

# Defining default input params
frozenParams = ['conv1.weight', 'bn1.weight', 'bn1.bias',  'prelu.weight']
frozenLayers = ['layer1', 'layer2', 'layer3']
module : torch.nn.Module = iresnet50()
input_images_path = "datasets/cusp_generated_v2/"
name_of_fine_tuned_model = "fine_tuned_model_1.pt"
test_images_path = "datasets/fgnet/"
device = torch.device('cuda:0')
plot_out_path = "plots/cusp/"
learning_rate = 0.001
momentum = 0.9
epochs = 10
model_input_path = "models/backbone.pth"
batch_size = 15 
opt = 'sgd'
experiment_name = "test_experiment"

# Defining input argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', type=str, default=input_images_path, help="path to input images")
parser.add_argument('--dist_plot_path', type=str, default=plot_out_path, help="output path for distribution plots")
parser.add_argument('--organize_fgnet', type=bool, default=False, help="output path for distribution plots" )
parser.add_argument('--test_data_path', type=str, default=test_images_path, help="path to test images")
parser.add_argument('--model_input_path', type=str, default=model_input_path, help="path to pre-trained model")
parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate')
parser.add_argument('--momentum', type=float, default=momentum, help='Momentum')
parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train model')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch Size')
parser.add_argument('--opt', type=str, default=opt, help='Optimizer')
parser.add_argument('--frozen_layers', type=str, default="1,2,3", help="The numbers of the layers of the model to freeze as string seperated by commas")
parser.add_argument('--experiment_name', type=str, default=experiment_name, help="Name of current experiment")

args = parser.parse_args()
print("Input img path: " + args.input_img_path)
plot_path = f"experiments/{args.experiment_name}/{args.dist_plot_path}"
print("Dist plot path: ", plot_path)
print("Orgranize FG-NET dataset: " + str(args.organize_fgnet))
print("Test imgs path: ", args.test_data_path)
print("Model input path: ", args.model_input_path)
print("Learning rate: ", str(args.lr))
print("Momentum: ", str(args.momentum))
print("Number of epochs: ", str(args.epochs))
print("Batch size: ", str(args.batch_size))
print("Optimizer: ", args.opt)
layers = GetListOfFrozenLayers(args.frozen_layers)
print("Layers: ", layers)
print("Experiment Name: ", args.experiment_name)

# Running finetuning pipeline
fine_tuning_pipeline(args.model_input_path, device, frozenParams, layers, args.input_img_path, name_of_fine_tuned_model, args.test_data_path, plot_path, args.organize_fgnet, args.lr, args.momentum, args.epochs, args.batch_size, args.opt, args.experiment_name)


