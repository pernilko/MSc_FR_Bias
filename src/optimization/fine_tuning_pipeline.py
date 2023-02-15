import sys
folder_root = '/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src'
sys.path.append(folder_root)
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
import utils.evaluation as evaluation
import torch;
from torch.nn import Module as m
from data.data_preprocessing import load_dataset, load_test_dataset
from torchvision import transforms, datasets
import os
import numpy as np

'''
Method loads pretrained model
Input: filename -> name of file (.pt or .pth) containing the model. 
Input: device -> use cpu or cuda
Return: IResNet model
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

def train_model(number_of_epochs : int, model, learning_rate, momentum, training_data_loader, validation_data_loader, batch_size):
    epoch_number = 0
    best_vloss = 1_000_000.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = torch.nn.CrossEntropyLoss()

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
            print(len(vinputs))
            voutputs = model(vinputs)
            print(len(voutputs))
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            if epoch + 1 == 10:    
                '''Sim scores'''
                sim_scores = []
                for i in range(0,len(vinputs)):
                    sim_score_identity = []
                    sim_score_mated = []
                    sim_score_non_mated =[]
                    sim_score_age_mated = []
                    for j in range(i+1, len(vinputs)):
                        if vlabels[i] == vlabels[j] and j < 3: #mated
                            print("mated match")
                            output1 = model(vinputs[i])
                            output2 = model(vinputs[j])
                            distance = evaluation.calculate_similarity_score(output1, output2)
                            sim_score_mated.append(distance)

                        if vlabels[i] != vlabels[j]: # non-mated
                            print("non-mated match")
                            output1 = model(vinputs[i])
                            output2 = model(vinputs[j])
                            distance = evaluation.calculate_similarity_score(output1, output2)
                            sim_score_non_mated.append(distance)
                        if vlabels[i] == vlabels[j] and j > 2: #age-mated
                            print("age-mated match") 
                            output1 = model(vinputs[i])
                            output2 = model(vinputs[j])
                            distance = evaluation.calculate_similarity_score(output1, output2)
                            sim_score_age_mated.append(distance)
                    sim_score_identity.append(np.mean(sim_score_age_mated))
                    sim_score_identity.append(np.mean(sim_score_age_mated))
                    sim_score_identity.append(np.mean(sim_score_non_mated))
                    sim_scores.append(sim_score_identity)

                print("sim scores: ", sim_scores, "end sim scores")
                evaluation.create_dataframe(sim_scores)

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}'.format(epoch_number)
            torch.save(model.state_dict(), model_path)

        '''
        if epoch == 10:
            evaluation.create_dataframe(test_data_loader)
            evaluation.create_distribution_plot()
        '''
        epoch_number += 1
    return model


'''
Method fine-tunes a pre-trained model with new data
Input: 
Input:
Return 
'''
def fine_tuning_pipeline(filename : str, device : torch.device, frozenParams: list, frozenLayers, model : m, path : str, name_of_fine_tuned_model):

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


    # Train the unfrozen layers
    fine_tuned_model = train_model(10, model, 0.001, 0.09, training_data_loader, validation_data_loader, batch_size)
    print("fine-tuned model: ", fine_tuned_model)
    os.makedirs("models/fine_tuned_models/", exist_ok=True)
    torch.save(fine_tuned_model, "models/fine_tuned_models/" + name_of_fine_tuned_model)

    test_data_loader = load_test_dataset("datasets/lfw/", 12, tsfm)
    #evaluation.evaluate_performance(fine_tuned_model, test_data_loader)

    


frozenParams = ['conv1.weight', 'bn1.weight', 'bn1.bias',  'prelu.weight']
frozenLayers = ['layer1', 'layer2']
module : torch.nn.Module = iresnet50()
input_images_path = "datasets/cusp_generated/"
name_of_fine_tuned_model = "fine_tuned_model_1.pt"

fine_tuning_pipeline("models/backbone.pth", torch.device('cuda', 0), frozenParams, frozenLayers, module, input_images_path, name_of_fine_tuned_model)