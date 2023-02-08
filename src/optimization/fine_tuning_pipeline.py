from insightface2.recognition.arcface_torch.backbones import iresnet50
import torch;
from torch.nn import Module as m
from data.data_preprocessing import load_dataset

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

'''
Method fine-tunes a pre-trained model with new data
Input: 
Input:
Return 
'''
def fine_tuning_pipeline(filename : str, device : torch.device, frozenParams: list, frozenLayers, model : m, path : str):

    # Fetching pretrained model and unfreezing some layers
    model = load_pretrained_model(filename, device)
    model = unfreeze_model_layers(frozenParams, frozenLayers)

    # Load dataset
    data, labels = load_dataset(path)

    # Preprocess data
    

    # Train the unfrozen layers

    return
