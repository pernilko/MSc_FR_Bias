import torch
from data.data_preprocessing import load_test_dataset
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
import eval.evaluation as evaluation
from torchvision import transforms
import argparse

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

"""
Method for loading the bare arcface model and evaluating its performance and bias on a test dataset

Parameters:
    model_filename (string) : the path for the file containing the pre-trained arcface model
    device (torch.device) : specifies whether to use cpu or cuda
    path (string) : the path for the test dataset which the model is to be evaluated on
    plot_output_filename (string) : the path for where the generated distribution plot is to be saved
Return: ----

"""
def arc_face_pipeline(model_filename : str, device : torch.device, path : str, plot_output_filename : str):

    # Fetching arcface model
    model = load_arc_face_model(model_filename, device)
    model.train(False)

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize((112,112))
    ])
    batch_size = 1002

    # Load test dataset and create distribution plot
    test_data_loader = load_test_dataset(path, batch_size, tsfm)
    sim_scores = evaluation.compute_sim_scores_fg_net(test_data_loader, model, output_plot_path, 0)
    garbe = evaluation.evaluate_fairness(model, test_data_loader)
    print("eval..")
    print("GARBE: ", garbe)


'''
Running ArcFace pipeline
'''

# Defining default parameters
model_path = "models/backbone.pth"
device = torch.device('cuda', 0)
test_dataset_path = 'datasets/fgnet/' #"datasets/cusp_generated/"
output_plot_path = "plots/arcface/"

# Defining input argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dist_plot_path', type=str, default=output_plot_path, help="output path for distribution plots")
parser.add_argument('--model_input_path', type=str, default=model_path, help="path to pre-trained model")
parser.add_argument('--test_data_path', type=str, default=test_dataset_path, help="path to test images")

args = parser.parse_args()
print("Model path: ", args.model_input_path)
print("Test data path: ", args.test_data_path)
print("Dist plot path: ", args.dist_plot_path)

# Running ArcFace pipeline
arc_face_pipeline(args.model_input_path,  device, args.test_data_path, args.dist_plot_path)
