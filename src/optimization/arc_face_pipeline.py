import torch
from data.data_preprocessing import load_test_dataset
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
import utils.evaluation as evaluation
from torchvision import transforms

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
        transforms.Resize(98)
    ])
    batch_size = 20

    # Load test dataset and create distribution plot
    test_data_loader = load_test_dataset(path, batch_size, tsfm)
    sim_scores = evaluation.compute_similarity_scores_for_test_dataset(test_data_loader, model)
    df = evaluation.create_dataframe(sim_scores)
    evaluation.create_distribution_plot(df, plot_output_filename)


'''
Running ArcFace pipeline
'''
model_path = "models/backbone.pth"
device = torch.device('cuda', 0)
test_dataset_path = "datasets/cusp_generated/"
output_plot_path = "arc_face_distribution_plot.png"
arc_face_pipeline(model_path,  device, test_dataset_path, output_plot_path)
