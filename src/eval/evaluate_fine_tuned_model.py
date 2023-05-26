from eval.evaluation import evaluate_fairness, compute_sim_scores_fg_net, create_arcface_vs_finetuned
import torch
from models.insightface2.recognition.arcface_torch.backbones import iresnet50
from data.data_preprocessing import load_test_dataset, count_number_of_images
from torchvision import transforms
import argparse

def load_model(filename : str, device : torch.device):
    loaded_model = torch.load(filename,  map_location = device)
    model = iresnet50()
    model.load_state_dict(loaded_model, strict=True)

    return model

def evaluate_fine_tuned_model(model_path, test_dataset_path, experiment_name : str):
    model = load_model(model_path, torch.device('cuda:0'))
    tfsm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize((112,112))
    ])
    
    total_num_of_images = count_number_of_images(test_dataset_path)
    test_data_loader = load_test_dataset(test_dataset_path, total_num_of_images, tfsm_test)

    print("starting evaluation")
    model.train(False)
    plot_path = f"experiments/{experiment_name}/plots/"
    sim_scores = compute_sim_scores_fg_net(test_data_loader, model, plot_path, 0, True)
    print("entering new method")
    create_arcface_vs_finetuned(sim_scores, test_data_loader, None, 0)
    
    garbe = evaluate_fairness(model, test_data_loader, experiment_name, 0)


model_path = "models/backbone.pth"
test_dataset_path = "datasets/fgnet/"
experiment_name = "test_experiment"


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=model_path, help="path to model")
parser.add_argument('--test_dataset_path', type=str, default=test_dataset_path, help="path to test dataset")
parser.add_argument('--organize_test_dataset', type=bool, default=False, help="True if you wish to organize test dataset, else False.")
parser.add_argument('--experiment_name', type=str, default=experiment_name, help="Name of current experiment")


args = parser.parse_args()
print("Model path: " + args.model_path)
print("Test dataset path: " + args.test_dataset_path)
print("Organize test dataset: " + str(args.organize_test_dataset))
print("Experiment name: " + args.experiment_name)

evaluate_fine_tuned_model(args.model_path, args.test_dataset_path, args.experiment_name)

