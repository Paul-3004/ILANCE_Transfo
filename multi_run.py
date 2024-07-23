import os
import glob
from argparse import ArgumentParser
import json
import torch
from main import train_and_validate, inference
DEVICE = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

def create_dir(root_dir_path, model_epoch, dataset):
    dir_name = os.path.join(root_dir_path, "res_model_") 
    if isinstance(model_epoch,str):
        dir_name += model_epoch
    else:
        dir_name += f"{model_epoch}"
    
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    subdir_path = os.path.join(dir_name,dataset)
    if not os.path.isdir(subdir_path):
        os.mkdir(subdir_path)
    return subdir_path

def run_inference(config, args):
    model = args.me
    dir_res = create_dir(config["dir_results"], model,"test")
    config["dir_results"] = dir_res
    inference(config,args,model)
    if args.overfit:
        dir_res = create_dir(config["dir_results"], model, "training")
        config["dir_results"] = dir_res
        config["dir_path_inference"] = config["dir_path_train"]
        inference(config,args,model)



if __name__ == "__main__":
    parser = ArgumentParser(prog= "Finding clusters training and inference ")
    #Inference
    parser.add_argument("-config_path", type = str, help = "Directory of ConfigFile")
    parser.add_argument("-device", type = int, help = "Number of cuda device to use")
    parser.add_argument("-model", type = int, help = "Number of model implementation to use")
    parser.add_argument("--tf", action="store_true", help = "Set true to train before inference")
    parser.add_argument("-me", choices = ["best", "all",range(0,100)], nargs="+", help = "Epoch of the model to infer with")
    parser.add_argument("--overfit", action = "store_true", help = "Set true to run inference on test and training set with model of specified epochs")

    args = parser.parse_args()
    DEVICE = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path + "ConfigFile.json") as f:
        config = json.load(f)

    if args.me == "all":
        files_model = list(glob.iglob(os.path.join(config["dir_model"], "model_*")))
        args.me = ["best"] + range(len(files_model))
    if args.tf:
        train_and_validate(config,args)
    
    model_type = args.me
    if isinstance(model_type, str) or isinstance(model_type, int):
        run_inference(config,args)
    else:
        for model in model_type:
            dir_res = create_dir(config["dir_results"], model,"test")
            config["dir_results"] = dir_res
            inference(config,args,model)
            if args.overfit:
                dir_res = create_dir(config["dir_results"], model, "training")
                config["dir_results"] = dir_res
                inference(config,args,model)


            

    

