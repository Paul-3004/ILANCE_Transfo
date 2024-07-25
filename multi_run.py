import os
import glob
from argparse import ArgumentParser
import json
import torch
from main import train_and_validate, inference
import main_tracks
DEVICE = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

def set_fracfiles(config):
    dir_path_train = config["dir_path_train"]
    tail, train = os.path.split(dir_path_train)
    tail, ds = os.path.split(tail)
    if ds == "ntau_10to100GeV_10":
        config["frac_files_test"] = 0.02/16
    else:
        config["frac_files_test"] = 0.1

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

def run_inference(config, args,model):
    dir_res_original = config["dir_results"]
    frac_test_original = config["frac_files_test"]
    path_inference_original = config["dir_path_inference"]
    path_inference2_original = config["dir_path_inference2"]

    dir_res = create_dir(config["dir_results"], model,"test")
    config["dir_results"] = dir_res
    if config["do_tracks"]:
        main_tracks.inference(config,args,model)
    else:
        inference(config,args,model)

    config["dir_results"] = dir_res_original
    print(f"inference on {dir_res} with model epoch {model} done")
    if args.overfit:
        dir_res = create_dir(config["dir_results"], model, "train")
        config["dir_results"] = dir_res
        config["dir_path_inference"] = config["dir_path_train"]
        config["dir_path_inference2"] = config["dir_path_train2"]
        set_fracfiles(config)
        if config["do_tracks"]:
            main_tracks.inference(config,args,model)
        else:
            inference(config,args,model)
       
        dir_path_inf = config["dir_path_inference"]
        dir_path_inf2 = config["dir_path_inference2"]
        if config["mix_datasets"]:
            print(f"inference on {dir_path_inf} and {dir_path_inf2} with model epoch {model} saved in {dir_res}")
        else:
            print(f"inference on {dir_path_inf} with model epoch {model} saved in {dir_res}")
    
    config["dir_results"] = dir_res_original
    config["dir_path_inference"] = path_inference_original
    config["dir_path_inference2"] = path_inference2_original
    config["frac_files_test"] = frac_test_original


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

    with open(os.path.join(args.config_path,"ConfigFile.json")) as f:
        config = json.load(f)


    if args.tf:
        if config["do_tracks"]:
            main_tracks.train_and_validate(config,args)
        else:
            train_and_validate(config,args)
        
        
    if args.me.count("all") > 0:
        files_model = list(glob.iglob(os.path.join(config["dir_model"], "model_*")))
        args.me = ["best"] + list(range(len(files_model)))
    
    model_type = args.me
    if isinstance(model_type, str) or isinstance(model_type, int):
        run_inference(config,args, model_type)
    else:
        for model in model_type:
            run_inference(config,args, model)
            

            

    

