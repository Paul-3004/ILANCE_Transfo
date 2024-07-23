import os 
import json
from argparse import ArgumentParser

def create_dir(root,name,param):
    dir_name = name + f"_{param}"
    dir_path = os.path.join(root,dir_name)
    
    if not os.path.isdir(dir_path):
        print(f"Creating directory {dir_path}")
        os.mkdir(dir_path)
    return dir_path

def modify_template(template, param_name,param_value, dir_path,ds_path):
    
    if param_name != "dmodel":
        template[param_name] = param_value
    else:
        template[param_name] = param_value
        template["nhid_ff_trsf"] = 2 *param_value
        template["d_out_embedder_src"] = param_value
        template["d_ff_embedder_src"] = 2 *param_value
        template["d_out_embedder_tgt"] = param_value
        template["d_ff_embedder_tgt"] = 2 *param_value
        template["dim_ff_sub_decoder"] = 2 *param_value
        template["dim_ff_main_decoder"] = 2*param_value
        template["d_ff_embedder_tracks"] = 2 *param_value
        template["d_out_embedder_tracks"] = param_value
    template["dir_results"] = dir_path
    template["dir_model"] = dir_path
    template["path_charges"] = dir_path
    template["path_PDGs"] = dir_path
    template["dir_path_inference"] = os.path.join(ds_path, "test")
    template["dir_path_train"] = os.path.join(ds_path, "training")
    template["dir_path_val"] = os.path.join(ds_path, "validation")

def convert_str(values, param_name):
    to_int = ["dmodel", "batch_size"]
    to_bool = ["do_tracks", "norm_first"]
    to_float = ["lr"]
    if to_int.count(param_name) > 0:
        values = [int(x) for x in values]
    elif to_bool.count(param_name) > 0:
        values = [bool(int(x)) for x in values] 
    elif to_float.count(param_name) >0:
        values = [float(x) for x in values]
    else:
        raise RuntimeError(f"{param_name} is an invalid parameter")
    return values

if __name__ == "__main__":
    parser = ArgumentParser(prog= "create the results directories as well as ConfigFile based on a template")
    parser.add_argument("-t", type=str,help="path of template ConfigFile")
    #parser.add_argument("-d", type = str, nargs="+", help= "names of dirs to create.")
    parser.add_argument("-p", type = str, help="name of parameter to vary")
    parser.add_argument("-v",type = str, nargs="+", help = "values of varying parameter")
    parser.add_argument("-d", type= str, help = "directory in which new directories will be created")
    parser.add_argument("-ds", type= str, help = "directory in which new directories will be created")
    args = parser.parse_args()

    param_vals = args.v
    param_name = args.p
    if not isinstance(param_vals,list):
        param_vals = list(param_vals)
    param_vals = convert_str(param_vals, param_name)
    dir_root = args.d
    dataset = args.ds
    ds_path = ""
    ds1 = "g1"
    ds2 = "g2"
    ds3 = "tau"
    if dataset == ds1:
        ds_path = "/data/suehara/mldata/pfa/gamma_10to100GeV_1"
    elif dataset == ds2:
        ds_path = "/data/suehara/mldata/pfa/gamma_5to50GeV_2"
    elif dataset == ds3:
        ds_path = "/data/suehara/mldata/pfa/ntau_10to100GeV_10"
    else:
        raise RuntimeError(f"Incorrect entry for dataset: {dataset}. Expected {ds1}, {ds2} or {ds3}")
    
    if not isinstance(param_vals,list):
        param_vals = list(param_vals)
    if not os.path.isdir(dir_root):
        print(f"The root directory {dir_root} does not exist. Will be created")
        os.mkdir(dir_root)

    for val in param_vals:
        dir_path = create_dir(dir_root,param_name,val)
        with open(args.t, 'r') as f:
            config = json.load(f)
        keys = list(config.keys())
        if keys.count(param_name) < 1:
            raise RuntimeError(f"Parameter {param_name} is invalid")
        modify_template(config,param_name,val, dir_path, ds_path)
        with open(os.path.join(dir_path,"ConfigFile.json"), 'w') as fp:
            json.dump(config,fp, indent=2)