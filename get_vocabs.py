import torch
import os
import numpy as np
from data_prepro import CollectionHits


def get_pdg(path,frac_files):
    special_symbols = {
        "pad":0,
        "bos":1,
        "eos":2,
        "sample":3
    }
<<<<<<< Updated upstream
    ds_col = CollectionHits(path,special_symbols,frac_files,False)
    ds_col.process_dataset()
    dict_pdg = ds_col.vocab_pdgs.vocab
    return list(dict_pdg.keys())
=======
    #ds_col = CollectionHits(path,special_symbols,1,False)
    #ds_col.process_dataset()
    #dict_pdg = ds_col.vocab_pdgs.vocab
    #return list(dict_pdg.keys())
    return [5,6,7,8]
>>>>>>> Stashed changes

if __name__=="__main__":
    dir_ds = "/data/suehara/mldata/pfa/"
    datasets = ["ntau_10to100GeV_10", "gamma_10to100GeV_1", "gamma_5to50GeV_2", "eg_5to50GeV_1", "el_10to100GeV_1", "el_5to50GeV_2"]

    pdg_all = []
<<<<<<< Updated upstream
    _, vocab_ntau_train, _ = torch.load("/data/suehara/mldata/pfa/ntau_10to100GeV_10/preprocessed/training/vocabs/vocabs_normalizer.pt")
    pdg_ntau_train = list(vocab_ntau_train.keys())
    print(f"pdg ntau train: {pdg_ntau_train}")
=======
    #_, vocab_ntau_train, _ = torch.load("/data/suehara/mldata/pfa/ntau_10to100GeV_10/preprocessed/training/vocabs/vocabs_normalizer.pt")
    #pdg_ntau_train = list(vocab_ntau_train.keys())
    #print(f"pdg ntau train: {pdg_ntau_train}")
>>>>>>> Stashed changes
    pdg_all = []
    #pdg_all.append(pdg_ntau_train)
    for ds in datasets:
        subdirs = ["train", "validation","test"]
        ds_with_sub = ["ntau_10to100GeV_10", "gamma_10to100GeV_1", "gamma_5to50GeV_2"]
        dir_voc = os.path.join(dir_ds,ds)
        frac_files = 1
        if ds in ds_with_sub:
            for subdir in subdirs:
                if not ((ds == "ntau_10to100GeV_10") and (subdir == "train")):
                    if ds == "ntau_10to100GeV_10":
                        frac_files = 0.1
                    dir_voc = os.path.join(dir_ds,ds,subdir)
                    print(f"getting pdg of {dir_voc}")
                    pdg = get_pdg(dir_voc, frac_files)
                    print(pdg)
                    pdg_all += pdg

        else:
            print(f"getting pdg of {dir_voc}")
<<<<<<< Updated upstream
            pdg = get_pdg(dir_voc,frac_files)
            pdg_all += pdg
            print(pdg)

=======
            pdg = get_pdg(dir_voc)
            pdg_all.append(pdg)
            print(pdg)
print(pdg_all)
>>>>>>> Stashed changes
pdg_np = np.array(pdg_all)
pdg_unique = np.unique(pdg_np)
print(f"unique pdgs: {pdg_unique}")
torch.save(pdg_unique,"vocab_values_all_ds.pt")
