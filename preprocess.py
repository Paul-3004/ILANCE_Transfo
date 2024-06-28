from data_prepro import train_val_preprocessing, get_data
from time import time
import numpy as np

dir_res  = "/data/suehara/mldata/pfa/ntau_10to100GeV_10/preprocessed"
dir_train = "/data/suehara/mldata/pfa/ntau_10to100GeV_10/train"
dir_val =  "/data/suehara/mldata/pfa/ntau_10to100GeV_10/validation"
frac_files = 0.6

testing = False
if testing:
    start = time()
    vocab_charges, vocab_pdgs, special_symbols, E_rms_normalizer, train_dl, val_dl = get_data((dir_res + "/training", dir_res + "/validation"), 10, 1, "training", True)
    dt = time() - start
    print(f"time to load from preprocessed {dt} sec")
    for i,(feats, labels) in enumerate(train_dl):
       # print(labels)
       # print(feats)
        pass
else:
   # fracs = np.arange(6,11)/ 10
   # for frac_files in fracs:
   #     train_val_preprocessing(dir_train, dir_val, dir_res, frac_files)
   train_val_preprocessing(dir_train, dir_val, dir_res, frac_files, 0.1)     
