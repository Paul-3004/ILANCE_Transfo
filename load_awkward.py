import json
import h5py
import awkward as ak
import numpy as np
<<<<<<< HEAD
import time

=======
from time import time
>>>>>>> d71f3dd2578d215fa8f0ec20ca675c99f375226f
def load_awkward2(filename):
    file = h5py.File(filename) #Create File object                                                                                                                                   
    feat = file["feature"] #access the feature group, File contains two subgroups, feature and label                                                                                 

    form = ak.forms.from_json(feat.attrs["form"]) #/feature has two attributes, form and length, JSON data returned as str                                                           
    length = json.loads(feat.attrs["length"]) #only one number                                                                                                                       
    container = {k: np.asarray(v) for k, v in feat.items()}

    ak_feat = ak.from_buffers(form, length, container)

    label = file["label"]

    form = ak.forms.from_json(label.attrs["form"])
    length = json.loads(label.attrs["length"])
    container = {k: np.asarray(v) for k, v in label.items()}

    ak_label = ak.from_buffers(form, length, container)

    if "pred" in file:
        pred = file["pred"]
        form = ak.forms.from_json(pred.attrs["form"])
        length = json.loads(pred.attrs["length"])
        container = {k: np.asarray(v) for k, v in pred.items()}
        ak_pred = ak.from_buffers(form, length, container)

        if "energy" in file:
            energy = file["energy"]
            form = ak.forms.from_json(energy.attrs["form"])
            length = json.loads(energy.attrs["length"])
            container = {k: np.asarray(v) for k, v in energy.items()}
            ak_energy = ak.from_buffers(form, length, container)
            return ak_feat, ak_label, ak_pred, ak_energy

        return ak_feat, ak_label, ak_pred

    # no pred: just return feat and label                                                                                                                                            
    return ak_feat, ak_label

def load_awkwards(filenames):
    #print(f"{filenames=}")
    assert(len(filenames)>0)
    feats_list, labels_list = [], []
    for i, file in enumerate(filenames):
        print(f"Reading file: {file=}")
        start = time()
        feat, label = load_awkward2(file)
<<<<<<< HEAD
        print(time()- start)
        if i==0:
            ak_feats = feat
            ak_labels = label
        else:
            ak_feats = ak.concatenate((ak_feats, feat), axis=0)
            ak_labels = ak.concatenate((ak_labels, label), axis=0)

=======
        feats_list.append(feat)
        labels_list.append(label)
    #print(len(feats_list))
    #print(ak.num(feats_list[0], axis = 0))
    #print(ak.num(feats_list[0],axis = 1))
    #start = time()    
    #feats_ak = ak.flatten(ak.Array(feats_list),axis = 1)
    #print(time() - start)
    #labels_ak = ak.flatten(ak.Array(labels_list), axis = 1)
    #return feats_ak, labels_ak
        #if i==0:
        #    ak_feats = feat
        #    ak_labels = label
        #else:
    start = time()
    ak_feats = ak.concatenate(feats_list, axis=0)
    ak_labels = ak.concatenate(labels_list, axis=0)
    print(time()-start)
>>>>>>> d71f3dd2578d215fa8f0ec20ca675c99f375226f
        #print (ak.num(ak_feats,axis=0), ak.num(ak_labels,axis=0))                                                                                                                   

    return ak_feats, ak_labels

    
