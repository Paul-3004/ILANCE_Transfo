import json
import h5py
import awkward as ak
import numpy as np

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
    print(f"{filenames=}")
    assert(len(filenames)>0)
    for i, file in enumerate(filenames):
        print(f"Reading file: {file=}")
        feat, label = load_awkward2(file)
        if i==0:
            ak_feats = feat
            ak_labels = label
        else:
            ak_feats = ak.concatenate((ak_feats, feat), axis=0)
            ak_labels = ak.concatenate((ak_labels, label), axis=0)

        #print (ak.num(ak_feats,axis=0), ak.num(ak_labels,axis=0))                                                                                                                   

    return ak_feats, ak_labels

    