import torch
from torch.utils.data import Dataset, DataLoader
import load_awkward as la
import awkward as ak
import numpy as np
import glob
from math import ceil
from time import time
'''creates masks to avoid padding tokens influencing the attention.
    Position with True will be ignored.
    args:
        src: source, i.e., input to encoder, shape: (N,S,E)
        tgt: target, i.e., input to decoder, shape: (T,S,E)
        pad_symbol: unique identifier for padding tokens
    output:
        src_padding_mask: mask for the source padding tokens, shape: (N,S)
        tgt_padding_mask: mask for the target padding tokens, shape: (N,T)'''
def create_mask(src, tgt, pad_symbol, device):
    pad_symbol_broad = torch.tensor(pad_symbol).unsqueeze(0).unsqueeze(0).to(device)
    src_padding_mask = src[...,-1] == pad_symbol_broad
    tgt_padding_mask = tgt[...,-1] == pad_symbol_broad

    return src_padding_mask, tgt_padding_mask

'''Callable object to add special tokens to dataset.
        1. Adds (0,0) feature to each sample
        2. Adds a bos token encoded as [0]*nfeats + [1,1] at start of event
        3. Adds a eos token encoded as [0]*nfeats + [1,0] at end of event
        4. Adds padding tokens encoded as [0]*nfeats + [0,1] where necessary'''
class AddSpecialSymbols(object):
    def __init__(self, special_symbols):
        self.special_symbols = special_symbols
    
    def __call__(self, data, data_type: str):
        nsamples = torch.from_numpy(ak.to_numpy(ak.num(data, axis = 1))).unsqueeze_(-1)
        #First adding (0,0) indicating hit/MC data at the end of features
        ones = np.array(self.special_symbols["sample"])[np.newaxis,np.newaxis, np.newaxis]
        feat_augmented = ak.concatenate((data,ones), axis = -1)
        #Adding bos and eos at beginning and end of each event
        nfeats = int(ak.num(data, axis = -1)[0,0]) #number of initial features
        bos = np.array(([0] * nfeats + [self.special_symbols["bos"]]))[np.newaxis]
        eos = np.array(([0] * nfeats + [self.special_symbols["eos"]]))[np.newaxis]
        pad = np.array([0]*nfeats + [self.special_symbols["pad"]])

        #setting charge and pdg features of tokens to corresponding values 
        if data_type == "labels":
            np.put(pad,[0,1], -50)
            np.put(bos,[0,1],  -50)
            np.put(eos,[0,1], -50)

        feat_augmented = ak.concatenate([bos[np.newaxis], feat_augmented, eos[np.newaxis]], axis = 1)
        #Padding
        nsample_max_event = int(ak.max(ak.num(feat_augmented,axis = 1))) #max number of samples in the batch
        target = nsample_max_event + 1 #to keep <eos> token in target input of training 
        feat_padded = ak.pad_none(feat_augmented,target =target, clip = True, axis = 1)
        
        return ak.fill_none(feat_padded, value = pad, axis = None)
    
''' Lookup table assigning an index value to the keys provided in the constructor.
    Used to mask entries in CrossEntropyLoss
    The n special symbols are assigned automatically, and corresponds to the n first indices
    args:
        keys: keys of the vocabulary, without special symbols
        special_keys: keys corresponding to the special symbols '''
class Vocab(object):
    def __init__(self,keys, values  = None):
        if values is None:
            values = torch.arange(len(keys))
        self.vocab = dict(zip(keys,values))
    
    @classmethod
    def from_dict(cls, dict_):
        keys, values= list(dict_.keys()), list(dict_.values())
        return cls(keys, values)

    def tokens_to_indices(self,tokens):
        unique,indices_unique = torch.unique(tokens, return_inverse= True)
        key, values = torch.tensor(list(self.vocab.keys())), torch.tensor(list(self.vocab.values()))
        indices = torch.isin(key, unique)
        return values[indices][indices_unique]

    def indices_to_tokens(self, index):
        keys = torch.tensor(list(self.vocab.keys()))
        return keys[index.type(torch.int64)]
    
    def get_index(self, key):
        return self.vocab[key]
    
    def get_token(self, index):
        return list(self.vocab.keys())[index]

    def __len__(self):
        return len(self.vocab.keys())


class RMSNormalizer:
    def __init__(self, mean = 0., RMS = 0.):
        self.mean = mean
        self.RMS = RMS

    def normallize(self, data):
        if torch.all((self.RMS) > 1e-15):
            data.sub_(self.mean).div_(self.RMS)
        else:
            data.sub_(self.mean)

    def inverse_normalize(self, data_normalized):
        data_normalized.mul_(self.RMS).add_(self.mean)
    
    def set_attributes(self, mean, RMS):
        self.mean = mean
        self.RMS = RMS

'''Custom Dataset to store the hits and tracks
        Raw Data:
            feats: 3D awkward array of size (N,P,F), where 
                - N: number of events (50/file)
                - P: number of hits + tracks per event (variable size)
                - F: number of features = 10 (edep, x, y, z, time, track, charge, px, py, pz) in this order
                    if track is 0: hit in calorimeter, no info on momentum and charge (all set to 0)
                             is 1: x,y,z is pos of entry in calorimeter, px,py,pz, momentum from trajectory, E is set to 0
            labels: 3D awkward array of size (N,P,F), where
                - N: number of events (50/file)
                - P: number of hits + tracks per event (variable size)
                - F: number of features = 9, (hitid, mcid, pdg, charge, mass, px, py, pz (of mcp)), status
            
        Attributes: Obtained by transforming raw data
            - feats will be sliced to only contained edep, x, y, z, t (optional). All normalised
            - labels will be modified to contain charge, abs(pdg), E, direction.  
                direction is obtained by normalising 3momentum.
                Only one representative of each cluster is kept (the rest is duplicata anyways)
            '''
class CollectionHitsTraining(Dataset):
    def __init__(self, dir_path: str, special_symbols: dict,frac_files: float,
                 preprocessed: bool = False, E_cut: float = 0.1, do_tracks: bool = False, 
                 do_time: bool = False):
        '''args:
            dir_path: string path of directory where data is stored
            special_symbols: dict containing the special symbols. format is of the form
                special_symbols = {"pad": {"cont": pad_cont, "CEL": pad_CEL},
                                    "bos: {"cont": bos_cont, "CEL": bos_CEL},
                                    "eos": {"cont": eos_cont, "CEL": eos_CEL},
                                    "sample": sample_cont}
            preprocessed: bool. If True, preprocessed data will be loaded
            do_tracks: bool. If true, tracks are stored in the Dataset
            do_time: bool, if True, time of hits is kept
            E_cut: float, energy value [GeV] above which clusters are kept
            '''
        if frac_files < 0 or frac_files > 1:
            raise ValueError("The fraction of files must lie inbetween 0 and 1")

        super(CollectionHitsTraining,self).__init__()
        if preprocessed:
            nfiles = ceil(frac_files / 0.02)
            filenames = list(sorted(glob.iglob(dir_path + '/data/*.pt')))
            vocab_path = dir_path + "/vocabs/vocabs_normalizer.pt"
            charges_dict, pdgs_dict, self.E_label_RMS_normalizer = torch.load(vocab_path)
            self.vocab_charges, self.vocab_pdgs = Vocab.from_dict(charges_dict), Vocab.from_dict(pdgs_dict)
            feats, labels = [], []
            for f in filenames:
                feats_f, labels_f = torch.load(f)
                feats.append(feats_f)
                labels.append(labels_f)
            
            self.feats = torch.concatenate(feats)
            self.labels = torch.concatenate(labels)
        else:
            filenames = list(sorted(glob.iglob(dir_path + '/*.h5')))
            nfiles = ceil(frac_files * len(filenames))
            #print(nfiles)
            print("NEW VERSION")
            feats, labels = self._get_data(filenames, nfiles)
                 
            self.E_cut = E_cut
            self.do_tracks = do_tracks
            #removing tracks
            if do_tracks is False:
                hits_mask = ~(feats[:,:,5] == 1)
                feats = feats[hits_mask]
                labels = labels[hits_mask]
            #keeping time
            if do_time:
                feats = feats[:,:,:5]
            else:
                feats = feats[:,:,:4]

            PDGs_mask = np.abs(labels[...,2]) < 1e3
            labels = labels[PDGs_mask]

            self.E_label_RMS_normalizer = RMSNormalizer()
            self.E_feats_RMS_normalizer = RMSNormalizer()
            self.pos_feats_RMS_normalizer = RMSNormalizer()
            self.formatting(feats, labels, special_symbols)
            
    def _get_data(self,filenames, nfiles):
        if nfiles == 1:
            feats, labels = la.load_awkward2(filenames[0]) #get the events from the only file
        elif nfiles > 1:
            feats, labels = la.load_awkwards(filenames[:nfiles]) #get the events from each file
        else:
            raise ValueError(f"There is no h5py file in the directory {dir_path}")
        return ak.values_astype(feats, np.float32), ak.values_astype(labels, np.float32)


    def RMS_normalize(self, data,data_type: str):
        mean = torch.mean(data, dim = 0)
        RMS = torch.std(data, dim = 0)
        if data_type == "pos":
            self.pos_feats_RMS_normalizer.set_attributes(mean, RMS)
            self.pos_feats_RMS_normalizer.normallize(data)
        elif data_type == "E_label":
                self.E_label_RMS_normalizer.set_attributes(mean, RMS)
                self.E_label_RMS_normalizer.normallize(data)
        elif data_type == "E_feats":
                self.E_feats_RMS_normalizer.set_attributes(mean, RMS)
                self.E_feats_RMS_normalizer.normallize(data)
           

    
    '''
    formats the feats and labels by adding special tokens and normalising.
    params:
        feats: 
        labels: ak.Array, ragged, features of each particle are (charge, pdg, m, px, py, pz)
    '''
    def formatting(self, feats, labels, special_symbols):
        #Adding special symbols after formatting feats and labels
        add_special_symbols = AddSpecialSymbols(special_symbols)
        feats = add_special_symbols(self.format_feats(feats), "feats")
        labels = add_special_symbols(self.format_labels(labels), "labels")
        feats = torch.from_numpy(ak.to_numpy(feats)).to(dtype = torch.float32)
        labels = torch.from_numpy(ak.to_numpy(labels)).to(dtype = torch.float32)

        self.labels = labels
        self.feats = feats

        #Creating vocabularies:
        charges_keys = [-50,-1,0,1]
        abs_pdg_keys = torch.unique(labels[...,1]).tolist()
        self.vocab_charges = Vocab(charges_keys)
        self.vocab_pdgs = Vocab(abs_pdg_keys)
        labels[...,0] = self.vocab_charges.tokens_to_indices(labels[...,0])
        labels[...,1] = self.vocab_pdgs.tokens_to_indices(labels[...,1])

    '''called during shrink_labels to compute energy and normalize momentum
        Args:
            labels_flat: numpy array as a flatten version of labels 
                         features are in order:  (mass, px, py, pz)
                         
    '''    
    def format_E_pos_label(self, labels_flat):
        #Computing energies
        E = labels_flat[...,0]
        pvec = labels_flat[...,-3:]
        pvec_norm2 = torch.sum(torch.square(pvec), dim = -1) 
        #Computing Energy, normalizing and sorting
        E.square_().add_(pvec_norm2).sqrt_() # E = sqrt{m^2 + p^2}
        E.log10_()
        self.RMS_normalize(E, "E_label")
        pvec.div_(pvec_norm2.sqrt_().unsqueeze_(-1))

    '''Keep only 1 representative of each clusters in the label dataset 
        and keep only the features (charge, pdg, E, px,py,z)
        Args:
            do_track: to remove track info since hasn't been done previously
            labels: ak.Array with features: (hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status)
        
        Note: To avoid additional copies when c
    '''
    def format_labels(self, labels):
        if self.do_tracks is True:
            track_mask = (labels[...,0] > 1e-15) #(hitid <= 0) <-> tracks
            labels = labels[track_mask]
        
        mcids = labels[...,1]
        counts = ak.run_lengths(ak.sort(mcids,axis = -1)) #computing number of indentical mcids for each event
        dim_count = ak.num(counts, axis = 1) #future dimensions of list of clusters

        nevents = int(ak.num(labels, axis = 0))
        #making each event's mcids unique
        event_label = np.arange(1, nevents+1, step = 1)[:,np.newaxis]
        mc_id_cartesian = ak.cartesian([event_label, mcids], axis = -1) 
        #now entries are of the form [[(1,0),(1,0),...,(1,1),...,(1,39)], [(2,0),(2,0),...,(2,27)], ...]
        mc_id_flat = ak.flatten(mc_id_cartesian, axis = 1)
        #computing the indices of every unique entry of the flattened array
        _, indices_unique = np.unique(ak.to_numpy(mc_id_flat), axis = 0, return_index = True) 

        labels_flat = ak.flatten(labels, axis = 1)[indices_unique] #taking first representative 
        labels_flat_torch = torch.from_numpy(labels_flat.to_numpy()) #ref 
        labels_flat_torch[...,2].abs_() #absolute value of PDGs
        self.format_E_pos_label(labels_flat_torch[...,4:8])
        indices_features = [3,2,4,5,6,7] #3: charge 2: pdg, 4: mass, 5-7: momentum (mass to compute energy)
        #norm2_torch = labels_flat_torch[]
        labels = ak.unflatten(labels_flat_torch.numpy(), dim_count)[..., indices_features] #putting back to expected shape
        #Discarding low energy clusters 
        self.E_cut = np.log10(self.E_cut)
        self.E_cut -= self.E_label_RMS_normalizer.mean
        self.E_cut /= self.E_label_RMS_normalizer.RMS
        print(self.E_cut)
        E_mask = labels[...,2] > self.E_cut.item()
        labels = labels[E_mask]
        indices_sort_E = ak.argsort(labels[...,2], axis = -1, ascending= False)
        return labels[indices_sort_E] #sorting by descending energy

    def format_feats(self, feats):
        dim = ak.num(feats,axis = 1)
        feats_flat_np = ak.flatten(feats).to_numpy()
        feats_flat_torch = torch.from_numpy(feats_flat_np)
        self.RMS_normalize(feats_flat_torch[...,0], "E_feat")
        self.RMS_normalize(feats_flat_torch[...,-3:], "pos")
        return ak.unflatten(feats_flat_np, counts = dim)


    #necessary methods to override
    #called when applying len(), must be an integer (note: same numbers of feats than label)
    def __len__(self):
        return self.feats.size(dim = 0)
    
    #the sample is the list of hits for 1 event, same for labels 
    #called when indexing the dataset
    def __getitem__(self,id1):
        return self.feats[id1], self.labels[id1]


def get_data(dir_path, batch_size, frac_files,model_mode:str, preprocessed: bool = False, E_cut: float = 0.1, shuffle: bool = False):
    special_symbols = {
            "pad": 0,
            "bos": 1,
            "eos": 2,
            "sample": 3
    }
    print(preprocessed)
    if model_mode == "training":
        dir_path_train, dir_path_val = dir_path
        data_set_train = CollectionHitsTraining(dir_path_train,special_symbols, frac_files, preprocessed, E_cut)
        data_set_val = CollectionHitsTraining(dir_path_val, special_symbols, frac_files, preprocessed, E_cut)
        vocab_charges, vocab_pdgs = data_set_train.vocab_charges, data_set_train.vocab_pdgs
        vocab_charges_val, vocab_pdgs_val = data_set_val.vocab_charges, data_set_val.vocab_pdgs

        if len(vocab_charges) < len(vocab_charges_val):
            vocab_charges = vocab_charges
        if len(vocab_pdgs) < len(vocab_pdgs_val):
            vocab_pdgs = vocab_pdgs_val

        E_label_RMSNormalizer = data_set_train.E_label_RMS_normalizer
        return (vocab_charges, vocab_pdgs,
                special_symbols, E_label_RMSNormalizer, 
                DataLoader(data_set_train, batch_size= batch_size, shuffle = shuffle),
                DataLoader(data_set_val, batch_size= batch_size, shuffle = shuffle))    

    elif model_mode == "inference":
        dir_path_inference = dir_path[0]
        data_set = CollectionHitsTraining(dir_path_inference, special_symbols, frac_files, preprocessed)
        E_label_RMSNormalizer = data_set.E_label_RMS_normalizer
        return special_symbols, E_label_RMSNormalizer, DataLoader(data_set, batch_size = batch_size)
    else:
        raise ValueError(model_mode + " is an invalid entry. Must be either training or inference")    

def train_val_preprocessing(dir_train, dir_val, dir_res,frac_files, E_cut):
    special_symbols = {
            "pad": 0,
            "bos": 1,
            "eos": 2,
            "sample": 3
    }
    preprocessing(dir_train, dir_res, "training",special_symbols,frac_files, E_cut)
    preprocessing(dir_val, dir_res, "validation",special_symbols, frac_files,E_cut)

def preprocessing(dir_data, dir_res, datatype: str, special_symbols, frac_files,E_cut):
    start = time()
    ds = CollectionHitsTraining(dir_data, special_symbols,frac_files, False,E_cut)
    end = time() - start
    print(f"time needed to make preprocessing: {end} sec")
    E_label_normalizer = ds.E_label_RMS_normalizer
    nevents = len(ds.feats)
    nevents_per_file = int(nevents * 0.02)
    print(nevents_per_file)
    vocab_charges, vocab_pdgs = ds.vocab_charges, ds.vocab_pdgs
    torch.save([vocab_charges.vocab, vocab_pdgs.vocab, E_label_normalizer], dir_res + "/" + datatype + "/vocabs/vocabs_normalizer.pt")
    dl = DataLoader(ds, batch_size= nevents_per_file)

    for i, (feats, labels) in enumerate(dl):
        torch.save([feats, labels], dir_res + "/" + datatype + f"/ntau_10to100GeV_{i}")
        print(f"file {i+1} saved")
    


testing = False
if testing:
    dir_path = "/Users/paulwahlen/Desktop/Internship/ML/Code/TransfoV1/data"
    vocab_charges, vocab_pgs, special_symbols,E_label_RMSNormalizer, data_ld, val_dl = get_data((dir_path,dir_path),25,0.1, "training")
    feat0,label0 = next(iter(data_ld))
    print(label0[0,:10])
    pvec = label0[0,:,-4:-1]
    print(torch.sum(torch.square(pvec), dim = -1))
    #print(torch.max(feat0[0]))
    #print(label0[0].size(dim = 0))
    #print(vocab_charges.indices_to_tokens(label0[0,:,0]))
    #print(feats0[0])
    #print(torch.square(label0[0][...,3]) + torch.square(label0[0][...,4]) + torch.square(label0[0][...,5]))
    mean_E =0.
    for i, (batch_feat, batch_label) in enumerate(data_ld):
        #print(batch_feat.size())
        mean_E += torch.mean(batch_feat[...,0])     #mean energy
    
    print(mean_E)

