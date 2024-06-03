import torch
from torch.utils.data import Dataset, DataLoader
import load_awkward as la
import awkward as ak
import numpy as np
import glob

'''creates masks to avoid padding tokens influencing the attention.
    Position with True will be ignored.
    args:
        src: source, i.e., input to encoder, shape: (N,S,E)
        tgt: target, i.e., input to decoder, shape: (T,S,E)
        pad_symbol: unique identifier for padding tokens
    output:
        src_padding_mask: mask for the source padding tokens, shape: (N,S)
        tgt_padding_mask: mask for the target padding tokens, shape: (N,T)'''
def create_mask(src, tgt, pad_symbol):
    pad_symbol_broad = torch.tensor(pad_symbol).unsqueeze(0).unsqueeze(0)
    src_padding_mask = torch.all((src[...,-2:] == pad_symbol_broad), dim = -1)
    tgt_padding_mask = torch.all((tgt[...,-2:]== pad_symbol_broad), dim = -1)

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
        #First adding (0,0) indicating hit/MC data at the end of features
        ones = np.array(self.special_symbols["sample"])[np.newaxis][np.newaxis]
        feat_augmented = ak.concatenate((data,ones), axis = -1)
        #Adding bos and eos at beginning and end of each event
        nfeats = int(ak.num(data, axis = -1)[0,0]) #number of initial features
        bos = np.array(([0] * nfeats + self.special_symbols["bos"]["cont"]))[np.newaxis]
        eos = np.array(([0] * nfeats + self.special_symbols["eos"]["cont"]))[np.newaxis]
        pad = np.array([0]*nfeats + self.special_symbols["pad"]["cont"]) 
        #setting charge and pdg features of tokens to corresponding values 
        if data_type == "labels":
            np.put(pad,[0,1], self.special_symbols["pad"]["CEL"])
            np.put(bos,[0,1], self.special_symbols["bos"]["CEL"])
            np.put(eos,[0,1], self.special_symbols["eos"]["CEL"])

        feat_augmented = ak.concatenate([bos[np.newaxis], feat_augmented, eos[np.newaxis]], axis = 1)
        #Padding
        nsample_max_event = int(ak.max(ak.num(feat_augmented,axis = 1))) #max number of samples in the batch
        #target = nsample_max_event + 1 to keep <eos> token in target input of training
        feat_padded = ak.pad_none(feat_augmented,target = nsample_max_event + 1, clip = True, axis = 1)
        
        return ak.fill_none(feat_padded, value = pad, axis = None)
    
''' Lookup table assigning an index value to the keys provided in the constructor.
    Used to mask entries in CrossEntropyLoss
    The n special symbols are assigned automatically, and corresponds to the n first indices
    args:
        keys: keys of the vocabulary, without special symbols
        special_keys: keys corresponding to the special symbols '''
class Vocab(object):
    def __init__(self,keys, special_keys = [-150,-100,-50]):
        keys_pad = special_keys + keys
        values = torch.arange(len(keys_pad))
        self.vocab = dict(zip(keys_pad,values))
    
    @classmethod
    def from_dict(cls, dict_):
        keys = list(dict_.keys())
        return cls(keys[3:], keys[:3])

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
        if np.all((self.RMS) > 1e-15):
            return (data - self.mean)/self.RMS
        else:
            return data-self.mean

    def inverse_normalize(self, data_normalized):
        return data_normalized * self.RMS + self.mean 
    
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
                             is 1: x,y,z is pos of entry in calorimeter, px,py,pz, momentum from trajectory
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
    '''args:
            dir_path: string path of directory where data is stored
            special_symbols: dict containing the special symbols. format is of the form
                special_symbols = {"pad": {"cont": pad_cont, "CEL": pad_CEL},
                                    "bos: {"cont": bos_cont, "CEL": bos_CEL},
                                    "eos": {"cont": eos_cont, "CEL": eos_CEL},
                                    "sample": sample_cont}
            do_tracks: bool. If true, tracks are stored in the Dataset
            do_time: bool, if True, time of hits is kept'''
    def __init__(self, dir_path: str, special_symbols: dict,do_tracks: bool = False, do_time: bool = False):
        super(CollectionHitsTraining,self).__init__()
        filenames = list(sorted(glob.iglob(dir_path + '/*.h5')))
        if len(filenames) == 1:
            feats, labels = la.load_awkward2(filenames) #get the events from the only file
        elif len(filenames) > 1:
            feats, labels = la.load_awkwards(filenames) #get the events from each file
        else:
            raise ValueError(f"There is no h5py file in the directory {dir_path}")
        
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

        PDGs_mask = np.abs(labels[...,2])
        labels = labels[PDGs_mask]

        self.E_label_RMS_normalizer = RMSNormalizer()
        self.E_feats_RMS_normalizer = RMSNormalizer()
        self.pos_feats_RMS_normalizer = RMSNormalizer()
        self.formatting(feats, self.shrink_labels(do_tracks,labels), special_symbols)
    
    def RMS_normalize(self, data, data_type: str):
        if data_type == "pos":
            data_flat = ak.to_numpy(ak.flatten(data))
            mean = ak.mean(data_flat, axis = 0)
            data_centered = data_flat - mean
            RMS = np.sqrt((1.0/int(ak.num(data_flat,axis = 0))) *np.einsum("ij,ij", data_centered, data_centered))
            self.pos_feats_RMS_normalizer.set_attributes(mean, RMS)
            return ak.unflatten(self.pos_feats_RMS_normalizer.normallize(data_flat), ak.num(data, axis = 1))
        else:
            mean = ak.mean(data)
            RMS = ak.std(data)
            if data_type == "E_label":
                self.E_label_RMS_normalizer.set_attributes(mean, RMS)
                return self.E_label_RMS_normalizer.normallize(data)
            if data_type == "E_feats":
                self.E_feats_RMS_normalizer.set_attributes(mean, RMS)
                return self.E_feats_RMS_normalizer.normallize(data)
           

    
    '''
    formats the feats and labels by adding special tokens and normalising.
    params:
        feats: 
        labels: ak.Array, ragged, features of each particle are (charge, pdg, m, px, py, pz)
    '''
    def formatting(self, feats, labels, special_symbols):
        #Computing labels energy + total momentum
        pvec = labels[...,-3:]
        pvec_norm2 = ak.sum(np.square(pvec), axis = -1) 
        E_label = np.log10(np.sqrt(np.square(labels[...,1]) + pvec_norm2)) # E = sqrt{m^2 + p^2}
        E_label = self.RMS_normalize(E_label, "E_label")
        cluster_direction = pvec / np.sqrt(pvec_norm2) #normalising momentum
        charges = labels[...,0]
        abs_pdg = np.abs(labels[...,1])
        labels = ak.concatenate([ak.singletons(charges, axis = -1), #charge
                                 ak.singletons(abs_pdg, axis = -1), #abs(pdg)
                                 ak.singletons(E_label, axis = -1), #energy
                                 cluster_direction], #normalised p
                                 axis = -1)
        
        #Normalizing E and positions of feats:
        E_feat = np.log10(feats[...,0])
        E_feat = self.RMS_normalize(E_feat, "E_feats")
        pos = self.RMS_normalize(feats[...,1:], "pos")
        feats = ak.concatenate([ak.singletons(E_feat,axis = -1), pos], axis = -1)
        #Adding special symbols
        add_special_symbols = AddSpecialSymbols(special_symbols)
        self.feats = torch.from_numpy(ak.to_numpy(add_special_symbols(feats, "feats")))
        self.labels = torch.from_numpy(ak.to_numpy(add_special_symbols(labels, "labels")))

        #Creating vocabularies:
        charges_keys = np.unique(ak.to_numpy(ak.flatten(charges))).tolist()
        abs_pdg_keys = np.unique(ak.to_numpy(ak.flatten(abs_pdg))).tolist()
        special_tokens_CEL = [val["CEL"] for val in special_symbols.values() if isinstance(val, dict)]
        self.vocab_charges = Vocab(charges_keys, special_tokens_CEL)
        self.vocab_pdgs = Vocab(abs_pdg_keys, special_tokens_CEL)
        self.labels[...,0] = self.vocab_charges.tokens_to_indices(self.labels[...,0])
        self.labels[...,1] = self.vocab_pdgs.tokens_to_indices(self.labels[...,1])
    
    '''Keep only 1 representative of each clusters in the label dataset 
        and keep only the features (charge, pdg, m, px,py,z)'''
    def shrink_labels(self, do_track, labels):
        if do_track is True:
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

        labels__flat = ak.flatten(labels, axis = 1)[indices_unique] #taking first representative 
        indices_features = [3,2,4,5,6,7] #3: charge 2: pdg, 4: mass, 5-7: momentum (mass to compute energy)
        return ak.unflatten(labels__flat, dim_count)[..., indices_features] #putting back to expected shape

    #necessary methods to override
    #called when applying len(), must be an integer (note: same numbers of feats than label)
    def __len__(self):
        return self.feats.size(dim = 0)
    
    #the sample is the list of hits for 1 event, same for labels 
    #called when indexing the dataset
    def __getitem__(self,id1):
        return self.feats[id1], self.labels[id1]

'''Custom Dataset to store the hits and tracks
        Raw Data:
            feats: 3D awkward array of size (N,P,F), where 
                - N: number of events (50/file)
                - P: number of hits + tracks per event (variable size)
                - F: number of features = 10 (edep, x, y, z, time, track, charge, px, py, pz) in this order
                    if track is 0: hit in calorimeter, no info on momentum and charge (all set to 0)
                             is 1: x,y,z is pos of entry in calorimeter, px,py,pz, momentum from trajectory
            
        Attributes: Obtained by transforming raw data
            - feats will be sliced to only contained edep, x, y, z, t (optional). All normalised
            '''
class CollectionHitsInference(Dataset):
    '''args:
            dir_path: string path of directory where data is stored
            special_symbols: dict containing the special symbols. format is of the form
                special_symbols = {"pad": {"cont": pad_cont, "CEL": pad_CEL},
                                    "bos: {"cont": bos_cont, "CEL": bos_CEL},
                                    "eos": {"cont": eos_cont, "CEL": eos_CEL},
                                    "sample": sample_cont}
            do_tracks: bool. If true, tracks are stored in the Dataset
            do_time: bool, if True, time of hits is kept'''
    def __init__(self, dir_path: str, special_symbols: dict,do_tracks: bool = False, do_time: bool = False):
        super(CollectionHitsInference,self).__init__()
        filenames = list(sorted(glob.iglob(dir_path + '/*.h5')))
        if len(filenames) == 1:
            feats, _ = la.load_awkward2(filenames) #get the events from the only file
        elif len(filenames) > 1:
            feats, _ = la.load_awkwards(filenames) #get the events from each file
        else:
            raise ValueError(f"There is no h5py file in the directory {dir_path}")
        
        #removing tracks
        if do_tracks is False:
            hits_mask = ~(feats[:,:,5] == 1)
            feats = feats[hits_mask]
        #keeping time
        if do_time:
            feats = feats[:,:,:5]
        else:
            feats = feats[:,:,:4]

        self.formatting(feats, special_symbols)
    
    #formats the feats by adding special tokens and normalising.
    def formatting(self, feats, special_symbols):
        #Adding special symbols
        add_special_symbols = AddSpecialSymbols(special_symbols)
        self.feats = torch.from_numpy(ak.to_numpy(add_special_symbols(feats, "feats")))

        #Normalising feats inplace
        self.feats[:,:,0] = torch.tanh(self.feats[:,:,0]) #energy
        self.feats[:,:,1:3] /= 2000. #position in detector
        
    def __len__(self):
        return self.feats.size(dim = 0)
    
    #called when indexing the dataset
    def __getitem__(self,id1):
        return self.feats[id1]

    


# def get_data(dir_path, batch_size, special_symbols, model_mode: str):
#     if model_mode == "training":
#         data_set = CollectionHitsTraining(dir_path,special_symbols)
#         vocab_charges, vocab_pdgs = data_set.vocab_charges, data_set.vocab_pdgs
#         E_label_RMSNormalizer = data_set.E_label_RMS_normalizer
#         return (vocab_charges, vocab_pdgs, E_label_RMSNormalizer, DataLoader(data_set, batch_size= batch_size))    

#     elif model_mode == "inference":
#         data_set = CollectionHitsInference(dir_path, special_symbols)
#         return DataLoader(data_set, batch_size = batch_size)
#     else:
#         raise ValueError(model_mode + " is an invalid entry. Must be either training or inference")    
def get_data(dir_path_train, dir_path_val, batch_size, model_mode:str):
    special_symbols = {
            "pad": {"cont": [0.,1.],"CEL":-150},
            "bos": {"cont": [1.,1.], "CEL":-100},
            "eos": {"cont": [1.,0.],"CEL":-50},
            "sample": [0.,0.]
    }
    if model_mode == "training":
        data_set_train = CollectionHitsTraining(dir_path_train,special_symbols)
        data_set_val = CollectionHitsTraining(dir_path_val, special_symbols)
        vocab_charges, vocab_pdgs = data_set_train.vocab_charges, data_set_train.vocab_pdgs
        vocab_charges_val, vocab_pdgs_val = data_set_val.vocab_charges, data_set_val.vocab_pdgs

        if len(vocab_charges) < len(vocab_charges_val):
            vocab_charges = vocab_charges
        if len(vocab_pdgs) < len(vocab_pdgs_val):
            vocab_pdgs = vocab_pdgs_val

        E_label_RMSNormalizer = data_set_train.E_label_RMS_normalizer
        return (vocab_charges, vocab_pdgs,
                special_symbols, E_label_RMSNormalizer, 
                DataLoader(data_set_train, batch_size= batch_size),
                DataLoader(data_set_val, batch_size= batch_size))    

    elif model_mode == "inference":
        data_set = CollectionHitsInference(dir_path, special_symbols)
        return DataLoader(data_set, batch_size = batch_size)
    else:
        raise ValueError(model_mode + " is an invalid entry. Must be either training or inference")    


special_symbols = {
            "pad": {"cont": [0.,1.],"CEL":-150},
            "bos": {"cont": [1.,1.], "CEL":-100},
            "eos": {"cont": [1.,0.],"CEL":-50},
            "sample": [0.,0.]
    }

testing = False
if testing:
    dir_path = "/Users/paulwahlen/Desktop/Internship/ML/Code/TransfoV1/data"
    vocab_charges, vocab_pgs, special_symbols,E_label_RMSNormalizer, data_ld = get_data(dir_path,25,special_symbols, "training")
    feat0,label0 = next(iter(data_ld))
    print(feat0[0,0:30])
    print(torch.max(feat0[0]))
    print(label0[0].size(dim = 0))
    print(vocab_charges.indices_to_tokens(label0[0,:,0]))
    #print(feats0[0])
    #print(torch.square(label0[0][...,3]) + torch.square(label0[0][...,4]) + torch.square(label0[0][...,5]))
    mean_E =0.
    for batch_feat, batch_label in data_ld:
        print(batch_feat.size())
        mean_E += torch.mean(batch_feat[...,0])     #mean energy
    
    print(mean_E)