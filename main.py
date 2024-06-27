from time import time

import torch
from torch.nn.functional import normalize
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import ceil

from data_prepro import create_mask, get_data, Vocab
from model import ClustersFinder, CustomDecoderLayer, CustomDecoder
from argparse import ArgumentParser
import json
import logging
DEVICE = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


def get_loss_log_freq(nlog_per_epoch,nbatches):
    period = nbatches // nlog_per_epoch
    if period == 0:
        period = 1

    nlog_update = nbatches // period
    return {"period": period, "nlog": nlog_update}

def add_pad_final(input, pad_tokens, size):
    nhits_input = input.shape[1]
    diff_size = size - nhits_input
    if diff_size > 0:
        pad = pad_tokens.repeat(input.shape[0], diff_size,1)
        input = torch.cat([input, pad], dim = 1)
    return input

def translate_E(logits_cont, rms_normalizer):
    if logits_cont.dim() < 3:
        E = logits_cont
    else:
        E = logits_cont[...,0]
    return torch.pow(10,rms_normalizer.inverse_normalize(E))

def translate(input, vocab_charges, vocab_pdgs, rms_normalizer):
    input[...,0] = vocab_charges.indices_to_tokens(input[...,0])
    input[...,1] = vocab_pdgs.indices_to_tokens(input[...,1])
    input[...,2] = translate_E(input[...,2], rms_normalizer)

def create_model(config, version, vcharges_size, vpdgs_size):
    if config["dtype"] == "torch.float32":
        dtype = torch.float32
    
    version = args.model
    if version == 1:
        decoder = None
    elif version == 2:
        decoder_layer = CustomDecoderLayer(d_label = config["d_out_embedder_tgt"], 
                                           d_proj = config["d_out_embedder_src"],
                                           nproj = config["nproj"],
                                           nhead = config["nhead"],
                                           dim_ff_sub = config["dim_ff_sub_decoder"],
                                           dim_ff_main = config["dim_ff_main_decoder"],
                                           batch_first= True,
                                           dtype = dtype,
                                           device = DEVICE)
        decoder = CustomDecoder(decoder_layer, config["nlayers_decoder"])

    #Creating model
    model = ClustersFinder(
        dmodel = config["dmodel"],
        nhead = config["nhead"],
        d_ff_trsf = config["nhid_ff_trsf"],
        custom_decoder= decoder,
        nlayers_encoder= config["nlayers_encoder"],
        nlayers_decoder= config["nlayers_decoder"],
        nlayers_embedder_src = config["nlayers_embedder_src"],
        d_ff_embedder_src = config["d_ff_embedder_src"],
        d_out_embedder_src= config["d_out_embedder_src"],
        nlayers_embedder_tgt = config["nlayers_embedder_tgt"],
        d_ff_embedder_tgt = config["d_ff_embedder_tgt"],
        d_out_embedder_tgt= config["d_out_embedder_tgt"],
        d_input_encoder = config["d_input_encoder"],
        d_input_decoder = config["d_input_decoder"],
        nparticles_max= vpdgs_size,
        ncharges_max= vcharges_size,
        DOF_continous= config["output_DOF_continuous"],
        device = DEVICE,
        dtype = dtype
    ).to(DEVICE)
    return model

'''
Transforms a 3D unit vector encoded by theta and phi to a 3D vector in cartesian coordinates.
args:
    input: output of decoder, with energy (to avoid another concatenation afterwards)
           features are in order (E, theta, phi)
'''
def get_cartesian_from_angles(input):
    #logits: (E, theta, phi)
    theta = input[...,-2]
    phi = input[...,-1]
    sin_theta = torch.sin(theta) 
    nx = torch.cos(phi) * sin_theta
    ny = torch.sin(phi) * sin_theta
    return torch.concat([input[...,0].unsqueeze(-1),
                         nx.unsqueeze(-1),
                         ny.unsqueeze(-1), 
                         torch.cos(theta).unsqueeze(-1)], dim = -1)
'''
Feeds the input to the model and outputs the translated version by a greedy algorithm. 
Each tokens per event is predicted sequentially, until either all the events were given an eos token or 
the max number of predictable clusters was reached (ncluster_max). 
Flow is:
    1. src fed to encoder with src_paddind_mask to obtain memory
    2. iterates until all event are associated with <eos> or ncluster_max is reached
        2.a. decoder output of prevous iter is fed to decoder, along with memory
        2.b. Logits are computed 
        2.c. classes indices and cont. DOF are obtained, 
        2.d. Special tokens are added to cont. DOF and events which are finished are marked as done
        2.c. concatenation of all variables to form batch of new tokens
        2.d. new batch is added to old, tgt_padding_mask is updated if any <eos> were predicted
args:
    model: model to use
    src: batch of hits, shape: (N,S,F)
    src_padding_mask: mask to avoid padding tokens to interfer, shape: (N,S)
    nlcuster_max: int, determine the max. number of clusters predictable
    special_symbols: dict, defines the special tokens with their values for cont. DOF and for categories
    nfeats_labels: int, number of labels features 

'''
def greedy_func(model,src,vocab_charges, ncluster_max: int, special_symbols: dict, nfeats_labels: int = 6, dtype = torch.float32):
    with torch.no_grad():
        src = src.to(DEVICE)
        #src_padding_mask = src_padding_mask.to(DEVICE)

        #Creating <bos> token
        bos_np = np.array([0]*nfeats_labels + [special_symbols["bos"]])
        bos = torch.from_numpy(bos_np).to(device = DEVICE, dtype = dtype)
        
        eos_cont_tensor = torch.tensor(special_symbols["eos"], device = DEVICE)
        bos_cont_tensor = torch.tensor(special_symbols["bos"], device = DEVICE)
        sample_cont_tensor = torch.tensor(special_symbols["sample"], device = DEVICE)
        
        pad_np  = np.array([0]*nfeats_labels + [special_symbols["pad"]]) 
        pad = torch.from_numpy(pad_np).to(device = DEVICE, dtype = dtype)
        
        batch_size = src.shape[0]
        clusters_transfo = torch.tile(bos,(batch_size,1)).to(DEVICE).unsqueeze_(1) #output of decoder, then updated to input decoder
        src_padding_mask, tgt_key_padding_mask = create_mask(src,clusters_transfo, special_symbols["pad"],DEVICE)
        #feeding source to encoder
        memory = model.encode(src,src_padding_mask).to(DEVICE)
        #tgt_key_padding_mask = torch.zeros(batch_size,1).type(torch.bool).to(DEVICE)
        is_done = torch.zeros(batch_size,1).type(torch.bool).to(DEVICE) #to keep track of which event has eos token
        is_done_prev = torch.clone(is_done) #to keep track of previous status of eos tokens
        for _ in range(ncluster_max-1):
            #Feeding previous decoder output as input
            out_decoder = model.decode(clusters_transfo, memory, tgt_key_padding_mask, src_padding_mask)
            #Computing the logits, only considering the last row. 
            logits_charges_batch = model.lastlin_charge(out_decoder)[:,-1] #shape is [N,T,F]
            logits_pdg_batch = model.lastlin_pdg(out_decoder)[:,-1]
            next_DOF_cont_batch = model.lastlin_cont(out_decoder)[:,-1]
        
            #no need to apply softmax since it's a bijection, and no need to print probabilities
            next_charges_batch = torch.argmax(logits_charges_batch, dim = 1, keepdim = True) #class id same as index by construction
            next_pdgs_batch = torch.argmax(logits_pdg_batch, dim = 1, keepdim = True) #class id same as index by construction
            
            #Addding special tokens of cont. DOF, ignoring the values if event has already <eos>
            new_eos_tokens_batch = ( (next_charges_batch == vocab_charges.get_index(special_symbols["eos"]["CEL"])) * ~is_done_prev).squeeze(-1)
            new_bos_tokens_batch = ( (next_charges_batch == vocab_charges.get_index(special_symbols["bos"]["CEL"])) * ~is_done_prev).squeeze(-1)
            new_sample_tokens_batch = ~new_eos_tokens_batch * ~new_bos_tokens_batch
            #Using spherical coordinates to get 3D direction vectors
            next_DOF_cont_batch = get_cartesian_from_angles(next_DOF_cont_batch)
            next_DOF_cont_batch_spe = torch.zeros((batch_size, next_DOF_cont_batch.shape[-1] + 2), device = DEVICE, dtype = dtype)
            n_new_eos = torch.count_nonzero(new_eos_tokens_batch)
            if  n_new_eos> 0:
                is_done[~is_done_prev] = new_eos_tokens_batch[~is_done_prev.squeeze(-1)]
                next_DOF_cont_batch_spe[new_eos_tokens_batch] = torch.cat(( next_DOF_cont_batch[new_eos_tokens_batch], 
                                                                                        eos_cont_tensor.repeat(n_new_eos,1) ), 
                                                                                      dim = -1)
            #shouldn't be predicted but just in case
            n_new_bos = torch.count_nonzero(new_bos_tokens_batch)
            if  n_new_bos > 0:
                next_DOF_cont_batch_spe[new_bos_tokens_batch] = torch.cat(( next_DOF_cont_batch[new_bos_tokens_batch], 
                                                                                        bos_cont_tensor.repeat(n_new_bos,1) ), 
                                                                                        dim = -1)
            n_new_samples = torch.count_nonzero(new_sample_tokens_batch)
            if  n_new_samples > 0:
                next_DOF_cont_batch_spe[new_sample_tokens_batch] = torch.cat(( next_DOF_cont_batch[new_sample_tokens_batch],
                                                                                           sample_cont_tensor.repeat(n_new_samples,1)),
                                                                                           dim = -1)

        
            next_clusters_batch = torch.cat((next_charges_batch, 
                                             next_pdgs_batch,
                                             next_DOF_cont_batch_spe), dim = -1) 
            #Need to change the next cluster of every event which was done previously to a pad
            #if torch.all(~is_done_prev):
            next_clusters_batch[is_done_prev.squeeze(-1)] = pad
            #Updating the tgt_padding_mask
            tgt_key_padding_mask = torch.hstack([tgt_key_padding_mask, is_done_prev])
            
            clusters_transfo = torch.concat([clusters_transfo, next_clusters_batch.unsqueeze(1)], dim = 1)
            is_done_prev = torch.clone(is_done)
            #print(f"Memory allocated: {torch.cuda.memory_allocated(device = DEVICE)}")
            
            if torch.all(is_done):
                break
    clusters_transfo = add_pad_final(clusters_transfo, pad, ncluster_max)
    return clusters_transfo

def inference(config, args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename= config["dir_results"] + "log_inference.txt", level= logging.INFO)
    file_handler = logging.FileHandler(logger.name, mode = 'w')
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.info("Inference chosen...")
    logging.info("Loading the vocabularies...")
    vocab_charges = Vocab.from_dict(torch.load(config["path_charges"] + "vocab_charges.pt"))
    vocab_pdgs = Vocab.from_dict(torch.load(config["path_PDGs"] + "vocab_PDGs.pt"))
    model = create_model(config, args.model, len(vocab_charges), len(vocab_pdgs))

    #Loading weights
    model.load_state_dict(torch.load(config["dir_model"] + "best_model.pt"))
    model.eval()
    logging.info(f"Model created on {model.device}, now loading the source")

    if config["dtype"] == "torch.float32":
        dtype = torch.float32
    
    
    special_symbols, E_label_RMS_normalizer, src_loader = get_data((config["dir_path_inference"], ), 
                                                                    config["batch_size_test"], 
                                                                    config["frac_files_test"], "inference", 
                                                                    config["preprocessed"], config["E_cut"])
    logging.info("Saving normalizer...")
    torch.save(E_label_RMS_normalizer, config["dir_results"] + "E_RMS_normalizer.pt")
    logging.info("Going to inference now")
    pred = []
    labels = []
    for src, tgt in src_loader:
        start_time = time()
        clusters_out = greedy_func(model, src,vocab_charges,config["ncluster_max"],special_symbols,config["d_input_decoder"] -2, dtype).to("cpu")
        #print(clusters_out.device)
        translate(clusters_out, vocab_charges, vocab_pdgs,E_label_RMS_normalizer)
        pred.append(clusters_out)
        translate(tgt, vocab_charges, vocab_pdgs, E_label_RMS_normalizer)
        labels.append(tgt)
        delta_t = time() - start_time
        logging.info(f"Batch done in {delta_t} seconds")
    
    output = torch.cat(pred, dim = 0)
    torch.save(output, config["dir_results"] + "prediction.pt")
    labels = torch.cat(labels, dim = 0)
    torch.save(labels, config["dir_results"] + "labels.pt")



#train the model for one epoch
#src: input hits from simulation, tgt: MC truth clusters 
def train_epoch(model, optim, train_dl, special_symbols,vocab_charges, vocab_pdgs, E_rms_normalizer,
                hyperweights_lossfn, loss_fn_charges, loss_fn_pdg,loss_fn_cont, loss_fn_tokens,
                nlog_period_epoch, loss_evo, epoch):
    model.train() #setting model into train mode

    period = nlog_period_epoch["period"]
    nlog_epoch = nlog_period_epoch["nlog"]
    loss_epoch_tot = 0.
    loss_epoch = 0.0
    loss_epoch_charges = 0.0
    loss_epoch_pdgs = 0.0
    loss_epoch_cont = 0.0
    loss_epoch_tokens = 0.
    count_log = 0
    for i,(src,tgt) in enumerate(train_dl):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        #print(f"allocated memory on GPU after moving: {torch.cuda.memory_allocated(device = DEVICE)}")
        src_padding_mask, tgt_padding_mask = create_mask(src,tgt,special_symbols["pad"], DEVICE)
        tgt_in_padding_mask = tgt_padding_mask[:,:-1]
        tgt_out_padding_mask = tgt_padding_mask[:,1:]
        tgt_in = tgt[:,:-1] #sets the dimensions of transformer output -> must have the same as tgt_out
        logits_charges, logits_pdg, logits_cont, logits_tokens = model(src,tgt_in, 
                                                                src_padding_mask,
                                                                tgt_in_padding_mask,
                                                                src_padding_mask)
        optim.zero_grad()
        tgt_out = tgt[:,1:,:] #logits are compared with tokens shifted
        tgt_out_charges = tgt_out[...,0].to(torch.long)
        tgt_out_pdg = tgt_out[...,1].to(torch.long)
        tgt_out_cont = tgt_out[...,2:-1] #only (E, n_x,n_y,n_z)
        tgt_out_tokens = tgt_out[...,-1]
        #Using spherical coordinates to get 3D direction vectors
        logits_cont = get_cartesian_from_angles(logits_cont)
        #special_tokens are not taken into account in the continuous loss
        eos_bos_mask = ((tgt_out_tokens == vocab_charges.get_index(special_symbols["eos"]))
                        + (tgt_out_tokens == vocab_charges.get_index(special_symbols["bos"]))) 
        spe_tokens_mask = eos_bos_mask + tgt_out_padding_mask
        #Computing the losses
        loss_charges = loss_fn_charges(logits_charges.transpose(dim0 = -2, dim1 = -1), tgt_out_charges)
        loss_pdg = loss_fn_pdg(logits_pdg.transpose(dim0 = -2, dim1 = -1), tgt_out_pdg)
        loss_cont_vec = loss_fn_cont(logits_cont, tgt_out_cont)
        loss_tokens = loss_fn_tokens(logits_tokens.transpose(-2,-1), tgt_out_tokens)
        #nspe_tokens = torch.count_nonzero(spe_tokens_mask, dim = -1)
        #n_nospe = spe_tokens_mask.shape[-1] - nspe_tokens
        loss_cont = torch.mean(torch.sum(loss_cont_vec[~spe_tokens_mask], dim = -1))
        
        loss = loss_charges * hyperweights_lossfn[0] + loss_pdg * hyperweights_lossfn[1] + loss_cont*hyperweights_lossfn[2] + loss_tokens * hyperweights_lossfn[-1]
        loss.backward()
        optim.step()

        #logging.info(f"training: batch done")
        loss_epoch += loss.item()
        loss_epoch_tot += loss.item()
        loss_epoch_charges += loss_charges.item()
        loss_epoch_pdgs += loss_pdg.item()
        loss_epoch_cont += loss_cont.item()
        loss_epoch_tokens += loss_tokens.item()
        size_batch = len(train_dl)

        if (i + 1) % period == 0:
            #logging.info(f"recording the losses, training, minibatch = {i + 1}")
            #print(nlog_epoch)
            loss_evo["train"][nlog_epoch *epoch + count_log] = loss_epoch / period
            loss_evo["charges_train"][nlog_epoch *epoch + count_log] = loss_epoch_charges / period
            loss_evo["pdgs_train"][nlog_epoch *epoch + count_log] = loss_epoch_pdgs / period
            loss_evo["cont_train"][nlog_epoch *epoch + count_log] = loss_epoch_cont / period
            loss_evo["tokens_train"][nlog_epoch *epoch + count_log] = loss_epoch_tokens / period

            loss_epoch = 0.
            loss_epoch_charges = 0.
            loss_epoch_pdgs = 0.
            loss_epoch_cont = 0.
            loss_epoch_tokens = 0.
            count_log += 1
        #print(f"Memory allocated end of batch: {torch.cuda.memory_allocated(DEVICE) / 1e9} GB")
        #return (loss_epoch / size_batch, loss_epoch_charges / size_batch, loss_epoch_pdgs / size_batch, loss_epoch_cont / size_batch)
    return loss_epoch_tot / size_batch

def validate_epoch(model, val_dl, special_symbols,vocab_charges, vocab_pdgs, E_rms_normalizer,
                   hyperweights_lossfn, loss_fn_charges, loss_fn_pdg,loss_fn_cont, loss_fn_tokens,
                   nlog_period_epoch, loss_evo,epoch):  
    model.eval() #setting model into train mode

    period = nlog_period_epoch["period"]
    nlog_epoch = nlog_period_epoch["nlog"]
    loss_epoch_tot = 0.
    loss_epoch = 0.0
    loss_epoch_charges = 0.0
    loss_epoch_pdgs = 0.0
    loss_epoch_cont = 0.0
    loss_epoch_tokens = 0.
    count_log = 0
    for i, (src,tgt) in enumerate(val_dl):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        with torch.no_grad():
            #print(f"allocated memory on GPU after moving: {torch.cuda.memory_allocated(device = DEVICE)}")
            src_padding_mask, tgt_padding_mask = create_mask(src,tgt,special_symbols["pad"], DEVICE)
        tgt_in_padding_mask = tgt_padding_mask[:,:-1]
        tgt_out_padding_mask = tgt_padding_mask[:,1:]
        tgt_in = tgt[:,:-1] #sets the dimensions of transformer output -> must have the same as tgt_out
        logits_charges, logits_pdg, logits_cont, logits_tokens = model(src,tgt_in, 
                                                                src_padding_mask,
                                                                tgt_in_padding_mask,
                                                                src_padding_mask)
        optim.zero_grad()
        tgt_out = tgt[:,1:,:] #logits are compared with tokens shifted
        tgt_out_charges = tgt_out[...,0].to(torch.long)
        tgt_out_pdg = tgt_out[...,1].to(torch.long)
        tgt_out_cont = tgt_out[...,2:-1] #only (E, n_x,n_y,n_z)
        tgt_out_tokens = tgt_out[...,-1]
        #Using spherical coordinates to get 3D direction vectors
        logits_cont = get_cartesian_from_angles(logits_cont)
        #special_tokens are not taken into account in the continuous loss
        eos_bos_mask = ((tgt_out_tokens == vocab_charges.get_index(special_symbols["eos"]))
                        + (tgt_out_tokens == vocab_charges.get_index(special_symbols["bos"]))) 
        spe_tokens_mask = eos_bos_mask + tgt_out_padding_mask
        #Computing the losses
        loss_charges = loss_fn_charges(logits_charges.transpose(dim0 = -2, dim1 = -1), tgt_out_charges)
        loss_pdg = loss_fn_pdg(logits_pdg.transpose(dim0 = -2, dim1 = -1), tgt_out_pdg)
        loss_cont_vec = loss_fn_cont(logits_cont, tgt_out_cont)
        loss_tokens = loss_fn_tokens(logits_tokens.transpose(-2,-1), tgt_out_tokens)
        #nspe_tokens = torch.count_nonzero(spe_tokens_mask, dim = -1)
        #n_nospe = spe_tokens_mask.shape[-1] - nspe_tokens
        loss_cont = torch.mean(torch.sum(loss_cont_vec[~spe_tokens_mask], dim = -1))
        
        loss = loss_charges * hyperweights_lossfn[0] + loss_pdg * hyperweights_lossfn[1] + loss_cont*hyperweights_lossfn[2] + loss_tokens * hyperweights_lossfn[-1]

        #logging.info(f"training: batch done")
        loss_epoch += loss.item()
        loss_epoch_tot += loss.item()
        loss_epoch_charges += loss_charges.item()
        loss_epoch_pdgs += loss_pdg.item()
        loss_epoch_cont += loss_cont.item()
        loss_epoch_tokens += loss_tokens.item()
        size_batch = len(val_dl)

        if (i + 1) % period == 0:
            #logging.info(f"recording the losses, training, minibatch = {i + 1}")
            #print(nlog_epoch)
            loss_evo["val"][nlog_epoch *epoch + count_log] = loss_epoch / period
            loss_evo["charges_val"][nlog_epoch *epoch + count_log] = loss_epoch_charges / period
            loss_evo["pdgs_val"][nlog_epoch *epoch + count_log] = loss_epoch_pdgs / period
            loss_evo["cont_val"][nlog_epoch *epoch + count_log] = loss_epoch_cont / period
            loss_evo["tokens_val"][nlog_epoch *epoch + count_log] = loss_epoch_tokens / period

            loss_epoch = 0.
            loss_epoch_charges = 0.
            loss_epoch_pdgs = 0.
            loss_epoch_cont = 0.
            loss_epoch_tokens = 0.
            count_log += 1
    #return (loss_epoch / size_batch, loss_epoch_charges / size_batch, loss_epoch_pdgs / size_batch, loss_epoch_cont / size_batch)
    return loss_epoch_tot / size_batch

def train_and_validate(config, args):

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename= config["dir_results"] + "log.txt", level= logging.INFO)
    file_handler = logging.FileHandler(logger.name, mode = 'w')
    logger.addHandler(file_handler)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info("Getting the training data from" +config["dir_path_train"])
    vocab_charges, vocab_pdgs, special_symbols, E_rms_normalizer, train_dl, val_dl = get_data(dir_path = (config["dir_path_train"], config["dir_path_val"]), 
                                                                                              batch_size = config["batch_size"], 
                                                                                              frac_files = config["frac_files"],
                                                                                              model_mode = "training",
                                                                                              preprocessed= config["preprocessed"],
                                                                                              E_cut= config["E_cut"])
    torch.save(vocab_charges.vocab, config["dir_results"] + "vocab_charges.pt")
    torch.save(vocab_pdgs.vocab, config["dir_results"] + "vocab_PDGs.pt")
    
    logging.info("Loaded the data and saved vocabularies")

    print(f"memory on CUDA, model not yet created: {torch.cuda.memory_allocated(DEVICE)}")
    model = create_model(config, args.model,len(vocab_charges), len(vocab_pdgs))
    print(f"Created model on CUDA {model.device}, memory allocated: {torch.cuda.memory_allocated(DEVICE) / 1e9} GB")
    nparams = sum(param.numel() for param in model.parameters())
    print(f"number of parameters: {nparams}")
    logging.info("Created model, computing logging frequencies...")
    optim = torch.optim.Adam(model.parameters(), lr = config["lr"])
    loss_fn_charges = nn.CrossEntropyLoss(reduction ='none')
    loss_fn_pdgs = nn.CrossEntropyLoss(reduction ='none')
    loss_fn_cont = nn.MSELoss(reduction = 'none')
    loss_fn_tokens = nn.CrossEntropyLoss(ignore_index= special_symbols["pad"], reduction= "mean")

    val_loss_min = 1e9
    nepoch = config["epochs"]
    nlog_epoch = config["nlog_epoch"]

    nbatches_train = len(train_dl)
    nbatches_val = len(val_dl)

    nlog_period_train = get_loss_log_freq(nlog_epoch,nbatches_train)
    nlog_period_val = get_loss_log_freq(nlog_epoch,nbatches_val)
    nloss_train = int(nepoch * nlog_period_train["nlog"])
    nloss_val = int(nepoch * nlog_period_val["nlog"])
    losses_evolution = {"train": np.zeros(nloss_train),
                        "val": np.zeros(nloss_val),
                        "charges_train": np.zeros(nloss_train),
                        "pdgs_train": np.zeros(nloss_train),
                        "cont_train": np.zeros(nloss_train),
                        "tokens_train": np.zeros(nloss_train),
                        "charges_val": np.zeros(nloss_val),
                        "pdgs_val": np.zeros(nloss_val),
                        "cont_val": np.zeros(nloss_val),
                        "tokens_val": np.zeros(nloss_val)}
    logging.info(f"Logging frequencies computed, nlog_epoch modified as: {nlog_epoch} to {nlog_period_train['nlog']}. Number of losses that will be recorded is thus: {nloss_train} for the training set and {nloss_val}")
    logging.info("Starting training...")
    for i in range(nepoch):
        start_time = time()
        # train_loss_epoch, charges_train, pdgs_train, cont_train = train_epoch(model,
        #                                optim = optim,
        #                                train_dl=train_dl,
        #                                hyperweights_lossfn= config["hyper_loss"],
        #                                special_symbols = special_symbols,
        #                                vocab_charges = vocab_charges,
        #                                vocab_pdgs= vocab_pdgs,
        #                                loss_fn_charges = loss_fn_charges,
        #                                loss_fn_pdg= loss_fn_pdgs,
        #                                loss_fn_cont= loss_fn_cont
        #                                nlog_epoch = nlog_epoch_train)
        train_loss_epoch = train_epoch(model,
                                       optim = optim,
                                       train_dl=train_dl,
                                       hyperweights_lossfn= config["hyper_loss"],
                                       special_symbols = special_symbols,
                                       vocab_charges = vocab_charges,
                                       vocab_pdgs= vocab_pdgs,
                                       E_rms_normalizer = E_rms_normalizer,
                                       loss_fn_charges = loss_fn_charges,
                                       loss_fn_pdg= loss_fn_pdgs,
                                       loss_fn_cont= loss_fn_cont,
                                       loss_fn_tokens= loss_fn_tokens,
                                       nlog_period_epoch = nlog_period_train,
                                       loss_evo = losses_evolution,
                                       epoch = i)
        time_epoch = time() - start_time 
        logging.info("Finished training for one epoch, going to valiation")
        # val_loss_epoch, charges_val, pdgs_val,cont_val = validate_epoch(model,
        #                                val_dl=val_dl,
        #                                hyperweights_lossfn= config["hyper_loss"],
        #                                special_symbols = special_symbols,
        #                                vocab_charges = vocab_charges,
        #                                vocab_pdgs= vocab_pdgs,
        #                                loss_fn_charges = loss_fn_charges,
        #                                loss_fn_pdg= loss_fn_pdgs,
        #                                loss_fn_cont= loss_fn_cont
        #                                nlog_epoch = nlog_epoch_val) 
        val_loss_epoch = validate_epoch(model,
                                       val_dl=val_dl,
                                       hyperweights_lossfn= config["hyper_loss"],
                                       special_symbols = special_symbols,
                                       vocab_charges = vocab_charges,
                                       vocab_pdgs= vocab_pdgs,
                                       E_rms_normalizer = E_rms_normalizer,
                                       loss_fn_charges = loss_fn_charges,
                                       loss_fn_pdg= loss_fn_pdgs,
                                       loss_fn_cont= loss_fn_cont,
                                       loss_fn_tokens= loss_fn_tokens,
                                       nlog_period_epoch = nlog_period_val,
                                        loss_evo = losses_evolution,
                                        epoch = i) 

        # losses_evolution["train"][i] = train_loss_epoch
        # losses_evolution["val"][i] = val_loss_epoch
        # losses_evolution["time"][i] = time_epoch
        # losses_evolution["charges_train"][i] = charges_train
        # losses_evolution["pdgs_train"][i] = pdgs_train
        # losses_evolution["cont_train"][i] = cont_train
        # losses_evolution["charges_val"][i] = charges_val
        # losses_evolution["pdgs_val"][i] = pdgs_val
        # losses_evolution["cont_val"][i] = cont_val

        if val_loss_epoch < val_loss_min:
            logging.info("New best Model, saving...")
            torch.save(model.state_dict(), config["dir_results"] + "best_model.pt")
            val_loss_min = val_loss_min

        logging.info(f"{i + 1} epoch done, time: {time_epoch}, val_loss: {val_loss_epoch}, train_loss: {train_loss_epoch}")
    
    torch.save(losses_evolution, config["dir_results"] + "losses.pt")
    logging.info("Finished all epochs and saved the losses")

if __name__ == "__main__":

    parser = ArgumentParser(prog= "Finding clusters training and inference ")
    #Inference
    parser.add_argument("--inference", action= "store_true", help = "Set true to run inference")
    parser.add_argument("-config_path", type = str, help = "Directory of ConfigFile")
    parser.add_argument("-device", type = int, help = "Number of cuda device to use")
    parser.add_argument("-model", type = int, help = "Number of model implementation to use")

    args = parser.parse_args()
    DEVICE = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path + "ConfigFile.json") as f:
        config = json.load(f)

    if args.inference:
        inference(config,args)
    else:
        train_and_validate(config,args)

        
        
