from time import time

import torch
from torch.nn.functional import normalize
import torch.nn as nn
import torch.optim as optim

from data import create_mask, get_data, Vocab
from model import ClustersFinder
from argparse import ArgumentParser
import json
import logging
DEVICE = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

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
def greedy_func(model,src,src_padding_mask,vocab_charges, ncluster_max: int, special_symbols: dict, nfeats_labels: int = 6):
    src = src.to(DEVICE)
    src_padding_mask = src_padding_mask.to(DEVICE)

    #feeding source to encoder
    memory = model.encode(src,src_padding_mask).to(DEVICE)
    #Creating <bos> token
    bos = torch.tensor([0]*nfeats_labels + special_symbols["bos"]["cont"])
    bos.put_([0,1], special_symbols["bos"]["CEL"])

    batch_size = src.size[0]
    clusters_transfo = torch.tile(bos,(batch_size,1)).to(DEVICE) #output of decoder, then updated to input decoder
    tgt_key_padding_mask = torch.zeros(batch_size,1).type(torch.bool).to(DEVICE)
    is_done = torch.zeros(batch_size,1).type(torch.bool) #to keep track of which event has eos token
    is_done_prev = is_done #to keep track of previous status of eos tokens
    for _ in range(ncluster_max-1):
        #Feeding previous decoder output as input
        out_decoder = model.decoder(clusters_transfo, memory, tgt_key_padding_mask, src_padding_mask)
        #Computing the logits, only considering the last row. 
        logits_charges_batch = model.lastlin_charges(out_decoder)[:,-1] #shape is [N,T,F]
        logits_pdg_batch = model.lastlin_pdg(out_decoder)[:,-1]
        next_DOF_cont_batch = model.lastlin_cont(out_decoder)[:,-1]

        #no need to apply softmax since it's a bijection, and no need to print probabilities
        next_charges_batch = torch.argmax(logits_charges_batch) #class id same as index by construction
        next_pdgs_batch = torch.argmax(logits_pdg_batch) #class id same as index by construction

        #Addding special tokens of cont. DOF, ignoring the values if event has already <eos>
        new_eos_tokens_batch = ( (next_charges_batch == vocab_charges.get_index(special_symbols["eos"]["CEL"])) * ~is_done_prev)
        new_bos_tokens_batch = ( (next_charges_batch == vocab_charges.get_index(special_symbols["bos"]["CEL"])) * ~is_done_prev)
        new_sample_tokens_batch = ~new_eos_tokens_batch * ~new_bos_tokens_batch

        #Using spherical coordinates to get 3D direction vectors
        next_DOF_cont_batch_sin_theta = torch.sin(next_DOF_cont_batch[...,0])
        next_DOF_cont_batch[...,0] = torch.cos(next_DOF_cont_batch[...,1]) * next_DOF_cont_batch_sin_theta
        next_DOF_cont_batch[...,1] = torch.sin(next_DOF_cont_batch[...,1]) * next_DOF_cont_batch_sin_theta
        next_DOF_cont_batch = torch.concat([next_DOF_cont_batch, torch.cos(next_DOF_cont_batch[...,0]).unsqueeze_(-1)], dim = -1)

        if torch.count_nonzero(new_eos_tokens_batch) > 0:
            is_done[is_done_prev == False] = new_eos_tokens_batch[is_done_prev == False]
            next_DOF_cont_batch[new_eos_tokens_batch] = torch.cat(( next_DOF_cont_batch[new_eos_tokens_batch], 
                                                                special_symbols["eos"]["cont"] ), 
                                                                dim = -1)
        #shouldn't be predicted but just in case
        if torch.count_nonzero(new_bos_tokens_batch) > 0:
            next_DOF_cont_batch[new_bos_tokens_batch] = torch.cat(( next_DOF_cont_batch[new_bos_tokens_batch], 
                                                                special_symbols["bos"]["cont"] ), 
                                                                dim = -1)
        if torch.count_nonzero(new_sample_tokens_batch) > 0:
            next_DOF_cont_batch[new_sample_tokens_batch] = torch.cat(( next_DOF_cont_batch[new_sample_tokens_batch],
                                                                    special_symbols["sample"] ), 
                                                                    dim = -1)
        
        next_clusters_batch = torch.cat((next_charges_batch.unsqueeze(-1), 
                                         next_pdgs_batch.unsqueeze(-1),
                                         next_DOF_cont_batch), dim = -1) 
        #Need to change the next cluster of every event which was done previously to a pad
        pad  = torch.tensor([0]*nfeats_labels + special_symbols["pad"]["cont"])
        pad.put_([0,1], special_symbols["pad"]["CEL"])
        next_charges_batch[is_done_prev] = pad
        #Updating the tgt_padding_mask
        tgt_key_padding_mask = torch.hstack([src_padding_mask, is_done_prev.unsqueeze(-1)])

        clusters_transfo = torch.concat([clusters_transfo, next_clusters_batch.unsqueeze(1)], dim = 1)
        is_done_prev = torch.clone(is_done)

        if torch.all(is_done):
            break

    return clusters_transfo

def inference(opts):
    
    vocab_charges = Vocab.from_dict(torch.load(opts.path_charges))
    vocab_pdgs = Vocab.from_dict(torch.load(opts.path_PDGs))

    #Creating model
    model = ClustersFinder(
        dmodel = config["dmodel"],
        nhead = config["nhead"],
        nhid_ff_trsf = config["nhid_ff_trsf"],
        nlayers_encoder= config["nlayers_encoder"],
        nlayers_decoder= config["nlayers_decoder"],
        nlayers_embder = config["nlayers_embder"],
        d_input_encoder = config["d_input_encoder"],
        d_input_decoder = config["d_input_decoder"],
        nparticles_max= len(vocab_pdgs),
        ncharges_max= len(vocab_charges),
        DOF_continous= config["output_DOF_continuous"],
        device = DEVICE
    ).to(DEVICE)
    #Loading weights
    model.load_state_dict(torch.load(opts.model_path))
    model.eval()
    
    special_symbols, src_loader = get_data(opts.data_path_inference,opts.data_path_inference, opts.batch_size, "inference")

    
    for src in src_loader:
        src_padding_mask, _ = create_mask(src,torch.tensor([0]), special_symbols["pad"]["cont"],device)
        clusters_out = greedy_func(model, src,src_padding_mask,opts.ncluster_max,special_symbols,opts.nfeats_labels)
        clusters_out[...,0] = vocab_charges.indices_to_tokens(clusters_out[...,0])
        clusters_out[...,1] = vocab_pdgs.indices_to_tokens(clusters_out[...,1]) 



#train the model for one epoch
#src: input hits from simulation, tgt: MC truth clusters 
def train_epoch(model, optim, train_dl, special_symbols,vocab_charges, vocab_pdgs, 
          hyperweights_lossfn, loss_fn_charges, loss_fn_pdg,loss_fn_cont):
    model.train() #setting model into train mode
    loss_epoch = 0.0
    for src,tgt in train_dl:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        print("src device abefore call")
        print(src.device)
        src_padding_mask, tgt_padding_mask = create_mask(src,tgt,special_symbols["pad"]["cont"], DEVICE)
        tgt_in_padding_mask = tgt_padding_mask[:,:-1]
        tgt_out_padding_mask = tgt_padding_mask[:,1:]

        tgt_in = tgt[:,:-1] #sets the dimensions of transformer output -> must have the same as tgt_out
        logits_charges, logits_pdg, logits_cont = model(src,tgt_in, 
                                                                    src_padding_mask,
                                                                    tgt_in_padding_mask,
                                                                    src_padding_mask)
        optim.zero_grad()

        tgt_out = tgt[:,1:,:] #logits are compared with tokens shifted
        tgt_out_charges = tgt_out[...,0]
        tgt_out_pdg = tgt_out[...,1]
        tgt_out_cont = tgt_out[...,2:-2] #only (E, theta, phi)
        #Using spherical coordinates to get 3D direction vectors
        tgt_out_sin_theta = torch.sin(tgt_out_cont[...,1])
        tgt_out_cont[...,1] = torch.cos(tgt_out_cont[...,2]) * tgt_out_sin_theta
        tgt_out_cont[...,2] = torch.sin(tgt_out_cont[...,2]) * tgt_out_sin_theta
        tgt_out_cont = torch.concat([tgt_out_cont, torch.cos(tgt_out_cont[...,1]).unsqueeze_(-1)], dim = -1)
        #setting values of special tokens to 0 so that it doesn't contribute to loss
        tgt_out_cont[tgt_out_padding_mask] = 0.
        logits_cont[tgt_out_padding_mask] = 0.
        #special_tokens are not taken into account in the continuous loss
        eos_bos_mask = ((tgt_out_charges == vocab_charges.get_index(special_symbols["eos"]["CEL"])) 
                        + (tgt_out_charges == vocab_charges.get_index(special_symbols["bos"]["CEL"]))) 
        spe_tokens_mask = eos_bos_mask + tgt_out_padding_mask
        logits_cont[spe_tokens_mask] = 0.
        
        #Computing the losses
        logging.info("Computing the losses")
        loss_charges = loss_fn_charges(logits_charges.transpose(dim0 = -2, dim1 = -1), tgt_out_charges)
        loss_pdg = loss_fn_pdg(logits_pdg.transpose(dim0 = -2, dim1 = -1), tgt_out_pdg)
        loss_cont_vec = loss_fn_cont(logits_cont, tgt_out_cont, reduction = 'none')
        nspe_tokens = torch.count_nonzero(spe_tokens_mask, dim = -1)
        n_nospe = spe_tokens_mask.shape[-1] - nspe_tokens
        loss_cont = torch.mean(loss_cont_vec * n_nospe)

        loss_vec = torch.tensor([loss_charges,loss_pdg, loss_cont])
        loss = torch.dot(torch.tensor(hyperweights_lossfn),loss_vec)
        logging.info("Backward propagation...")
        loss.backward()
        optim.step()

        loss_epoch += loss.item()

    return loss_epoch / len(list(train_dl))

def validate_epoch(model, val_dl, special_symbols,vocab_charges, vocab_pdgs, 
                   hyperweights_lossfn, loss_fn_charges, loss_fn_pdg,loss_fn_cont):  
    model.eval() #setting model into train mode
    loss_epoch = 0.0
    for src,tgt in val_dl:
        src.to(DEVICE)
        tgt.to(DEVICE)

        src_padding_mask, tgt_padding_mask = create_mask(src,tgt,special_symbols["pad"]["cont"])
        tgt_in_padding_mask = tgt_padding_mask[:,:-1]
        tgt_out_padding_mask = tgt_padding_mask[:,1:]

        tgt_in = tgt[:,:-1] #sets the dimensions of transformer output -> must have the same as tgt_out
        logits_charges, logits_pdg, logits_cont = model(src,tgt_in, 
                                                                    src_padding_mask,
                                                                    tgt_in_padding_mask,
                                                                    src_padding_mask)

        tgt_out = tgt[:,1:,:] #logits are compared with tokens shifted
        tgt_out_charges = tgt_out[...,0]
        tgt_out_pdg = tgt_out[...,1]
        tgt_out_cont = tgt_out[...,2:-2] #only (E, theta, phi)
        #Using spherical coordinates to get 3D direction vectors
        tgt_out_sin_theta = torch.sin(tgt_out_cont[...,1])
        tgt_out_cont[...,1] = torch.cos(tgt_out_cont[...,2]) * tgt_out_sin_theta
        tgt_out_cont[...,2] = torch.sin(tgt_out_cont[...,2]) * tgt_out_sin_theta
        tgt_out_cont = torch.concat([tgt_out_cont, torch.cos(tgt_out_cont[...,1]).unsqueeze_(-1)], dim = -1)
        #setting values of special tokens to 0 so that it doesn't contribute to loss
        tgt_out_cont[tgt_out_padding_mask] = 0.
        logits_cont[tgt_out_padding_mask] = 0.
        #special_tokens are not taken into account in the continuous loss
        eos_bos_mask = ((tgt_out_charges == vocab_charges.get_index(special_symbols["eos"]["CEL"])) 
                        + (tgt_out_charges == vocab_charges.get_index(special_symbols["bos"]["CEL"]))) 
        spe_tokens_mask = eos_bos_mask + tgt_out_padding_mask
        logits_cont[spe_tokens_mask] = 0.
        
        #Computing the losses
        loss_charges = loss_fn_charges(logits_charges.transpose(dim0 = -2, dim1 = -1), tgt_out_charges)
        loss_pdg = loss_fn_pdg(logits_pdg.transpose(dim0 = -2, dim1 = -1), tgt_out_pdg)
        loss_cont_vec = loss_fn_cont(logits_cont, tgt_out_cont, reduction = 'none')
        nspe_tokens = torch.count_nonzero(spe_tokens_mask, dim = -1)
        n_nospe = spe_tokens_mask.shape[-1] - nspe_tokens
        loss_cont = torch.mean(loss_cont_vec * n_nospe)

        loss_vec = torch.tensor([loss_charges,loss_pdg, loss_cont])
        loss = torch.dot(torch.tensor(hyperweights_lossfn),loss_vec)

        loss_epoch += loss.item()

    return loss_epoch / len(list(val_dl))

def train_and_validate(config):

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename= config["dir_results"] + "log.txt", level= logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info("Getting the training data from" +config["dir_path_train"])
    vocab_charges, vocab_pdgs, special_symbols, _, train_dl, val_dl = get_data(config["dir_path_train"], config["dir_path_val"], config["batch_size"], "training")
    torch.save(vocab_charges.vocab, config["dir_results"] + "vocab_charges.pt")
    torch.save(vocab_pdgs.vocab, config["dir_results"] + "vocab_PDGs.pt")
    
    logging.info("Loaded the data and saved vocabularies")
    #Creating model
    model = ClustersFinder(
        dmodel = config["dmodel"],
        nhead = config["nhead"],
        nhid_ff_trsf = config["nhid_ff_trsf"],
        nlayers_encoder= config["nlayers_encoder"],
        nlayers_decoder= config["nlayers_decoder"],
        nlayers_embder = config["nlayers_embder"],
        d_input_encoder = config["d_input_encoder"],
        d_input_decoder = config["d_input_decoder"],
        nparticles_max= len(vocab_pdgs),
        ncharges_max= len(vocab_charges),
        DOF_continous= config["output_DOF_continuous"],
        device = DEVICE
    ).to(DEVICE)


    
    
    logging.info("Created model, now training")
    print(f"DEVICE: {DEVICE}") 
    optim = torch.optim.Adam(model.parameters(), lr = config["lr"])
    loss_fn_charges = nn.CrossEntropyLoss(ignore_index=vocab_charges.get_index(special_symbols["pad"]["CEL"]), reduction ='mean')
    loss_fn_pdgs = nn.CrossEntropyLoss(ignore_index=vocab_pdgs.get_index(special_symbols["pad"]["CEL"]), reduction ='mean')
    loss_fn_cont = nn.MSELoss()

    val_loss_min = 1e9
    nepoch = config["epochs"]
    losses_evolution = {"train": torch.zeros(nepoch),
                        "val": torch.zeros(nepoch),
                        "time":  torch.zeros(nepoch)}
    for i in range(nepoch):
        start_time = time()
        train_loss_epoch = train_epoch(model,
                                       optim = optim,
                                       train_dl=train_dl,
                                       hyperweights_lossfn= config["hyper_loss"],
                                       special_symbols = special_symbols,
                                       vocab_charges = vocab_charges,
                                       vocab_pdgs= vocab_pdgs,
                                       loss_fn_charges = loss_fn_charges,
                                       loss_fn_pdg= loss_fn_pdgs,
                                       loss_fn_cont= loss_fn_cont)
        time_epoch = start_time - time() 
        val_loss_epoch = validate_epoch(model,
                                       val_dl=val_dl,
                                       hyperweights_lossfn= config["hyper_loss"],
                                       special_symbols = special_symbols,
                                       vocab_charges = vocab_charges,
                                       vocab_pdgs= vocab_pdgs,
                                       loss_fn_charges = loss_fn_charges,
                                       loss_fn_pdg= loss_fn_pdgs,
                                       loss_fn_cont= loss_fn_cont) 

        losses_evolution["train"][i] = train_loss_epoch
        losses_evolution["val"][i] = val_loss_epoch
        losses_evolution["time"][i] = time_epoch

        if val_loss_epoch < val_loss_min:
            logging.info("New best Model, saving...")
            torch.save(model.state_dict(), config["dir_results"] + "best_model.pt")
            val_loss_min = val_loss_min

        logging.info(f"{i + 1} epoch done, time: {time_epoch}, val_loss: {val_loss_epoch}, train_loss: {train_loss_epoch}")
    
    torch.save(losses_evolution, "losses.pt")

if __name__ == "__main__":

    parser = ArgumentParser(prog= "Finding clusters training and inference ")
    #Inference
    parser.add_argument("--inference", action= "store_true", help = "Set true to run inference")
    parser.add_argument("--config_path", type = str, help = "Directory of ConfigFile")

    args = parser.parse_args()

    with open(args.config_path + "ConfigFile.json") as f:
        config = json.load(f)

    if args.inference:
        inference(config)
    else:
        train_and_validate(config)

        
        
