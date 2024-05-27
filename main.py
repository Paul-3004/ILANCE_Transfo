import torch
import torch.nn as nn
import torch.optim as optim

from data import create_mask, get_data, Vocab
from model import ClustersFinder
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def greedy_func(model,src,src_padding_mask,ncluster_max: int, special_symbols: dict, nfeats_labels: int = 6):
    src = src.to(DEVICE)
    src_padding_mask = src_padding_mask.to(DEVICE)

    #feeding source to encoder
    memory = model.encode(src,src_padding_mask).to(DEVICE)
    #Creating <bos> token
    bos = torch.tensor([0]*nfeats_labels + special_symbols["bos"]["cont"])
    bos.put_([0,1], special_symbols["bos"]["CEL"])

    batch_size = src.size[0]
    clusters_transfo = torch.ones(batch_size,1).fill_(bos).to(DEVICE) #output of decoder, updated to input decoder
    tgt_key_padding_mask = torch.zeros_like(clusters_transfo).type(torch.bool).to(DEVICE)
    is_done = torch.zeros(batch_size,1).type(torch.bool) #to keep track of which event has eos token
    is_done_prev = is_done #to keep track of previous status of eos tokens
    for _ in range(ncluster_max-1):
        out_decoder = model.decoder(clusters_transfo, memory, src_padding_mask) #Feeding previous decoder output as input
        #Computing the logits, only considering the last row. 
        logits_charges_batch = model.lastlin_charges(out_decoder)[:,-1] #shape is [N,S,F]
        logits_pdg_batch = model.lastlin_pdg(out_decoder)[:,-1]
        next_DOF_cont_batch = model.lastlin_cont(out_decoder)[:,-1]

        #no need to apply softmax since it's a bijection, and no need to print probabilities
        next_charges_batch = torch.argmax(logits_charges_batch) #class id same as index by construction
        next_pdgs_batch = torch.argmax(logits_pdg_batch) #class id same as index by construction

        #Addding special tokens of cont. DOF, ignoring the values if event has already <eos>
        eos_tokens_batch = (next_charges_batch == special_symbols["eos"]["CEL"])[is_done == False]
        bos_tokens_batch = (next_charges_batch == special_symbols["bos"]["CEL"])[is_done == False]
        sample_tokens_batch = ~eos_tokens_batch * ~bos_tokens_batch
        if torch.count_nonzero(eos_tokens_batch) > 0:
            is_done[is_done == False] = eos_tokens_batch
            next_DOF_cont_batch[eos_tokens_batch] = torch.cat(( next_DOF_cont_batch[eos_tokens_batch], 
                                                                special_symbols["eos"]["cont"] ), 
                                                               dim = -1)
        #shouldn't be predicted but just in case
        if torch.count_nonzero(bos_tokens_batch) > 0:
            next_DOF_cont_batch[bos_tokens_batch] = torch.cat(( next_DOF_cont_batch[bos_tokens_batch], 
                                                                special_symbols["bos"]["cont"] ), 
                                                                dim = -1)
        if torch.count_nonzero(sample_tokens_batch) > 0:
            next_DOF_cont_batch[sample_tokens_batch] = torch.cat(( next_DOF_cont_batch[sample_tokens_batch],
                                                                    special_symbols["sample"] ), 
                                                                    dim = -1)
        
        next_clusters_batch = torch.cat((next_charges_batch.unsqueeze(-1), 
                                         next_pdgs_batch.unsqueeze(-1),
                                         next_DOF_cont_batch), dim = -1) 
        pad  = torch.tensor([0]*nfeats_labels + special_symbols["pad"]["cont"])
        pad.put_([0,1], special_symbols["pad"]["CEL"])
        if torch.all(~is_done_prev) is False:
            next_charges_batch[is_done_prev] = pad

        src_padding_mask = torch.hstack([src_padding_mask, is_done_prev.unsqueeze(-1)])
        clusters_transfo = torch.concat([clusters_transfo, next_clusters_batch.unsqueeze(1)], dim = 1)

        if torch.all(is_done):
            break

    return clusters_transfo

def inference(opts, vocab_charges, vocab_pdgs):

    #Creating model
    model = ClustersFinder(
        dmodel = opts.dmodel,
        nhead = opts.nhead,
        nhid_ff_trsf = opts.nhid_ff_trsf,
        nlayers_encoder= opts.nlayers_encoder,
        nlayers_decoder= opts.nlayers_decoder,
        nlayers_embder = opts.nlayers_embder,
        d_input = opts.d_input,
        nparticles_max= opts.nparticles_max,
        ncharges_max= opts.ncharges_max,
        DOF_continous= opts.DOF_continous
    )
    #Loading weights
    model.load_state_dict(torch.load(opts.model_path))
    model.eval()

    special_symbols = opts.special_symbols
    src, _= get_data(opts.data_path, opts.batch_size, special_symbols)
    batch_size = src.size[0]
    nhits = src.size[1]
    src_padding_mask = torch.zeros((batch_size, nhits)).type(torch.bool)
    clusters_out = greedy_func(model, src,src_padding_mask,opts.ncluster_max,special_symbols,opts.nfeats_labels)
    clusters_out[...,0] = vocab_charges.indices_to_tokens(clusters_out[...,0])
    clusters_out[...,1] = vocab_pdgs.indices_to_tokens(clusters_out[...,1]) 



#train the model for one epoch
#src: input hits from simulation, tgt: MC truth clusters 
def train(model, optim, train_dl, special_symbols,vocab_charges, vocab_pdgs, 
          hyperweights_lossfn, loss_fn_charges, loss_fn_pdg,loss_fn_cont):
    model.train() #setting model into train mode
    loss_epoch = 0.0
    for src,tgt in train_dl:
        src.to(DEVICE)
        tgt.to(DEVICE)

        src_padding_mask, tgt_padding_mask = create_mask(src,tgt,special_symbols["pad"]["cont"])
        tgt_in_padding_mask = tgt_padding_mask[:,:-1,:]
        tgt_out_padding_mask = tgt_padding_mask[:,1:,:]

        tgt_in = tgt[:,:-1] #sets the dimensions of transformer output -> must have the same as tgt_out
        logits_charges, logits_pdg, logits_cont = model(src,tgt_in, 
                                                                    src_padding_mask,
                                                                    tgt_in_padding_mask,
                                                                    src_padding_mask)
        optim.zero_grad()

        tgt_out = tgt[:,1:,:] #logits are compared with tokens shifted
        tgt_out_charges = tgt_out[...,0]
        tgt_out_pdg = tgt_out[...,1]
        tgt_out_cont = tgt_out[...,2:] #to the end or only 2:5 ? 
        #setting values of padding tokens to 0 so that it doesn't contribute to loss
        tgt_out_cont[tgt_out_padding_mask] = 0.0
        logits_cont[tgt_out_padding_mask] = 0.0
        #Computing the losses
        loss_charges = loss_fn_charges(logits_charges.transpose(dim0 = -2, dim1 = -1), tgt_out_charges)
        loss_pdg = loss_fn_pdg(logits_pdg.transpose(dim0 = -2, dim1 = -1), tgt_out_pdg)
        loss_cont_vec = loss_fn_cont(logits_cont, tgt_out_cont, reduction = 'none')
        npad = torch.count_nonzero(tgt_out_padding_mask, dim = -1)
        n_nopad = tgt_out_padding_mask.shape[-1] - npad
        loss_cont = torch.mean(loss_cont_vec * n_nopad)

        loss_vec = torch.tensor([loss_charges,loss_pdg, loss_cont])
        loss = torch.dot(torch.tensor(hyperweights_lossfn),loss_vec)

        loss.backward()
        optim.step()

        loss_epoch += loss.item()

    return loss_epoch / len(list(train_dl))

def validate(model, optim, val_dl, special_symbols,vocab_charges, vocab_pdgs, 
          hyperweights_lossfn, loss_fn_charges, loss_fn_pdg,loss_fn_cont):
    
    model.eval() #setting model into validation mode
    loss_epoch = 0.0
    for src,tgt in val_dl:
        src.to(DEVICE)
        tgt.to(DEVICE)

        src_padding_mask, tgt_padding_mask = create_mask(src,tgt,special_symbols["pad"]["cont"])
        tgt_in_padding_mask = tgt_padding_mask[:,:-1,:]
        tgt_out_padding_mask = tgt_padding_mask[:,1:,:]

        tgt_in = tgt[:,:-1] #sets the dimensions of transformer output -> must have the same as tgt_out
        logits_charges, logits_pdg, logits_cont = model(src,tgt_in, 
                                                                    src_padding_mask,
                                                                    tgt_in_padding_mask,
                                                                    src_padding_mask)
        optim.zero_grad()

        tgt_out = tgt[:,1:,:] #logits are compared with tokens shifted
        tgt_out_charges = tgt_out[...,0]
        tgt_out_pdg = tgt_out[...,1]
        tgt_out_cont = tgt_out[...,2:] #to the end or only 2:5 ? 
        #setting values of padding tokens to 0 so that it doesn't contribute to loss
        tgt_out_cont[tgt_out_padding_mask] = 0.0
        logits_cont[tgt_out_padding_mask] = 0.0
        #Computing the losses
        loss_charges = loss_fn_charges(logits_charges.transpose(dim0 = -2, dim1 = -1), tgt_out_charges)
        loss_pdg = loss_fn_pdg(logits_pdg.transpose(dim0 = -2, dim1 = -1), tgt_out_pdg)
        loss_cont_vec = loss_fn_cont(logits_cont, tgt_out_cont, reduction = 'none')
        npad = torch.count_nonzero(tgt_out_padding_mask, dim = -1)
        n_nopad = tgt_out_padding_mask.shape[-1] - npad
        loss_cont = torch.mean(loss_cont_vec * n_nopad)

        loss_vec = torch.tensor([loss_charges,loss_pdg, loss_cont])
        loss = torch.dot(torch.tensor(hyperweights_lossfn),loss_vec)

        loss.backward()
        optim.step()

        loss_epoch += loss.item()

    return loss_epoch / len(list(val_dl))
