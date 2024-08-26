from matplotlib import pyplot as plt
import numpy as np
import torch
import awkward as ak
from argparse import ArgumentParser
from matplotlib import rc
import os
from time import time
import seaborn as sns
#from ..load_awkward import save_awkward

rc('text', usetex = True)
rc("axes", labelsize = 25)
rc("xtick", labelsize = 22)
rc("ytick", labelsize = 22)
rc("axes", titlesize = 25)
rc("legend", fontsize = 25)
plt.rcParams['figure.constrained_layout.use'] = True

def make_histogram(data, nbins, xlabel, r = None,colour = None):
    fig, ax = plt.subplots()
    if colour is not None:
        if r is not None:
            ax.hist(data,nbins, c = colour, range = r)
        else:
            ax.hist(data,nbins, c = colour)
    else:
        if r is not None:
            ax.hist(data,nbins, range = r)
        else:
            ax.hist(data,nbins)

    ax.set(xlabel = xlabel)
    counts, bins = np.histogram(data,nbins, range = r)
    indice_peak = np.argmax(counts)
    value_peak = bins[indice_peak] + (bins[indice_peak] + bins[indice_peak+1])/2
    return fig

def cartesian_to_angles(input):
    theta = np.arccos(input[...,-3])
    phi = np.arctan2(input[...,-4], input[...,-5])
    return theta, phi

# def sample_tokens_mask(input):
#     special_symbols = {
#         "pad": {"cont": [0.,1.],"CEL":-150},
#         "bos": {"cont": [1.,1.], "CEL":-100},
#         "eos": {"cont": [1.,0.],"CEL":-50},
#         "sample": [0.,0.]
#     }
#     mask = torch.all(input[...,-2:] == torch.tensor(special_symbols["sample"]).unsqueeze_(0).unsqueeze_(0), dim = -1)
#     nsamples = torch.count_nonzero(mask, dim = -1).tolist()
#     return mask, nsamples
def sample_tokens_mask(input):
    special_symbols = {
            "pad": 0,
            "bos": 1,
            "eos": 2,
            "sample": 3
    }
    mask = input[...,-1] == special_symbols["sample"]
    nsamples = torch.count_nonzero(mask, dim = -1).tolist()
    return mask, nsamples

def samples_to_awkward(input):
    mask, nsamples = sample_tokens_mask(input)
    input_ak = ak.unflatten(input[mask].numpy(), nsamples)
    return input_ak

def load_pred_labels(dir_path):
    pred = samples_to_awkward(torch.load(os.path.join(dir_path, "prediction.pt")))
    labels = samples_to_awkward(torch.load(os.path.join(dir_path, "labels.pt")))
    return pred, labels
    
def unpack(input):
    return input[...,0], input[...,1], input[...,2], input[...,3:-1]

def reduce(input, reduction, abs_val:bool):
    if abs_val:
        mean_event = ak.mean(abs(input), axis = -1)
    else:
        mean_event = ak.mean(input, axis = -1)
    if reduction is None:
        return input
    elif reduction  == "event":
        return mean_event
    elif reduction == "full":
        #return ak.mean(mean_event)
        return ak.mean(input)
    else: 
        raise ValueError(f"reduction must be either None type, event or full. {reduction} is invalid")

def create_mask(len_input, len_overlap):
    len_diff = len_input - len_overlap
    mask_overlap = ak.unflatten(np.ones(ak.sum(len_overlap), dtype = bool), len_overlap) #overlap
    mask_excess = ak.unflatten(np.zeros(ak.sum(len_diff),dtype= bool), len_diff) #excess
    return ak.concatenate([mask_overlap, mask_excess], axis = -1)

def getOverlap(pred, labels):
    nhits_pred = ak.num(pred, axis = 1)
    nhits_labels = ak.num(labels, axis = 1)
    #comparing number of hits, length of array with less number of hits is selected
    len_comparison = ak.where((nhits_pred-nhits_labels) < 0, nhits_pred, nhits_labels)
    len_labels_mask = ak.where((len_comparison - nhits_labels) < 0 , len_comparison, nhits_labels)
    #number of excess prediction
    mask_overlap_pred = create_mask(nhits_pred, len_comparison)
    mask_overlap_label = create_mask(nhits_labels, len_labels_mask) 
    return mask_overlap_pred, mask_overlap_label

def accuracy_discrete(pred, labels, reduction):
    overlap_pred, overlap_labels = getOverlap(pred, labels)
    pred = pred[overlap_pred]
    labels = labels[overlap_labels]
    correct = pred == labels
    return reduce(correct, reduction, True)

def accuracy_energy(pred, labels, reduction: bool):
    overlap_pred, overlap_labels = getOverlap(pred, labels)
    pred = pred[overlap_pred]
    labels = labels[overlap_labels]
    dim = ak.num(pred,axis = 1)
    pred = ak.flatten(pred)
    labels = ak.flatten(labels)
    delta_E = np.abs(pred - labels)
    div = ak.where(abs(labels) > 1e-15, labels, ak.ones_like(labels))
    error = ak.unflatten(delta_E / div, dim)
    return reduce(error, reduction, True)

def getEnergy_tot(pred, labels, reduction):
    mask_overlap_pred, mask_overlap_labels= getOverlap(pred, labels)
    E_event_pred_overlap = ak.sum(pred[mask_overlap_pred], axis = -1) 
    E_event_pred_excess = ak.sum(pred[~mask_overlap_pred], axis = -1)
    E_event_labels_overlap = ak.sum(labels[mask_overlap_labels], axis = -1) 
    E_event_labels_excess = ak.sum(labels[~mask_overlap_labels], axis = -1)
    E_event_labels = ak.sum(labels, axis = -1)
    reduction_excess = None
    if reduction == "full":
        reduction_excess = "event"
    return (accuracy_energy(pred, labels, reduction), 
            reduce(E_event_pred_excess, reduction_excess, True),
            E_event_labels)

def accuracy_momentum(pred, labels, reduction):
    mask_overlap_pred, mask_overlap_label = getOverlap(pred, labels)
    pred = pred[mask_overlap_pred]
    labels = labels[mask_overlap_label]
    dim = ak.num(pred, axis = 1)
    scalar_prod_np = ak.sum(ak.flatten(pred) * ak.flatten(labels), axis = -1).to_numpy()
    mask_min = scalar_prod_np < -1
    mask_max = scalar_prod_np > 1
    scalar_prod_np[mask_min] = -1
    scalar_prod_np[mask_max] = 1
    scalar_prod = ak.unflatten(ak.from_numpy(scalar_prod_np), dim)
    mask = np.abs(scalar_prod_np) > 1
    print(f"scalar prod exceeds one from {ak.sum(np.abs(1 - scalar_prod[mask]))}, replacing by 1")
    #mask_min = scalar_prod < -1
    #mask_max = scalar_prod > 1
    #scalar_prod[mask_min] = -1
    #scalar_prod[mask_max] = 1
    #print(f"{test:.16f}",)

    #Converting to angles 
    angles = np.arccos(scalar_prod) * (180/np.pi)
    print(f"RMS values of angle distribution {ak.std(angles, axis = None)}")
    return reduce(angles, reduction, False)

def excess_prediction(pred, labels, reduction):
    nhits_pred = ak.num(pred, axis = 1)
    nhits_labels = ak.num(labels, axis = 1)
    #print(ak.num(nhits_pred,0))
    #print(ak.num(nhits_labels,0))
    diff = nhits_pred - nhits_labels
    #mask = np.abs(diff) > 1e-15
    #non_zero = diff[mask]
    #non_zero.show()
    #print(ak.num(non_zero,0))
    if reduction is None:
        return diff
    elif reduction == "full":
        return ak.mean(diff)
    else:
        raise ValueError(f"{reduction} is invalid for excess_prediction, choose either None or full")

def compare_excess_pred(preds, labels, plot_labels, nbins = 40, order =None):
    fig, ax = plt.subplots()
    if isinstance(nbins, int):
        nbins = [nbins] * len(preds)
    if order is not None:
        nbins = [nbins[index] for index in order]
        preds = [preds[index] for index in order]
        plot_labels = [plot_labels[index] for index in order]
    for i, pred in enumerate(preds):
        excess = excess_prediction(pred, labels, reduction= None)
        ax.hist(excess,nbins[i], label = plot_labels[i])
    ax.set(xlabel = r"$\textrm{Excess predictions}$")
    ax.legend(loc = "upper right")

def accuracy(pred, labels, mode: str, sort: bool, red_E, red_C, red_PDGs, red_p, red_excess, nevents = None, indices_event = None):
    if sort:
        index = ak.argsort(pred[...,2])
        pred = pred[index]
    if mode == "random":
        if nevents is None:
            raise ValueError("Please specify the number of events")
        indices_event = torch.randint(low = 0, high = len(pred),size = nevents).tolist()
        pred = pred[indices_event]
    elif mode == "spe":
        if indices_event is None:
            raise ValueError("Please specify the specific events")
        if isinstance(indices_event,int):
            indices_event = [indices_event]
            pred = pred[indices_event]
        
    #needed_sorting = torch.zeros(len(indices_event)).type(torch.bool)
    charges_pred, pdgs_pred, E_pred, p_pred = unpack(pred)
    charges_labels, pdgs_labels, E_labels, p_labels = unpack(labels)

    charges_acc = accuracy_discrete(charges_pred, charges_labels, red_C)
    pdgs_acc = accuracy_discrete(pdgs_pred, pdgs_labels, red_PDGs)
    E_acc, E_excess, E_tot_label = getEnergy_tot(E_pred, E_labels, red_E)
    p_acc = accuracy_momentum(p_pred,p_labels, red_p)
    excess_pred = excess_prediction(pred, labels, red_excess)

    return (charges_acc,
            pdgs_acc,
            E_acc,
            p_acc, 
            excess_pred)

def energy_distribution(input,type_input: str, nbins = 50, c = None):
    E_xlabel = ""
    if type_input == "pred":
        E_xlabel = r"$E_p \textrm{ [GeV]}$"
    else:
        E_xlabel = r"$E_l \textrm{ [GeV]}$"
    
    make_histogram(input[...,2],nbins, E_xlabel,colour = c)

def angles_distribution(input,type_input: str, nbins_theta = 50, nbins_phi = 50, c = None):
    theta, phi = cartesian_to_angles(input)
    theta_xlabel = ""
    phi_xlabel = ""
    if type_input == "pred":
        theta_xlabel = r"$\theta_p \, [^\circ]$"
        phi_xlabel = r"$\phi_p \, [^\circ]$"
    else:
        theta_xlabel = r"$\theta_l \, [^\circ]$"
        phi_xlabel = r"$\phi_l \, [^\circ]$"       
     
    make_histogram(theta, nbins_theta, theta_xlabel)
    make_histogram(phi, nbins_theta, phi_xlabel)

def make_energy_angles_distributions(pred, labels, n):
    print(ak.num(pred,axis = 0))
    #indices= torch.randint(int(ak.num(pred, axis = 0)), (n,)).tolist()
    events_pred = pred
    events_labels = labels
    #for i in range(len(indices)):
    #    angles_distribution(events_pred[i], "pred")
    #    angles_distribution(events_labels[i], None)
    #    energy_distribution(events_pred[i], "pred")
    #    energy_distribution(events_labels[i], None)
    energy_distribution(events_labels, None)
    energy_distribution(events_pred, "pred")
    nbins = 50
    make_histogram(np.log10(events_labels[...,2]) - np.log10(events_pred[...,2]),50, r"$\log_{10}E_l - \log_{10}E_p$")
    fig, ax = plt.subplots()
    log10_l = np.log10(events_labels[...,2]).to_numpy().reshape(1000)
    log10_p =  np.log10(events_pred[...,2]).to_numpy().reshape(1000)
    
    ax.hist2d(log10_l, log10_l-log10_p, [50,50])

def energy_distrib_diff_log10(pred,labels,n, name):
    #indices= torch.randint(int(ak.num(pred, axis = 0)), (n,)).tolist()
    events_pred = ak.concatenate(pred)
    events_labels = ak.concatenate(labels)
    events_pred = ak.concatenate(pred[...,2])
    events_labels = ak.concatenate(labels[...,2])
    events_labels.show()
    events_pred.show()
    fig = make_histogram(np.log10(events_labels) - np.log10(events_pred),50, r"$\log E_l - \log E_p$", (-0.2,0.2))
    fig.savefig(name, format = "svg", transparent = True)
def energy_distrib_diff(pred,labels,n,name):
    events_pred = ak.concatenate(pred)
    events_labels = ak.concatenate(labels)
    events_pred = ak.concatenate(pred[...,2])
    events_labels = ak.concatenate(labels[...,2])
    #events_labels.show()
    #events_pred.show()
    fig = make_histogram((events_labels - events_pred)/events_labels,50, r"$\frac{E_l - E_p}{E_l}$", (-0.5,0.5))
    print(f"RMS {name}: {ak.std((events_labels - events_pred)/events_labels, axis = None)}")
    print(f"Mean {name}: {ak.mean((events_labels - events_pred)/events_labels, axis = None)}")
    print(f"number of events: {ak.num(events_labels, axis = 0)}")
    counts, bins = np.histogram((events_labels - events_pred)/events_labels,50, range = (-0.5,0.5))
    indice_peak = np.argmax(counts)
    value_peak = bins[indice_peak] + (bins[indice_peak] + bins[indice_peak+1])/2
    print(f"{name}: highest peak of {np.max(counts)} at {value_peak}")
    fig.savefig(name, format = "svg", transparent = True)
    fig.suptitle(name)
def energy_zone_split(pred,labels):
    mask10_20 = labels[...,2] < 20
    mask_less50 = labels[...,2] < 50
    mask_big20 = labels[...,2] > 20
    mask20_50 = mask_less50 * mask_big20
    mask_big50 = labels[...,2] > 50
    print(ak.count_nonzero(mask_big50))
    print(ak.count_nonzero(mask20_50))
    print(ak.count_nonzero(mask10_20))
    
    #pred10_20 = ak.concatenate(pred[mask10_20])
    #pred20_50 = ak.concatenate(pred[mask20_50])
    #pred50_100 = ak.concatenate(pred[mask_big50])

    #labels10_20 = ak.concatenate(labels[mask10_20])
    #labels20_50 = ak.concatenate(labels[mask20_50])
    #labels_100 = ak.concatenate(labels[mask_big50])
    pred10_20 =pred[mask10_20]
    pred20_50 = pred[mask20_50]
    pred50_100 = pred[mask_big50]
    labels10_20 = labels[mask10_20]
    labels20_50 = labels[mask20_50]
    labels_100 = labels[mask_big50]
    #energy_distrib_diff_log10(pred10_20, labels10_20,int(ak.num(pred10_20, axis = 0)),"Diff_log_E_10_20GeV.svg")
    #energy_distrib_diff_log10(pred20_50, labels20_50,int(ak.num(pred20_50,axis = 0)),"Diff_log_E_20_50GeV.svg")
    #energy_distrib_diff_log10(pred50_100, labels_100,int(ak.num(pred50_100, axis = 0)),"Diff_log_E_50_100GeV.svg")
    energy_distrib_diff(pred10_20, labels10_20,1,"10-20GeV")
    energy_distrib_diff(pred20_50, labels20_50,1,"20-50GeV")
    energy_distrib_diff(pred50_100, labels_100,1,"50-100GeV")

def energy_zone_split5to50(pred, labels):
    mask5_15 = labels[...,2] < 15
    mask_less30 = labels[...,2] < 30
    mask_big15 = labels[...,2] > 15
    mask15_30 = mask_less30 * mask_big15
    mask_big30 = labels[...,2] > 30
    print(ak.count_nonzero(mask_big30))
    print(ak.count_nonzero(mask15_30))
    print(ak.count_nonzero(mask5_15))

    #pred10_20 = ak.concatenate(pred[mask10_20])                                                                                                                                     
    #pred20_50 = ak.concatenate(pred[mask20_50])                                                                                                                                     
    #pred50_100 = ak.concatenate(pred[mask_big50])                                                                                                                                   

    #labels10_20 = ak.concatenate(labels[mask10_20])                                                                                                                                 
    #labels20_50 = ak.concatenate(labels[mask20_50])                                                                                                                                 
    #labels_100 = ak.concatenate(labels[mask_big50])                                                                                                                                 
    pred5_15 =pred[mask5_15]
    pred15_30 = pred[mask15_30]
    pred30_50 = pred[mask_big30]
    labels5_15 = labels[mask5_15]
    labels15_30 = labels[mask15_30]
    labels_50 = labels[mask_big30]
    #energy_distrib_diff_log10(pred10_20, labels10_20,int(ak.num(pred10_20, axis = 0)),"Diff_log_E_10_20GeV.svg")                                                                    
    #energy_distrib_diff_log10(pred20_50, labels20_50,int(ak.num(pred20_50,axis = 0)),"Diff_log_E_20_50GeV.svg")                                                                     
    #energy_distrib_diff_log10(pred50_100, labels_100,int(ak.num(pred50_100, axis = 0)),"Diff_log_E_50_100GeV.svg")                                                                  
    energy_distrib_diff(pred5_15, labels5_15,1,"5-15GeV")
    energy_distrib_diff(pred15_30, labels15_30,1,"15-30GeV")
    energy_distrib_diff(pred30_50, labels_50,1,"30-50GeV")
def energy_2d_hist(pred,labels,n):
    #indices= torch.randint(int(ak.num(pred, axis = 0)), (n,)).tolist()                                                                                                              
    events_pred = pred
    events_labels = labels
    print(ak.num(events_pred, axis = 0))
    print(ak.num(events_labels, axis = 0))
    fig,ax = plt.subplots()
    #energy_distribution(events_labels, None)                                                                                                                                        
    #energy_distribution(events_pred, "pred")                                                                                                                                        
    print(ak.all(ak.num(events_pred,axis = -1) == 6))
    print(ak.all(ak.num(events_pred,axis = -1) == 6))
    E_pred = ak.flatten(events_pred[...,2]).to_numpy()
    E_labels =ak.flatten(events_labels[...,2]).to_numpy()
    print(ak.num(E_pred,axis = 0))
    print(ak.num(E_labels,axis =0))
    nbins = 50                                                            
    fig, ax = plt.subplots()
    #log10_l = np.squeeze(np.log10(ak.concatenate(events_labels[...,2])).to_numpy())                                                                               
    print(ak.count_nonzero(E_pred < 0))
    print(ak.count_nonzero(E_labels < 0))
    #log10_l = np.log10(E_labels).to_numpy()
    #log10_p = np.log10(E_pred).to_numpy()
    #print(log10_p.shape)
    #print(log10_l.shape)
    #ax.hist2d(log10_l, log10_l-log10_p, [50,50], range = [[np.min(log10_l)-0.01, np.max(log10_l)+0.01], [np.min(log10_l-log10_p)-0.01, np.max(log10_l-log10_p)+0.01]])              
    #ax.hist2d(log10_l, log10_l-log10_p, [50,50], range = [[0.99, 2.01], [-0.22, 0.22]])                                                                                             
    #ax.hist2d(E_labels, E_labels-E_pred, [50,50], range = [[0.68, 1.7], [-0.22, 0.22]])
    ax.hist2d(E_labels, E_labels-E_pred, [50,50])   
    ax.set(xlabel = r"$E_l$",ylabel = r"$E_l -E_p$")
    fig.savefig("energy_2D_hist.svg", format = "svg", transparent = True)
    
def energy_2d_histlog10(pred,labels,n):
    #indices= torch.randint(int(ak.num(pred, axis = 0)), (n,)).tolist()
    events_pred = pred
    events_labels = labels
    print(ak.num(events_pred, axis = 0))
    print(ak.num(events_labels, axis = 0))
    fig,ax = plt.subplots()
    #energy_distribution(events_labels, None)
    #energy_distribution(events_pred, "pred")
    print(ak.all(ak.num(events_pred,axis = -1) == 6))
    print(ak.all(ak.num(events_pred,axis = -1) == 6))
    E_pred = ak.flatten(events_pred[...,2])
    E_labels =ak.flatten(events_labels[...,2])
    print(ak.num(E_pred,axis = 0))
    print(ak.num(E_labels,axis =0))
    nbins = 50
    #make_histogram(np.log10(events_labels[...,2]) - np.log10(events_pred[...,2]),50, r"$\log_{10}E_l - \log_{10}E_p$")
    fig, ax = plt.subplots()
    #log10_l = np.squeeze(np.log10(ak.concatenate(events_labels[...,2])).to_numpy())
    #log10_p =  np.squeeze(np.log10(ak.concatenate(events_pred[...,2])).to_numpy())
    print(ak.count_nonzero(E_pred < 0))
    print(ak.count_nonzero(E_labels < 0))
    log10_l = np.log10(E_labels).to_numpy()
    log10_p = np.log10(E_pred).to_numpy()
    print(log10_p.shape)
    print(log10_l.shape)
    #ax.hist2d(log10_l, log10_l-log10_p, [50,50], range = [[np.min(log10_l)-0.01, np.max(log10_l)+0.01], [np.min(log10_l-log10_p)-0.01, np.max(log10_l-log10_p)+0.01]])
    ax.hist2d(log10_l, log10_l-log10_p, [50,50], range = [[0.99, 2.01], [-0.22, 0.22]])
    #ax.hist2d(log10_l, log10_l-log10_p, [50,50], range = [[0.68, 1.7], [-0.22, 0.22]])
    ax.set(xlabel = r"$\log_{10}E_l$",ylabel = r"$\log_{10}E_l -\log_{10}E_p$")
    fig.savefig("energy_2D_hist_log10.svg", format = "svg", transparent = True)

def compare_distributions(accuracies, nbins, legend,colours, savefig, dir_res):
    acc_bytypes = list(zip(*accuracies))
    labels = [r"\textrm{charges accuracy}", 
              r"\textrm{PDGs accuracy}", 
              r"$\frac{\vert E_{p} - E_{l} \vert}{E_{l}}$",
              r"$\theta_{lp} \, [^{\circ}]$", 
              r"\textrm{Excess predictions of number of clusters}"]
    ranges = [(-0.01,1.01), (-0.01,1.01), (-0.01,1.01), (-0.01,50), None]
    for i, accs_type in enumerate(acc_bytypes):
        fig, ax = plt.subplots()
        #print(accs_type)  
        #bins = np.histogram(accs_type, nbins[i])[1]
        _,bins,_ = ax.hist(accs_type[0],nbins[i], label = legend[0], color = colours[0], range = ranges[i])
        for j, acc in enumerate(accs_type[1:]):
            j += 1
            ax.hist(acc, bins, label = legend[j], color = colours[j], alpha = 0.5)
            print(len(acc))
        ax.set(xlabel = labels[i])
        ax.legend(loc = "upper right")
        format_fig = "svg"
        names = ["charges.", "pdg.", "E.", "theta.", "excess."]
        if savefig:
            #plt.savefig(names[i],transparent = False, format = "png")
            plt.savefig(dir_res + names[i] + format_fig,transparent = True, format = format_fig)


def analyse_mix_gamma(pred,labels):
    print(f"labels bigger than 2: {ak.count_nonzero(ak.num(labels) > 2)}")
    print(f"labels smaller than 1: {ak.count_nonzero(ak.num(labels) < 1)}")
    dim_pred = ak.num(pred,axis =1)
    dim_labels = ak.num(labels, axis = 1)
    mask_excess_pos = dim_pred > dim_labels
    excess_pos = ak.count_nonzero(mask_excess_pos)
    mask_excess_neg = dim_pred < dim_labels
    excess_neg = ak.count_nonzero(mask_excess_neg)
    print(f"total number of events {ak.num(labels,axis = 0)}")
    print(f"excess predictions +: {excess_pos}, max excess: {ak.max(dim_pred[mask_excess_pos])}")
    print(f"excess prediction -: {excess_neg}, min excess: {ak.min(dim_pred[mask_excess_neg])}")
    #true one photon labels
    label_1g = ak.count_nonzero(dim_labels == 1)
    #true 2 photons labels
    labels_2g = ak.count_nonzero(dim_labels == 2)
    #correct pred 1 photon
    pred_1g_correct = ak.count_nonzero((dim_pred == 1) * (dim_labels == 1))
    pred_2g_instead_1g = ak.count_nonzero((dim_pred == 2) * (dim_labels == 1))
    #false pred 1 photon
    pred_1g_instead_2g = ak.count_nonzero((dim_pred == 1) * (dim_labels == 2))
    pred_2g_correct = ak.count_nonzero((dim_pred == 2) * (dim_labels ==2))
    mask_pred_other = np.logical_or(dim_pred > 2, dim_pred < 1)
    #false pred other instead of 1
    pred_other_instead1 = ak.count_nonzero(mask_pred_other * (dim_labels == 1))
    #false pred other instead of 2
    pred_other_instead2 = ak.count_nonzero(mask_pred_other * ~(dim_labels == 1))
    mat2x2 = np.array([[pred_1g_correct, pred_2g_instead_1g], [pred_1g_instead_2g, pred_2g_correct]])
    mat2x3 = np.array([[pred_1g_correct, pred_2g_instead_1g, pred_other_instead1], [pred_1g_instead_2g, pred_2g_correct, pred_other_instead2]])

    xlabels = [r"$1$", r"$2$", r"$\textrm{others}$"]
    ylabels = ["$1$", "$2$"]
    fig2x2, ax2x2 = plt.subplots()
    fig2x3, ax2x3 = plt.subplots()
    sns.heatmap(mat2x2, annot = True, fmt = "d", cmap = "Blues", xticklabels = xlabels[:-1], yticklabels = ylabels, ax = ax2x2)
    sns.heatmap(mat2x3, annot = True, fmt = "d", cmap = "Blues", xticklabels = xlabels, yticklabels = ylabels, ax = ax2x3)
    ax2x2.set(xlabel = r"$\textrm{predictions}$", ylabel = r"$\textrm{labels}$")
    ax2x3.set(xlabel = r"$\textrm{predictions}$", ylabel = r"$\textrm{labels}$")

    fig2x2.savefig("matrix2x2.svg", format = "svg")
    fig2x3.savefig("matrix2x3.svg", format  = "svg")

    
if __name__ == "__main__":
    Nbins  = [40,40,50,60,60]
    colour = [[0.57647  ,0.00000  ,0.00784], [0.04314  ,0.46275  ,0.62745]]
    savefig = False
    dir_res = "128/"
    dir_path = "Results/gamma_1_MSE/good_vocab/"
    #dir_path = "Results/electron_1_tracks/"
    #dir_path = "Results/mixg1g2/"
    #run_path = "true_70cont/dmodel_64/res_model_best/test"
    #run_path = "10GeV_100mrad/dmodel_128"
    run_path = "scheduler/scheduler_False/res_model_best/test"
    #pred_torch = torch.load(dir_path + run_path + "/prediction.pt")
    #labels_torch = torch.load(dir_path + run_path + "/labels.pt")
    pred = samples_to_awkward(torch.load(dir_path + run_path + "/prediction.pt"))
    labels = samples_to_awkward(torch.load(dir_path + run_path + "/labels.pt"))
    pred[:10].show()
    labels[:10].show()
    mask = ak.num(labels,axis = 1) == ak.num(pred,axis = 1)
    print(f"number of similar cluster {ak.count_nonzero(mask)}")
    
    pred_all = pred
    label_all = labels
    mask_events = mask
    pred_masked = ak.flatten(pred[mask_events],axis =0)
    labels_masked = ak.flatten(labels[mask_events], axis = 0)
    #pred_masked = pred
    #labels_masked = labels
    print(ak.num(pred_masked,axis = 0))
    print(ak.num(labels_masked,axis =0))
    #make_energy_angles_distributions(pred_masked, labels_masked, 1000)
    energy_2d_histlog10(pred_masked,labels_masked,int(ak.num(pred_masked,axis = 0)))
    #energy_2d_hist(pred_masked,labels_masked,int(ak.num(pred_masked,axis = 0)))
    energy_zone_split(pred_masked,labels_masked)
    #energy_zone_split5to50(pred_masked,labels_masked)
    #energy_distrib_diff_log10(pred_masked,labels_masked, int(ak.num(pred_masked,axis = 0)), "diff_log10_all.svg")

    pred = pred_all
    label = label_all
    #analyse_mix_gamma(pred,label)
    #make_energy_log10_diff_distrib(pred,labels,1000)
    #energy_distrib_diff(pred[:3000],labels[:3000],3000, "diff_E.svg")
    #energy_zone_split(pred[:3000], labels[:3000])

    #dir_path = "Results/Tracks/"

    #run_path2 = "all_70cont/dmodel_64/res_model_best/test"
    #pred2 = samples_to_awkward(torch.load(dir_path + run_path2 + "/prediction.pt"))
    #labels2 = samples_to_awkward(torch.load(dir_path + run_path2 + "/labels.pt"))
    
    #make_energy_angles_distributions(pred2, labels2, 1000)

    
    # run_path3 = "V1_notracks_norm_first"
    # pred3 = samples_to_awkward(torch.load(dir_path + run_path3 + "/prediction.pt"))
    # labels3 = samples_to_awkward(torch.load(dir_path + run_path3 + "/labels.pt"))
    
    # run_path4 = "V1_notracks_notnorm_first"
    # pred4 = samples_to_awkward(torch.load(dir_path + run_path4 + "/prediction.pt"))
    # labels4 = samples_to_awkward(torch.load(dir_path + run_path4 + "/labels.pt"))
    #pred.show()
    #labels.show()

    
    # dir_path3 = "Results/RV2_newLoss/"
    # run_path3 = "run1/epoch_1_res"
    # predV2_4epoch = samples_to_awkward(torch.load(dir_path3 + run_path3 + "/prediction.pt"))

    # preds = [pred_V1,pred, predV2_4epoch]
    # plot_labels = [r"$\textrm{V1 10 epochs}$",r"$\textrm{V2 10 epochs}$", r"$\textrm{V2 4 epochs}$"]
    # order = [0,2,1]
    
 
    start = time()
    accuracies_full = accuracy(pred, labels, "full", False, "full", "full", "full", "full", "full")
    #accuracies_full2 = accuracy(pred2, labels2, "full", False, "full", "full", "full", "full", "full")
    accuracies_event = accuracy(pred, labels, "full", False, "event", "event", "event", "event", None)
    #accuracies_event2 = accuracy(pred2, labels2, "full", False, "event", "event", "event", "event", None)
    #accuracies_event3 = accuracy(pred3, labels3, "full", False, "event", "event", "event", "event", None)
    #accuracies_event4 = accuracy(pred4, labels4, "full", False, "event", "event", "event", "event", None)

    
    #compare_distributions((accuracies_event, accuracies_event2), 
    #                        Nbins, ["10", "70"], colour, savefig, dir_res)
    #compare_distributions((accuracies_event3, accuracies_event4), Nbins, ["norm first", "not norm first"], colour)
    titles = [r"\textrm{charges accuracy}", 
              r"\textrm{PDGs accuracy}", 
              r"$\frac{\vert E_{p} - E_l \vert}{E_l}$", 
              r"$\theta \, [^{\circ}]$", 
              r"\textrm{Excess predictions}"]
    ranges = [(-0.01,1.01), (-0.01,1.01), (-0.01,0.25), (-0.01,20), None]
    for i,accuracy in enumerate(accuracies_event):
        fig = make_histogram(accuracy, Nbins[i], titles[i], ranges[i])
        fig.savefig(f"{titles[i]}.svg", format = "svg", transparent = True)
        print(min(accuracy))
    
    fields = ["charges", "pdgs", "energy", "dir", "excess"]
    dict_full = dict(zip(fields,accuracies_full))
    #dict_full2 = dict(zip(fields,accuracies_full2))
    print(run_path + f"  {dict_full}")
    #print(run_path2 + f"  {dict_full2}")
    plt.show()
