from data_prepro import train_val_preprocessing, get_data

dir_res  = "/Users/paulwahlen/Desktop/Internship/ML/Code/TransfoV1/data/training/"
dir_train = "/Users/paulwahlen/Desktop/Internship/ML/Code/TransfoV1/data/training/"

#train_val_preprocessing(dir_train, dir_train, dir_res)

vocab_charges, vocab_pdgs, special_symbols, E_rms_normalizer, train_dl, val_dl = get_data((dir_res, dir_res), 10, 1, "training", True)

for i,(feats, labels) in enumerate(train_dl):
    print(labels)
    print(feats)
    pass
