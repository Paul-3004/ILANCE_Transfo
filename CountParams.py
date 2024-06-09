def count_params_model(nlayers_encoder, nlayers_decoder, nlayers_embedding, d_ff, 
                                                         d_charges = 6, 
                                                         d_PDGs = 12, 
                                                         d_cont = 6,
                                                         d_model = 512,
                                                         d_init_decoder = 8,
                                                         d_init_encoder = 6):
    multi_attention = 4 * d_model**2 
    ff = 2*d_model * d_ff
    encoder = nlayers_encoder * (multi_attention + ff)
    decoder = nlayers_decoder * (2 * multi_attention + ff)
    last_lin = d_model*(d_charges + d_PDGs + d_cont)
    Transfo = encoder + decoder + last_lin
    embedding_src = d_init_encoder * d_model + (nlayers_embedding-1) * d_model**2
    embedding_tgt = d_init_decoder * d_model + (nlayers_embedding-1) * d_model**2
    embedding = embedding_src + embedding_tgt
    total = embedding + Transfo 

    params = {
        "total": total,
        "embedding": embedding,
        "Transfo": Transfo,
        "last_lin": last_lin,
        "encoder": encoder,
        "decoder": decoder}

    return params

params_all_you_need = count_params_model(6,6,2,2048,6,12,6,512)
params_current =  count_params_model(nlayers_encoder = 1,
                                     nlayers_decoder = 1,
                                     nlayers_embedding = 2,
                                     d_ff = 512,
                                     d_charges = 6,
                                     d_PDGs = 12,
                                     d_cont = 6,
                                     d_model = 256,
                                     d_init_decoder = 8,
                                     d_init_encoder = 6)

print(params_current["Transfo"])
print(params_current["total"])