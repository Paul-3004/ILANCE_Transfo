def param_ff_layer(d_input, d_output):
    return d_input * d_output + d_output

def triangle_ff(d_input, d_hid, d_ouput):
    return param_ff_layer(d_input, d_hid) + param_ff_layer(d_hid,d_ouput)

def projection(d_input, d_ouput):
    return d_input * d_ouput

def param_MHA(d_input):
    return 4*d_input**2

def param_encoder(d_input, d_ff):
    return 4 * param_MHA(d_input) + triangle_ff(d_input,d_ff,d_input)

def param_decoderV2(d_input, nproj,d_ff):
    MHA_labels = param_MHA(d_input)
    MHA_proj = nproj * param_MHA(d_input)
    return MHA_labels + MHA_proj + triangle_ff(d_input * nproj, d_ff,d_input * nproj)

def count_params_model(nlayers_encoder, nlayers_decoder, nlayers_embedding, d_ff, 
                                                         d_charges = 6, 
                                                         d_PDGs = 12, 
                                                         d_cont = 6,
                                                         d_model = 512,
                                                         d_init_decoder = 8,
                                                         d_init_encoder = 6):
    multi_attention = 4 * d_model**2 
    ff = 2*d_model * d_ff + d_model + d_ff
    encoder = nlayers_encoder * (multi_attention + ff)
    decoder = nlayers_decoder * (2 * multi_attention + ff)
    last_lin = d_model*(d_charges + d_PDGs + d_cont)
    Transfo = encoder + decoder + last_lin
    embedding_src = d_init_encoder * d_model + (nlayers_embedding-1) * d_model**2 + 2*d_model
    embedding_tgt = d_init_decoder * d_model + (nlayers_embedding-1) * d_model**2 + 2*d_model
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

def count_params_V2(nlayers_encoder, 
                    nlayers_decoder,    
                    nproj = 4, 
                    d_hid_tgt = 1024,
                    d_hid_src = 512, 
                    d_ff_transfo_src = 256,
                    d_ff_transfo_tgt = 2048,
                    d_charges = 6, 
                    d_PDGs = 12, 
                    d_cont = 6,
                    d_model_tgt = 512,
                    d_model_src = 128,
                    d_init_decoder = 8,
                    d_init_encoder = 6):

    src_embedding = triangle_ff(d_init_encoder, d_hid_src, d_model_src)
    tgt_embedding = triangle_ff(d_init_decoder, d_hid_tgt, d_model_tgt)
    projections = nproj * projection(d_model_tgt,d_model_src)
    encoder = param_encoder(d_model_src, d_ff_transfo_src)
    decoder = param_decoderV2(d_model_src,nproj,d_ff_transfo_tgt)
    last_lin = param_ff_layer(d_model_tgt,d_charges) + param_ff_layer(d_model_tgt, d_PDGs) + param_ff_layer(d_model_tgt, d_cont)
    total = src_embedding + tgt_embedding + projections + encoder + decoder + last_lin
    Transfo = projections + nlayers_encoder *encoder + nlayers_decoder *decoder
    params = {
        "total": total,
        "embedding_src": src_embedding,
        "embedding_tgt": tgt_embedding,
        "Transfo": Transfo,
        "last_lin": last_lin,
        "encoder": encoder,
        "decoder": decoder}

    return params

params_all_you_need = count_params_model(6,6,2,2048,6,12,6,512)
params_current =  count_params_model(nlayers_encoder = 3,
                                     nlayers_decoder = 3,
                                     nlayers_embedding = 2,
                                     d_ff = 256,
                                     d_charges = 6,
                                     d_PDGs = 12,
                                     d_cont = 6,
                                     d_model = 128,
                                     d_init_decoder = 8,
                                     d_init_encoder = 6)
params_V2 = count_params_V2(nlayers_encoder = 1, 
                    nlayers_decoder = 1,    
                    nproj = 4, 
                    d_hid_tgt = 1024,
                    d_hid_src = 512, 
                    d_ff_transfo_src = 256,
                    d_ff_transfo_tgt = 1024,
                    d_charges = 6, 
                    d_PDGs = 12, 
                    d_cont = 6,
                    d_model_tgt = 512,
                    d_model_src = 128,
                    d_init_decoder = 8,
                    d_init_encoder = 6)                                     

print(params_current["Transfo"])
print(params_current["total"])
print(params_V2["total"])