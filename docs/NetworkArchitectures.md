# Network Architectures
This document starts by explaining the basic principles of the original transformer which will be used to motivate the choices of implementations later on. Then, two implementations to our problem are presented. The first one follows the original architecture whereas the second one introduces some modifications impacting the decoder part only. As side modifications, two ways of dealing with the special tokens will be presented. 

Transformers were originally designed by [Vaswani and al.](http://arxiv.org/abs/1706.03762)  for machine translation tasks for which the network is fed an input sequence in one language and outputs its translation in another.As in this project the particle flow algorithm is interpreted in complete analogy with a machine translation task, the latter, simpler, will be used to explain the network's general architecture.

## The original Transformer in Machine Translation
### Preprocessing of the input sentence
As the foundation of neural networks is based on linear transformation of vectors, the first step in a machine translation task is the _tokenization_, i.e., to split the sentence into different _tokens_. Those are then arbitrarily associated to a unique integer, forming the _vocabulary_. Note that different tokenization can be used to choose the tokens the sentence is being split into. Nowadays, those are usually composed of a few characters which allows for a vocabulary of a limited size. Indeed, if tokens are chosen as individual words, the size of the vocabulary could easily exceed 50'000 words, making the process a lot more expensive computationally. Albeit interesting, these considerations are out of the scope of this document and since taking tokens as individual words can bring a more intuitive explanation of the process, this is what will be used in the following paragraphs.

 These steps are followed by the _embedding_, during which a neural network maps each integer to a $d_{model}$ dimensional vector. If $n$ is the number of tokens in a sequence, the final result is thus a matrix of dimension $(n+2) \times d_{model}$. Indeed, two additional tokens are added to mark the beginning and end of the sentence. The one at the beginning of the sentence (bos) is usually denoted `<bos>` whereas the one at the end of the sequence (eos) is denoted `<eos>`. Their integers values are usually specified as 1 and 2 respectively. This forms the input of the Transformer.

 > **NOTE:**
> In machine translation, a last step called _positional encoding_ is used to keep track of the tokens' positions in the sentence. However, for reasons explained later on, this is not used in this report's implementation and thus not presented.

 A concrete example of these steps is illustrated below:

|<img width="355" alt="First steps of the input sentence processing before feeding it to the transformer" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/2ebc8398-27da-40f0-a16b-76ddc762fb46">|
:---:
|Figure 1: First steps of the input sentence processing before feeding it to the transformer.|

As the meaning of words is highly context dependent, a way of communicating information between different parts of a sentence needs to be implemented and is the purpose the _attention_ mechanism. Although attention had already been developed prior to Transformers, Vaswani and al. introduced the so-called _Scaled Dot-Product Attention_.
### Scaled Dot-Product Attention
Attention allows to transfer information between tokens in the same sentence by updating their embedding according to the \textit{attention weights_. In a simple attention head, each token $i$ gives rise to three new vectors of dimension $d_k$, $d_k$, $d_v$, called the Query ($Q_i$), the Key ($K_i$) and the Value ($V_i$) resp., obtained by the action of three linear transformations on the original token. 

> **NOTE:**
> In the following, the usage of token can refer both to the characters or their corresponding embedding. The meaning should be clear from context.

For each token $i$, an euclidean dot-product, scaled by $\frac{1}{\sqrt{d_k}}$ between its query $Q_i$ and all keys $K_j$ is computed. The results are then normalised by the softmax function (smax), and are used as weights to sum each of the values vector, $V_j$, giving the new embedding of the token $i$. Since the embeddings are stored in the embedding matrix as $1\times d_{model}$ vectors (see Fig. above), it is natural to also define the individual keys, queries and values as $1\times d_k$, $1\times d_k$ and $1\times d_v$ vectors resp.  The new embedding, $E_i$ for the $i^{th}$ token can thus be expressed, using Einstein summation convention, by:
    $$E_i = \textrm{smax}\left(\frac{Q_iK_j^T}{\sqrt{d_k}}\right)V_j, \textrm{ where } \textrm{smax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$

This can be computed for every token, leading to the new embedding matrix $E$ of dimension $(n+2) \times d_{v}$ given by:
$$E = \textrm{smax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,$$
where $Q$ and $K$ are now $(n+2) \times d_k$ matrices containing the queries and keys vector resp. of every tokens and $V\in \mathbb{M}_{(n+2)\times d_v}(\mathbb{R})$ contains all values vectors. A _plausible_, hand-wavy interpretation can be obtained by assuming that the queries of each tokens represent a direction in which the latter can be contextualised. It follows that keys encode the ability of each token to bring useful context with respect to queries and values then bring context itself. If a key and query point in the same direction, the corresponding attention weight will be close to 1 once softmax is applied. Thus, at least in appearance, high importance is given to the corresponding token's value.

A typical attention layer in a Transformer is a Multi-Head Attention (MHA) layer. Queries, keys, and values are projected onto $h$ different spaces by $3h$ linear transformations, leading to the "sub"-queries/keys/values, $Q^{(j)}, K^{(j)}, V^{(j)}$ with $j = 1, \ldots, h$ of dimensions $(n+2)\times \frac{d_k}{h}$, $(n+2)\times \frac{d_k}{h}$ and $(n+2)\times \frac{d_v}{h}$. Each tuple ($Q^{(j)}, K^{(j)}, V^{(j)}$) are then used to compute the attention as in the equation right above. The $h$ resulting sub-embeddings are then concatenated together forming the layer's output embedding matrix $(n+2) \times d_{model}$. This process is illustrated on Fig. 2 below, taken from [Vaswani and al.](http://arxiv.org/abs/1706.03762). 

|<a name = "MHA"><img width="175" alt="Multi-Head attention layer in an original transformer architecture" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/7532cf9d-6197-449d-ab86-68205824fc7b" title = "Multi-Head Attention Layer"></a>|
:---:
|Figure 2: Multi-Head Attention Layer. The input is projected onto its Query (Q), Key (K) and Value (V). The resulting vectors are then projected onto $h$ different subspaces, where the attention is computed.|



### Encoder Architecture 


The encoder is one of the two submodules of the transformer consisting of duplicates of a single structure called an _encoder layer_, stacked on each other. The input of a single layer is the output of the previous layer. 


The purpose of the encoder is to contextualise each of the tokens and to create a representation of the meaning of the sentence in the embedding's abstract vector space. As attention is computed in-between every tokens, this is usually referred to as _self-attention_. The output of the encoder is computed only once and is called the _memory_.

This function is accomplished by two main modules, as illustrated by Fig. 3a (adapted from  [Vaswani and al.](http://arxiv.org/abs/1706.03762)): A MHA layer and a fully connected, 2 layers, Feed-Forward network. After each of those, an _Add & Norm_ layer is added. The latter is used to create a [residual connection](http://arxiv.org/abs/1512.03385) between the input of the module it follows, $\vec{x}$, and its output, $\mathcal{F}(\vec{x})$, resulting in $\vec{x}'$ defined by:

$$\vec{x}' = \mathcal{F}(\vec{x}) + \vec{x}.$$

Although this was originally motivated to make the identity mapping of a layer equipped with a residual connection easier, it was shown that this connection was of particular importance in training deeper neural network by reducing their training error. 

The "Norm" of _Add & Norm_ is a function normalising its argument, [defined](http://arxiv.org/abs/1607.06450) by:

$$\textrm{Norm}(\vec{x}) = \frac{1}{\sigma}(\vec{x} - \vec{\mu}),$$


where 


$$\vec{\mu} = \frac{1}{d}\sum_{i = 1}^{d}\vec{x}_i \textrm{ and } \sigma = \sqrt{\frac{1}{d}(\vec{x}_i - \vec{\mu})^2},$$


with $d$ the dimension of $\vec{x}$. This normalisation was originally designed to prevent exploding gradients and reducing training time. 

### Decoder Architecture 


The second submodule of a transformer is a decoder. As for the encoder, it is composed of a stack of single _decoder layers_, whose architecture is shown on Fig. 3b. It consists of three modules: two MHA Layers, and a FFN. After each of those, an Add \& Norm layer is added. 

Its input varies depending if the model is in training mode or inference. During training, its input is the translated sentence preceded by a `<bos>` token. Note that no `<eos>` token is added. During inference, during the first iteration of the model only a `<bos>` token is provided. Then, its input consists of the translated tokens from previous iterations. 

The purpose of the decoder is twofold. Firstly, in the same fashion as for the encoder, it allows to contextualise its input tokens with a _masked_ self-attention layer. The mask is used to ensure that a token at position $i$ gets context only from tokens at position $j \leq i$ thus mimicking the inference process, for which only the information from tokens up to position $i$ can be used to predict the next token. It is implemented by setting the attention values from the subsequent tokens to $-\infty$, so that they will be sent to $0$ once softmax is applied. 

Secondly, it compares its input with the memory, using a classic Multi-Head Attention layer. In this MHA Layer, as information must flow from the original sentence to the one in making, keys and values come from the memory, whereas the Masked MHA's output is only used to produce the queries. 

|![Encoder-2](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/258a87a2-459c-451d-929f-675af56120ce)|![Decoder](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/cad6b99d-fd6d-4f9a-888b-ec6590cb3d41)|
|:---:|:---:|
|Fig. 3(a) Single encoder layer architecture|Fig. 3(b) Single decoder layer architecture|




### The Transformer Architecture 


As illustrated on Fig. 4, also adapted from  [Vaswani and al.](http://arxiv.org/abs/1706.03762), a Transformer can be decomposed into an encoder and a decoder, although encoder-only Transformers are also used, depending on the task at hands. Indeed, encoder-only Transformers are usually used for classification problems for which the output needs not to be reused as it is the case for image recognition in computer vision or jet tagging in physics. Having a decoder allows for an auto-regressive network in the sense that the output of the decoder at one iteration of the process will be used, after transformation, as its input for the next iteration. 

|![TransfoArchitecture](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/c717a036-d627-406e-adc8-52f17f5acbd8)|
:---:
|Figure 4: Architecture of the original transformer. On the LHS, the main rectangle is the encoder and the RHS is the decoder.|

Part of this last transformation is to convert the output of the decoder as something usable to make the prediction of the next token. The last embedding of the predicted sentence has attended with every other tokens. It is thus chosen as the vector on which operates a last linear transformation $f: \mathbb{R}^{d_{model}} \to \mathbb{R}^{l_{voc}}$, where $l_{voc}$ is the size of the vocabulary associated with the language the translation is made to. Applying a softmax function on the resulting vector allows to normalise each entry of this vector such that their sum adds up to 1 and each of them takes value between 0 and 1. The number at entry $i$ is now interpreted as the probability that the token associated to the integer $i$ is next. To predict the next token, it is then enough to find the maximum probability in this output vector.

This process is concretely illustrated on Fig. 5. During inference, the sentence to translate is given to the encoder whereas only a `<bos>` token is given to the decoder. Once the initial sentence has been processed by the encoder, the decoder uses it, as well as the `<bos>` token, to predict the first token of the translation, $t_1$. The output $\textrm{bos} t_1$ is then fed again to the decoder, where the same process will be used to predict the second token $t_2$ and so on. The translation stops when a `<eos>` is predicted by the network. 

|![MT_TransfoFlow-2](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/19628751-4595-432f-815b-26653f6aca86)|
|:---:|
|Figure 5: Machine translation, inference, using a transformer architecture. Starting input are in green, processes in blue, intermediate results in orange and final output in red.|




During training, thanks to the masked MHA layer, a token from the decoder's output at position $i$ can be considered as the last token of a sentence of length $i$ and used to predict the $i+1$ token. Thus, the last linear and softmax transformations are applied to all tokens forming the decoder's output. The result is a matrix of dimension $(n+1) \times l_{voc}$ where the element at $(i,\,j)$ is the probability that the token at position $i+1$ and associated to $j$ in the vocabulary is next. This output can be directly compared, using an appropriate loss function, with the vocabulary version of the translated sentence to which was added an `<eos>` token.

## Transformer as PFA: Version 1
> **NOTE:**
> - As several notions presented in the [Preprocessing of the dataset](DatasetsPreprocessing.md), it is advised to first read it first before continuing.
> - Tokens corresponding to hits or labels will be referred to as sample to contrast with special tokens reserved for `<pad>`, `<bos>`, and `<eos>`.
> - This version, as described below, is the one implemented in the branch `main`

The transformer should take as input hits from the calorimeters, whose features are their energies and positions, to predict the associated cluster, characterised by the particle's charge, PDG, energy and direction. This implies changes to the preprocessing of the data and to the transformation applied to the decoder's output, compared to the original architecture. Summarised, these are:

- Addition of 2 binary features to both hits and labels to distinguish between samples and labels.
- Implementation of two plain Feed Forward Networks as embedders for the hits and labels.
- Decoder's output undergoes three different linear transformations in parralell to
    - Produce probabilities for the next charge
    - Produce probabilities for the next PDG
    - Produce the values for the continuous degrees of freeedom (DOFs)

The rest of the modifications follow as consequences of the above considerations and are presented in the following sections. Since choices regarding data preprocessing are motivated by the Network final output, the transformations to the decoder's output as well as the different workflows during training and inference will be presented first in [Producing the outputs](#producing-the-outputs). How input feats and labels are handled before going through the transformer follows in [Data preprocessing](#data-preprocessing).


### Data preprocessing
To distinguish between samples and special tokens, this first implementation gives samples two additional binary features, encoded as follows:
 - (0,0) is associated to `<sample>`
 - (0,1) is associated to `<pad>`
 - (1,0) is associated to `<eos>`
 - (1,1) is associated to `<bos>`

Furthermore, as stated above, the transformer will need to output predictions for the next charge, PDG, and the cluster's continuous DOFs. Anticipating on the presentation of the training and validation workflow, the transformer output will be compared to the labels using three loss functions:

- 1 Mean Square Error (MSE) for the continuous DOFs.
- 2 Cross Entropy for the charge and PDGs.

In other words, the continous DOFs are treated as a linear regression problem, whereas charges and PDGs as classification problems. For the latter case, this is similar to machine translation and can be dealt with by constructing two vocabularies: one for the charges and one for the PDGs. As in machine translation, these vocabularies will associate each charge and PDG value to a unique integer, as well as creating entries for special tokens. For both vocabularies, the correspondance between the special tokens and their unique integer, referred to as indices, is constructed as:

 - -150 $\leftrightarrow$ 0 for `<pad>`
 - -100 $\leftrightarrow$ 1 for `<bos>`
 - -50 $\leftrightarrow$ 2 for `<eos>`
   
In the following, the values (-150, -100, -50) will be referred to as dummy values to distinguish them from the physical values corresponding to the samples' charges or PDGs. The only requirement that those dummy values need to satisfy is being negative, and for the charges, smaller than -1, so that they will not be confused with physical values. 

> **NOTE:**
> Recall that for the PDGs, the feature actually corresponds to the absolute value of the PDG.

The rest of the vocabularies is created by sorting in ascending order the unique values of charges and PDGs found in the dataset and associating them to integers starting from 3 onwards. This conversion from integer values to integer values might seem useless at first but is a requirement to fully exploit Pytorch's [Cross Entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Indeed, when initializing the loss function, it is possible to specify an index which will be ignored when computing the loss. This is particularly useful to avoid taking into account padding tokens in the loss. This loss function can take as input unnormalized probabilities (i.e. before softmax is applied) and compare it with the labels' indices for each token. In order for the specified index to be ignored, the class' indices must be correspond to their position in the probabilities vector. Thus, since the index $0$ corresponds to a padding token, the probability at entry 0, correpsonds to the probability of the next token being a padding token and will be ignored. 

From these considerations, the preprocesing of the feats follows:
1. Normalisation of energies and positions as described in [Datasets & Preprocessing](DatasetsPreprocessing.md)
2. Addition of $(0,0)$ at the end of each hit
3. Addition of `<eos>`, `<bos>` and `<pad>`, constructed by concatenation of a 4D zero vector and their correponding binary encoding.

On the other hand, the preprocessing of the labels follows:
1. Normalisation of energies and positions
2. Addition of $(0,0)$ at the end of each cluster
3. Addition of `<eos>`, `<bos>` and `<pad>`, constructed by first
    1. concatenation of a 6D zero vector and their correponding binary encoding.
    2. Modification of entries 0 and 1 to their corresponding dummy indices, e.g.
       $$\textrm{pad} = (-150, -150,0,0,0,0,0,1) \textrm{, bos} = (-100, -100 ,0,0,0,0,1,0) \textrm{ and bos} = (-50,-50,0,0,0,0,1,1)$$
4. Translation of the dummy and physical values to their corresponding indices by the charge and PDGs vocabularies.

Once these steps are executed, both feats and labels go through their corresponding Embedders.

### Embedders
Although in machine translation the embedders for the encoder's input can be the same as for the decoder's, it is not possible in this case, since processed feats and labels do not have the same dimensions, nor contain the same kind of information. Therefore, two embedders are created as two different instances of the same `Embedder` class. 

The `Embedder` class implements a plain Feed-Forward Network with variable number of layers, chosen by the user. At each layer, ReLU is used as an activation function. For the feats, the first layer corresponds to a linear transfomration from $\mathbb{R}^{6} \to $\mathbb{R}^{d_{model}}$. All the other layers have transformations from $\mathbb{R}^{d_{model}} \to \mathbb{R}^{d_{model}}$. The labels embedder has a similar structure, except that the first transformation is a map from $\mathbb{R}^{8}\to \mathbb{R}^{d_{model}}$. 

Once feats and labels went through their associated embedders, they are sent to the encoder and decoder respectively.

### Producing the outputs
The decoder outputs an embedding matrix of dimension $(n+1)\times d_{model}$. From these, three kind of information need to be retrieved:
1. Probabilities for the next charge
2. Probabilities for the next PDG
3. Values of the next cluster's energy and direction

Thus, the decoder's output undergoes three different linear transformations, $f_C$, $f_P$, $f_E$, to produce the logits corresponding to the charge, PDG and continous DOFs respectively. For $f_C$ and $f_P$, their outputs is interpreted as the unnormalized probabilities that the indice $i$ is the next token, such that $f_C: \mathbb{R}^{d_{model}}\to \mathbb{R}^{l_C}$ and $f_P: \mathbb{R}^{d_{model}}\to \mathbb{R}^{l_P}$, where $l_{C/P}$ are the lengths of the charge/PDGs vocabularies. Both loss functions are initialised as Pytorch's CrossEntropyLoss, ignoring the padding tokens. 

As the vector for the cluster direction should be normalized to 1, $f_E$ is defined as map $f_E: \mathbb{R}^{d_{model}}\to \mathbb{R}^{3}$, with the first entry of its output interpreted as the (normalised) log base 10 of the energy, and the two others as $\theta$ and $\phi$
respectively. In order to avoid problems from the non-periodicity of the MSE loss function, these angles are converted to 3D cartesian coordinates, which are then compared to the ground truth. Moreover, since there is no direct way of ignoring entries in Pytorch's [MSE Loss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) function, this is done by applying a mask to the predictions and ground truth, selecting only positions for which the ground truth is not a padding token.

> **NOTE:**
> Indeed, if the predicted angles are (0.01, 359.78) with a ground truth being $(0,0)$, the MSE loss function will strongly penalize this prediction.

During inference, it is still necessary to decide which kind of token was predicted. In this first version, this is based exclusively on the prediction of the charge tokens.
> **NOTE:**
> This is an arbitrary choice

Thus, if the entry with the highest probability in the charges logits is correspond to an `<eos>` token, the inference will be stopped. Note that padding token cannot be predicted since they do not appear in the loss function by construction. If the charge index corresponds to another token, the inference process continues after the additional features [(0,0), (0,1), ...] were added to the newly formed cluster. Again, which of those binary encoding is added depend solely on the token predicted by the charge output. 

## Transformer as PFA: Version 2
>**NOTE:**
>This version, as described below, is the one implemented in the branch `dev_MMHA`

This version tries to address the potential problem of finding a common embedding space between feats and labels. The main differences with the first version are:
- Changes to the dimensions of the embedder's hidden layers
- Implementation of a custom decoder architecture in the transformer

### Embedders
Those changes are mainly motivated by examples of embedders found in the literature. If the embedder consists of more than one layer, the first layer is changed from being a map from $\mathbb{R}^{n} \to \mathbb{R}^{d_{model}}$, to a map $\mathbb{R}^{n} \to \mathbb{R}^{d_{ff}}$, where $d_{ff}$ is a hyperparameter, usually chosen as $2d_{model}$. As before, the following layers are maps from $\mathbb{R}^{d_{ff}} \to \mathbb{R}^{d_{ff}}$, except for the last one, for which it has to be $\mathbb{R}^{d_{ff}} \to \mathbb{R}^{d_{out}}$. Note that for reasons that will become clear in the next section, $d_{out}$ can now be different for feats embedder and the labels. 

### Custom decoder 
The main motivation behind the changes and the architecture is to give the possibility to the model to store more information in the labels' embedding than the hits' whilst still being able be compared to hits. This train of thoughts takes its origin in two reasons:
1. Labels contains more information than hits, since it also adds the charges and PDGs. It seems thus natural that their embedding should be larger.
2. Since labels and hits do not contain the same type of information, maybe it is more difficult to create to embedding spaces where the dot products between the memory's keys and labels' queries is relevant. Projecting the input labels into different subspaces allows to give more freedom to the model to adapt if needed.

This is achieved by feeding the decoder a higher dimensional embedding of the labels than the feats, and then projecting them on $n$ different vectors, such that $d_{labels} / n = d_{feats}$, where $d_{labels}$ is the dimension of the labels embedding (set by the $d_{out}$ of the labels embedder described above) and $d_{feats}$ the dimension of the feats embedding. The results are $n$ $(N \times n_{samples} \times d_{feats})$ "sub-embedding" matrices, with $N$ the number of events and $n_{samples}$ the number of tokens per event.
> **NOTE:**
> Recall that both feats and labels embedding matrices were padded such that $n_{samples}$ for feats and labels is the same for each event.

Each individual sub-embedding matrix will then go through a usual single layer decoder, and compared with the memory. The $n$ resulting outputs are then concatenated back together again, followed by an Add & Norm layer, a one hidden layer Feed-Forward layer, and a last Add & Norm layer. This is more clearly illustrated on the Figure below, where for simplicity $n = 2$. Note also that the dimension of the FFN hidden layer is a hyperparameter and was chosen here to be $2d_{labels}$.
|![CustomDecoder](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/ec66ecd9-4e8a-4fed-a11a-e71eedb0c134)|
|:---:|
|Figure 5: Custom decoder layer. Note that the link between the decoders and memory is not shown on the diagramm.|

## Other changes
What follows is a list of different changes, for some not directly related to the model architecture but that can easily be implemented. All of those were implemented in the branch `new_loss`.

- In the data preprocessing, cluster are sorted by decreasing energy.
- In the data preprocessing, cluster having an energy lower than a fixed threshold are discarded (usually 0.1 GeV)
- Optimization of the data preprocessing, in terms of memory usage and time efficiency.
- Implementation of a new loss function, specialised in predicting the kind of tokens.

### Implementing a new loss function
In the previous implementations, the additional features indicating the kind of tokens were redundant information for the labels, since this information was also given in both the PDGs and charges indices. To remedy to this redundancy as well as building a more coherent network, this information was removed from the charges and PDGs, and a new loss function was implemented, specialising in predicting the next token's kind. This implies the following changes:
- Since predicting a 2 dimensional encoding is difficult, it was changed to 4 integers ranging from 0 to 3, such that 0 is associated with `<pad>`, 1 to `<bos>`, 2 to `<eos>` and 3 `<sample>`.
- To keep the functionality of the CrossEntropyLoss to ignore padding tokens, the charges and PDGs vocabularies special tokens were reduced to a unique entry in the dictionnary, corresponding to a special token with dummy value of -50 and index of 0. The rest of the vocabularies are constructed the same way. After preprocessing, but before charges and PDGs translation, the special tokens for the labels are:
  $$\textrm{pad} = (-50,-50,0,0,0,0,0), \textrm{ bos} =  (-50,-50,0,0,0,0,1), \textrm{and eos} = (-50,-50,0,0,0,0,2)$$
This ensures that the corresponding loss functions will take into account only entries corresponding to sampels and not special symbols, reserved to the new loss function. 
- The output of the (custom) decoder is now split into 4 by an additional linear transformation from $\mathbb{R}^{d_{labels}} \to \mathbb{R}^{4}$.

Those changes are implemented in the file data_prepro.py. 
