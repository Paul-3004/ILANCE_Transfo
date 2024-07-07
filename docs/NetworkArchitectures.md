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



