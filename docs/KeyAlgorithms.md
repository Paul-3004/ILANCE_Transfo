# Key Algorithms
This document presents in details some key algorithms implemented in functions/methods and used in the project, to accelerate familiarization with the code.

## Training and validation
The training and validation algorithm are the same except that during validation, the weights are not updated and no backpropagation is performed. Thus, only the training algorithm will be presented. 

To accelerate the process, the training for one epoch is done in minibatches by the `train_epoch` function. For each minibatches, the function will make the forward pass, i.e. feed the preformatted feats and labels to the transformer and compare the outputs with the labels. By construction of the transformer architecture, for each tokens at position $i$, the model's forward method will output what can be interpreted as the logits for the prediction at position $i+1$. In other words, one event is used as $n_{samples}$ different examples, where $n_{samples}$ is the number of tokens in the event, excluding the padding tokens.

In order to do so, the last token for each event in the labels are discarded when the labels are used for the forward pass, and the output predictions will be compared to the labels but shifted right, i.e. starting from the second token in each event, thus discarding the `<bos>` tokens. In the implementatino of the model's forward method, the last linear transformations used to produce the unnormalised probabilities and continuous DOFs are already integrated. The forward's method output thus varies between 3 or 4 matrices depending on the model version. In any case, each of those logits can be used with their corresponding loss functions and the labels (shifted right) to compute the losses. For the continuous DOFs, their values are not taken into consideration in the MSE loss function is the labels' token is special, which is why a mask is applied. Again, the method by which this mask is obtained depends on the model's version.

The total loss is computed as a weighted sum of the individual losses, with the weights left as hyperparameters. Once it is computed, backpropagation is made and weights are updated accordingly.

This algorithm is illustrated on the Fig. 1 below, for the model 1 version. 
|![trainV1](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/4c05cce4-9e5a-4536-bd69-98593ca3dea8)|
|:---:|
|Figure 1: Flowchart of the `train_epoch` function used to train the model for one epoch.|

The `train_and_validate` is in charged of calling both `train_epoch` and `val_epoch`  for the number of epochs determined by the user. At the end of each epoch, the model's parameters are saved, and the best model is taken as the one producing the smallest validation loss averaged over an epoch.

## Greedy Algorithm
This algorithm is implemented in `main.py` as the function `greedy_func`. Its purpose is to generate the clusters by feeding the hits to the encoder, and the clusters from the previous iterations to the decoder. To speed up the inference process, this function was implemented such that inference could be done in minibatches and not event per event. 

The function starts by creating the special tokens and the decoder's input, consisting of one `<bo>` token per event, i.e. a $N \times d_{labels}$ matrix, where $N$ is the number of events in the minibatch and $d_{labels}$ the dimension of the labels' embedding, which depends on the implementation (see [Network Architectures](NetworkArchitectures.md) for more details.). Since the memory is computed only once, at the beginning of the inference process, the second step is to feed the hits to the model's encoder. 

Since the prediction is made in minibatches, it is necessary to stop the inference process for events where a `<eos>` token has already been predicted. This is done by keeping track of which event has an `<eos>` token by the variables `is_done` for the current iteration and `is_done_prev` for the previous iteration. If, for some events, `is_done_prev` is `True`, the current prediction will be manually replaced by a `<pad>`  token. Moreover, each time an `<eos>` token is predicted, the events corresponding entries in `is_done` are updated to `True` and the inference process stops for the entire minibatch once all entries of `is_done` are `True` or if the number of clusters predicted for an event exceeds a maximum number, fixed by the user. 

Lastly, as the number of predicted clusters will grow, the mask used to avoid padding tokens attending in the attention mechanism needs to be updated at each iteration. This is also done using the information stored in `is_done`.


### Iteration n = 0
The `<bos>` tokens are fed to the decoder, as well as the memory. Note that the `decode` method of the model does the embedding of those tokens automatically. The model's last linear transformations are then applied on the ouput to produce, for each minibatch, the predictions for the charges, PDGs, continuous degrees of freedom (DOFs) and, depending on the implementation, the kind of token. As also explained in [Network Architectures](NetworkArchitectures.md). Only the last token of each category is used to make the prediction.

>**NOTE:**
>Note that since the transformer was created with `batch_first = True`, the first dimension for each logits correspond to the minibatch number.

To obtain the prediction for the discrete DOFs (charges, PDGs and, depending on the implementation, the kind of token), it is enough to take the `argmax` of their corresponding logits. For the continuous DOFs, since the direction in the labels is in cartesian coordinate, it is necessary to make the conversion between the angles in spherical coordinates and the 3D cartesian. 

The last step is to keep track of the kind of token predicted. Independently of the version used, the underlying idea is the same: Use one of the predictions to classify the kind of token predicted. Then add, according to this prediction, the additionnal features to the predicted label corresponding to the kind of token predicted and the version used. In version 1, the prediction of the next charge is used, and the additional features are a 2D binary encoding [(0,0), (0,1), (1,0), (1,1)]. In version 2, the kind of token is predicted thanks to its own logits and the additional feature is an integer ranging from 0 to 3. Furthermore, if an `<eos>` token is predicted for event $i$, the $i^{th}$ entry of `is_done` is updated to `True`. 

The next cluster for each minibatch is finally obtained by concatenating all those information together, and can be added to the previous `<bos>` tokens.

Lastly, at the end of the iteration, `is_done` is cloned into `is_done_prev`, to store the values for the next iteration. 

### Iteration n > 1
Naturally works the same way as for the case $n= 0$ but instead of having only `bos` tokens as input to the decoder, it now consists of the output of the previous iteration $n-1$.

This algorithm is illustrated on Fig. 2 where the fisrt version is used.
|![GreedyFunction-2](https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/38a5a581-8f44-4e29-a36d-d8221799f625)|
|:---:|
|Figure 2: Greedy function algorithm as implemented in version 1 of the model.|

## Inference
The inference algorithm is used to predict the clusters from the hits. It is implemented in `main.py` by the `inference` function. As presented above, `greedy_func` was implemented to process events in minibatches. Thus after, having loaded the formatted hits in a `DataLoader`, `inference` will iterate over all minibatches to predict their clusters. Once all clusters were predicted for one minibatch, the next step is to translate all the quantities into their phyisical values. In particular charges and PDGs indices need to be translated and the energies need to be unnormalized and taken to the power 10. This is achieved by the `translate` function. 

>**NOTE:**
>Recall that the feature corresponding to the energies in both the labels and the feats is in reality a normalised version of the logarithm base 10 of the original values.




