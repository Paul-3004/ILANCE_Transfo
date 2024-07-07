# Results 
This document presents a non exhaustive list of initial results obtained with the different versions. 

To analyse the performances of the models, several quantities are computed. 
1. Excess prediction of the number of clusters: this represents difference between the number of predicted clusters and the number of labels clusters.
2. Relative error on the energy
3. Charge accuracy: obtained by making a one-to-one comparison between predictions and labels. If correct a value of +1 is given, else 0, the average is taken for each event.
4. PDGs accuracy, obtained by making a one-to-one comparison between predictions and labels. If correct a value of +1 is given, else 0, the average is taken for each event.
5. Angle $\theta$: obtained by computing the scalar product between prediction and labels directions, establishing a one-to-one correspondance. Since by construction the direction vectors are unitary for both labels and predictions, the anlge is easily retrieved.

>**NOTE:**
>As stated performances are always based on a one-to-one correspondance between the predictions and the labels. The $i^{th}$ clusters of the $j^{th}$ event in the prediction will be compared with $i^{th}$ clusters of the $j^{th}$ event in the labels. All the quantities, except for the excess prediction of clusters, are thus computed only on the overlap between prediction and labels.

## Tau dataset 
### Version 1
The following results were obtained on the tau datasets with the first implementation of the transformer. Before each graph is a table summarising the values of the different hyperparameters. A fraction of 2% of the training set was used and the model was trained for 10 epochs. Since recordings of the losses were made 10 per epoch, each 10 marks on the evolution graphs correspond to 1 epoch.

|<img width="576" alt="Screen Shot 2024-07-08 at 05 38 48" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/5f44fed3-9edd-4645-a8de-3cfcae3d926a">|
|:---:|
|Figure a: Values of the hyperparameters. For the weights of the loss function, the order is charge, PDG, cont. DOF|
|<img width="727" alt="Comparison between predictions and labels for the tau dataset using model 1" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/5cfe27a6-cf70-4709-a464-5f8a4d11f591">|
|Figure b: Comparison between predictions and labels for the tau dataset, using the first implementation of the model.|
|<img width="768" alt="Evolution of the training and validation losses." src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/03066cb0-dec1-4705-b1b9-6cd360b2fba0">|
|Figure c: Evolution of the training and validation losses. For the figure with 3 loss functions, these correspond to the individual training losses of the total train loss.|

Adding more layers of encoders and decoders as well as increasing the weights on the charge loss function to try to decrease the excess predictions of number of clusters:
|<img width="621" alt="Screen Shot 2024-07-08 at 05 42 14" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/38eca63e-198a-4a4c-9328-7cb55bce42a1">|
|:---:|
|Figure a: Values of the hyperparameters. For the weights of the loss function, the order is charge, PDG, cont. DOF|
|<img width="723" alt="Comparison between predictions and labels for the tau dataset, model 1" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/aae08a14-fca5-404b-853f-2ea2a4440084">|
|Figure b: Comparison between predictions and labels for the tau dataset, using the first implementation of the model.|
|<img width="768" alt="Evolution of the training and validation losses" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/e8336686-7e1e-42e6-ae37-c16135222835">|
|Figure c: Evolution of the training and validation losses. For the figure with 3 loss functions, these correspond to the individual training losses of the total train loss.|

### Version 2
Results obtained using the second implementation of the model, on 20% of the testing set, for a maximum of 10 epochs.
|<img width="553" alt="table comparing averaged performances of model 1 and 2" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/53cf03a3-4c07-4a07-904f-9787f7a10151">|
|:---:|
|<img width="829" alt="Comparison between the excess predictions of model 1 and 2 and evolution of the loss functions for the model 2." src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/00c72881-d741-4571-9199-31ba31bbd9e2">|
|Figure : Comparison between the excess predictions of model 1 and 2 and evolution of the loss functions for the model 2.|

# Photons
Obtained using 100% of the training set for both the 1 and 2 photons datasets. For the 1 photon dataset, the first model was used, with all the changes brought by the implementation of the loss function specific to the kind of tokens, as well as energy sorting and threshold. The results for the 2 photons dataset were obtained with the second model with the same changes. 
|<img width="720" alt="comparison photon 1 and 2, averaged over 2000 events" src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/65b09fa5-dab5-4510-b054-a1e20e899843">|
|:---:|
|<img width="970" alt="Comparison between model performances with the 1 and 2 photons dataset." src="https://github.com/Paul-3004/ILANCE_Transfo/assets/77359118/0fe13faa-d9a8-42ac-a219-668fbbc28fce">|
|Figure: Comparison between model performances with the 1 and 2 photons dataset.|
