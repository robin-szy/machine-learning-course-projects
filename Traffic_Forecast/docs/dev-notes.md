# Data exploration

Tried to figure out, what column belongs to which feature:
- 0-9: historical sequence, 10 most recent data points
- 10-16: Week day
- ERROR: 17-41: Hours of day -> Column 39: Weird output, not binary, but integre with values of 5

In the paper, there are two separate roads but in the data, all nodes are connected.  
=> For our results, we should also receive two clusters

Pretty sure the 10th column is the most recent traffic data. 

Tried to make the one-hot encoded time and days a cyclical feature. That failed quite hard and showed the importance of making the hours and days independent by one-hot encoding. In a second experiment, I tried to use embeddings for these columns. This gave almost the same RMSE (0.0344 instead of 0.0340). However, no improvement, so did not keep it.

Tried TCN instead of 1D CNN. Much worse, RMSE around 0.0377. 

### Embedding the sensors
 Without sensor embedding, two sensors are assumed to be the similar if all
features like recent traffic history, road, lanes, direction, graph neighbors
are similar. But two sensors could still be different, even if these features look
similar. E.g. sensor A could be highway merge bottleneck and sensor B at suburban straight road
Embedding helps the model to learn: This sensor usually behaves like this.

Experiment over 10 seeds:
* Embedding always improves the model, for both versions (GRU-GCN and 1D CNN-GCN)
* 1D CNN-GCN usually is better than GRU-GCN with current configs. But we haven't fine-tuned the hyperparams yet.


Embedding the sensors together with the hours did not really improve anything. Only tested on one seed, though:
Original: train=0.0180 val_rmse=0.0328 val_mae=0.0212 best=0.0328 Embedding hour with sensor: epoch 193 : train=0.0171 val_rmse=0.0327 val_mae=0.0208 best=0.0327

### Attention

Using attention on CNN by a Conv1D layer with kernel size 1 did not improve RMSE. It made it a bit more interpretable, though, since we can clearly see below that the last 3 time steps matter most. This was probably already learned by previous Conv1D layer, and the attention just makes it more interpretable.

Average temporal attention weights over 10 timesteps:
[0.08451255 0.08451863 0.07820263 0.08952256 0.08872515 0.08806809
 0.08933216 0.10906468 0.14368138 0.14437218]


### Stratification

Stratification by traffic volume does not seem to improve overall RMSE.

          best_rmse  best_mae
no_strat   0.033843  0.021994  
q3         0.034507  0.022110  
q4         0.034115  0.021848

We'll stick with the old approach for the validation set.
