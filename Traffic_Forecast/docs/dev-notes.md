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

Results 1D-CNN:
* RMSE seed 523: With emb.: 0.0328, w/o emb.: 0.0333