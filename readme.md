# What videos do you love to take? - Personal video classification with deep learning
Final Project, MIT 15.773 - Hands-on Deep Learning <br>
Team Members: [Jason Jia](https://www.linkedin.com/in/jasonjiajs/), [Maxime Wolf](https://www.linkedin.com/in/maxime-wolf/), [Maria Besedovskaya](https://www.linkedin.com/in/mariabesedovskaya/), [Minnie Arunaramwong](https://www.linkedin.com/in/minnie-arunaramwong/)

<img src="https://github.com/jasonjiajs/videos-you-love-to-take/assets/90637415/b65379e5-0225-4948-94eb-984d70efe257" width="400" />

## Problem and Motivation
Videos help capture some of our favorite and most important moments, but we often lose track of what we took in the past. Currently, available systems (e.g., application Photo for IOS) can automatically categorize and group photos, but they do not automatically organize videos, and manual sorting is excessively time-consuming. We want to automatically categorize videos and make it easier for people to find what they want. 

## Task Details

### Data Source

To address this challenge, we leveraged the extensive Youtube-8M dataset, which comprises around 8 million videos randomly sampled from publicly available YouTube content (https://research.google.com/youtube8m/index.html). 

To download our frame-level training files, run the following code (note that the full frame-level dataset is 1.5TB): 
```shell
curl data.yt8m.org/download.py | shard=1,300 partition=2/frame/train mirror=us python
```

Our focus was on a specific subset of this dataset aligned with our objective and focusing on the most common categories of personal videos, including those featuring food, sport, nature, and family themes.

Our goal is to correctly label videos from the following categories: “Vehicle”, “Concert”, “Dance”, “Football”, “Animal”, “Food”, “Outdoor recreation”, “Nature”, “Mobile phone” and “Cooking”.

We took the first 45s of each video and recorded 1 frame per second. 

### RGB to Embeddings

Given the large size of RGB data (e.g. 720x720x3 for each frame of a 720p video), we transformed RGB vectors first into 2048-sized embedding vectors using the Inception TensorFlow model. We then compressed them into 1024-sized embedding vectors using Youtube-8M’s PCA matrix and applied quantization (1 byte per coefficient), reducing the size of the data by a factor of 8. This follows the methodology laid out by the Youtube-8M team. 

### Data Format
The data is frame-level and is of the following format:
Context
Video id
Labels (categories)
Features
RGB: 1024-dimension embedding vector for each frame
Audio: 1024-dimension embedding vector for each frame (not used in analysis)

### Label prediction

Our input is a vector of size 1024 x 45.
Our output layer is a vector of size 10, each corresponding to one of the 10 labels.
For each node in the output layer, the output is 1 if the category appears in the video and 0 otherwise. A video can be assigned to multiple categories.

## Models 
We tried out various models for this multi-label classification task.

### Baseline
Given that 29% of the dataset is categorized under the label “Vehicle” (the largest class of the data), setting a baseline prediction for all videos as “Vehicle” would result in a minimum accuracy of 29%. This approach establishes a foundational benchmark for evaluating the performance of more sophisticated predictive models against this initial accuracy threshold.

### Simple RNN model:
We created a model with 2 simple RNN layers with 64 nodes in each layer to predict the label. The model performed slightly better than the baseline at around 35.2% validation accuracy. The initial Simple RNN model's low accuracy can be attributed to several factors. Firstly, Simple RNNs may struggle with capturing long-term dependencies in sequential data, limiting their ability to effectively learn complex patterns. Additionally, with only two layers of 64 nodes each, the model may lack the capacity to sufficiently represent the intricate relationships present in the data. Furthermore, the absence of regularization techniques such as dropout may lead to overfitting, causing the model to generalize poorly to unseen data, as we can see the drop in the model’s performance from training to validation.

### RNN and LSTM model
We also built an architecture using both RNN and LSTM layers. The full architecture is shown below. We leverage both spatial features and temporal features by using CNN blocks and LSTM layers in parallel. We then concatenate the embeddings into a single vector that is passed into several dense layers.

Figure 1. NN architecture using both RNN and LSTM layers

The model’s performance was significantly better when combining CNN and LSTM layers, which are capable of capturing long-term dependencies in sequential data. Additionally, the increased complexity of the model architecture, particularly with the inclusion of multiple LSTM layers and additional dense layers, provides the model with a higher capacity to learn and represent complex patterns present in the data. Moreover, the model demonstrated high validation accuracy, indicating the effectiveness of dropout regularization in preventing overfitting.

## Results



