# Vaccine sentiment model
This project is about the refinement of a vaccine sentiment model I developed as part of my dissertation project.

The vaccine sentiment model aims to classify Tweets into four categories:
- Category 0 :  Not relevant or an article headline connected to a link
- Category 1 : Vaccine hesitant
- Category 2 : Anti-vaccination / negative sentiment towards vaccines
- Category 3 : Pro-vaccination / positive sentiment towards vaccines

The Tweets in the vaccine_tweets_labelled.csv file were manually labelled myself and are therefore subject to my interpretation. 

## Quick summary
At the moment the model currently achieves an accuracy around 60 - 63%. I am looking to improve on this.

The sentiment model is based on a Recurrent Neural Network architecture using Bi-directional Long Short-Term Memory layers, and dropout layers.

As the dataset is imbalanced, largely towards to category 0, the dataset is resampled using RandomOverSampler to ensure that all classes have the same number of occurrences.

The Tweets have been transformed into another format for input into the model. The Tweets are turned into an array of numbers based on the words being tokenised. The Tweets are then padded to ensure that they are all of the same length. 

The model is then fitted over 100 epochs with validation loss as the early stopping and callback metric.
