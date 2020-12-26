# Toxic Messages Classifier

## Description:
The project is aimed towards creating an automatic moderation tool for text-based communication. It analyzes the text that a user has input, and classifies it against six different categories of unacceptable behaviour. If the comment is classified as a part of any of them, it is flagged as inappropriate, and is automatically censored. The dataset used to train the model is a collection of Wikipedia user comments, with a binary classification for each category.

The model is learnt using Linear Regression. The entire corpus is first vectorized into TF-IDF format, and then the numeric representation of each sentence is used to adjust the weights of the regression model. We use gradient descent to minimize the error between our predicted outcome and the real outcome, and perform this for multiple epochs to achieve the closest value of weights that represent the underlying pattern in the data.

## Requirements:
- Library: streamlit, chatterbot, nltk, sklearn, pandas, numpy
- Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data




