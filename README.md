
# Fake News Detector
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Overview

This is a Fake News Detection model using Python.
Multinomial Naive Bayes algorithm will be used to train the fake news detection model, the algorithm is based on the Bayes theorem and is used for the analysis of the categorical text data. 
The model takes in 'Title' of any news to predict whether the news is fake or not.
 

 

## Documentation

[ScikitLearn](https://scikit-learn.org/stable/user_guide.html)     

[Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)   

[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)   

 
## Installation

To install the packages you can execute the following command:-


Install ScikitLearn

```bash
  !pip install scikit-learn
```

Install Numpy

```bash
  !pip install numpy
 ```
        
Install Pandas

```bash
  !pip install pandas
```        
## Deployment

For Fake News Detection place your News Title in news_headline

Input :
```bash
news_headline = "Place your News Title here"
FakeNews = cv.transform([news_headline]).toarray()
print(model.predict(FakeNews))

```
 
 Output :

 ```bash
 ['FAKE']

```

## Acknowledgements

 - [Using Bayesian Classifiers to detect Fake News](https://towardsdatascience.com/using-bayesian-classifiers-to-detect-fake-news-3022c8255fba)
 - [StackOverflow](https://stackoverflow.com)
  
