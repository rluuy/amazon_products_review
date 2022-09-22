## amazon_products_review

The purpose of this assignment is to experiment with text representations and how to use text representations for sentiment analysis. It takes an dataset of amazon reviews (~ 16 million reviews) and trys to train some classifiers to predict a products rating (1-5 stars) based on the reviews written.

E.g) 4-5 star reviews will typically have words like: Great! Good! Happy!

 1-2 Star reviews will have words like: Bad! Yuck! Gross! 

The code flow is like so: -> Reading in file into pandas -> preprocessing on dataset (removing html,punctuation, numbers, etc) -> extracting the tfidf features -> Training Preceptron, SVM, Logisitc Regression, and Multinomial Naive Bayes classifiers
