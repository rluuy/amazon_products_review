import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import itertools
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import ssl
import os
import contractions
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'amazon_reviews_us_Jewelry_v1_00.tsv')
output_file = "output.txt"
#nltk.download('wordnet')
import re
from bs4 import BeautifulSoup


'''
CSCI544 Assignment 1: Sentiment Analysis on Amazon Reviews Assignment. 

@author: Ryan Luu 

@date: 9/8/2022


The purpose of this assignment is to experiment with text representations and how to use text representations 
for sentiment analysis. It takes an dataset of amazon reviews (~ 16 million reviews) and trys to train some 
classifiers to predict a products rating (1-5 stars) based on the reviews written. 

E.g) 4-5 star reviews will typically have words like: Great! Good! Happy! 

     1-2 Star reviews will have words like: Bad!Yuck! Gross! 

The code flow is like so: 
  -> Reading in file into pandas
  -> preprocessing on dataset (removing html,punctuation, numbers, etc)
  -> extracting the tfidf features 
  -> Training Preceptron, SVM, Logisitc Regression, and Multinomial Naive Bayes classifiers 

'''

def main():

    #============================================================================================================
    #
    # Part 1: Dataset Prep
    #
    # ============================================================================================================

    data = pd.read_table(filename, usecols=['star_rating', 'review_body'], low_memory=False)

    data = data.dropna()                # Gets rid of NaN's in Table
    data = data.reset_index(drop=True)  # Resets the index

    # Assigns ratings to int
    data['star_rating'] = data['star_rating'].astype(int)
    data = data.sample(frac=1).reset_index(drop=True)

    # Adds 20000 reviews with 1-5 star reviews. Total is balanced 100,000 reviews.

    sampled_amazon_df = data[data['star_rating'] == 1][:20000]
    sampled_amazon_df = sampled_amazon_df.append(data[data['star_rating']== 2] [:20000])
    sampled_amazon_df = sampled_amazon_df.append(data[data['star_rating'] == 3][:20000])
    sampled_amazon_df = sampled_amazon_df.append(data[data['star_rating'] == 4][:20000])
    sampled_amazon_df = sampled_amazon_df.append(data[data['star_rating'] == 5][:20000])
    sampled_amazon_df = sampled_amazon_df.reset_index(drop=True)


    #============================================================================================================
    #
    # Part 2: Cleaning
    #
    # ============================================================================================================

    # Prints Initial length of reviews before Data Cleaning
    pre_clean_avg = get_avg_review_len('review_body', 'data cleaning:', False, sampled_amazon_df)


    # Lower Case
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].str.lower()

    # Fixes Contractions
    sampled_amazon_df['fix_contractions'] = sampled_amazon_df['review_body'].apply(lambda l: [contractions.fix(word) for word in l.split()])
    sampled_amazon_df['review_body'] = [' '.join(map(str, k)) for k in sampled_amazon_df['fix_contractions']]
    del sampled_amazon_df['fix_contractions']

    # Removes HTML, HTML Tags, and URL's.
    sampled_amazon_df["review_body"] = sampled_amazon_df["review_body"].str.replace('<[^<]+?>', '', regex=True).str.strip()
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)

    # Removes all non-alphabetical chars
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].str.replace('[^a-zA-Z]', ' ', regex=True)

    # removes extra spaces
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].replace(r'\s+', ' ', regex=True)

    # Prints Initial length of reviews after Data Cleaning
    post_clean_avg = get_avg_review_len('review_body', 'data cleaning:', True, sampled_amazon_df)

    print(pre_clean_avg + "," + post_clean_avg)


    # ============================================================================================================
    #
    # Part 3: Preprocessing with NLTK
    #
    # ============================================================================================================

    pre_proc_avg = get_avg_review_len('review_body', 'removing stop words and lemmization:', False, sampled_amazon_df)

    stop_words = set(stopwords.words('english'))
    regexp = RegexpTokenizer('\w+')
    wnl = WordNetLemmatizer()

    # Creates column of NLTK tokens
    sampled_amazon_df["nltk_tokens"] = sampled_amazon_df["review_body"].apply(regexp.tokenize)

    # Removes the stop words
    sampled_amazon_df['nltk_tokens'] = sampled_amazon_df['nltk_tokens'].apply(lambda x: [word for word in x if word not in stop_words])

    # Gets Parts of Speech for each word
    sampled_amazon_df['part_of_speech_tags'] = sampled_amazon_df['nltk_tokens'].apply(nltk.tag.pos_tag)

    # Creates column of Wordnet POS tokens
    sampled_amazon_df['wordnet_part-of_speech_tags'] = sampled_amazon_df['part_of_speech_tags'].apply(lambda x: [(word,word_net_pos_converter(pos_tag)) for (word,pos_tag) in x])

    # Lemmatizes the words with NLTK wordnetlemmatizer
    sampled_amazon_df['lemmatized_reviews'] = sampled_amazon_df['wordnet_part-of_speech_tags'].apply(lambda x: " ".join([wnl.lemmatize(word,pos_tag) for word,pos_tag in x]))

    post_proc_avg = get_avg_review_len('nltk_tokens', 'removing stop words and lemmization:', True, sampled_amazon_df)

    print(pre_proc_avg + "," + post_proc_avg)

    # ============================================================================================================
    #
    # Part 4: Feature Extraction
    #
    # ============================================================================================================

    tf_idf_vectorizer = TfidfVectorizer()

    X_train, X_test, Y_train, Y_test = train_test_split(sampled_amazon_df['lemmatized_reviews'], sampled_amazon_df['star_rating'], test_size = 0.20, random_state = 727)

    #print("Train: ", X_train.shape, Y_train.shape,"Test: ", (X_test.shape, Y_test.shape))

    tf_x_train = tf_idf_vectorizer.fit_transform(X_train)
    tf_x_test = tf_idf_vectorizer.transform(X_test)

    # ============================================================================================================
    #
    # Part 5: Perceptron
    #
    # ============================================================================================================

    p_classifier = Perceptron(tol=1e-3, random_state=727)
    p_classifier.fit(tf_x_train, Y_train)
    y_test_pred = p_classifier.predict(tf_x_test)
    report = classification_report(Y_test, y_test_pred, output_dict=True)
    p_df = pd.DataFrame(report).transpose()
    print(p_df.loc['1', "precision"], end=',')
    print(p_df.loc['1', "recall"], end=',')
    print(p_df.loc['1', "f1-score"], end='\n')
    print(p_df.loc['2', "precision"], end=',')
    print(p_df.loc['2', "recall"], end=',')
    print(p_df.loc['2', "f1-score"], end='\n')
    print(p_df.loc['3', "precision"], end=',')
    print(p_df.loc['3', "recall"], end=',')
    print(p_df.loc['3', "f1-score"], end='\n')
    print(p_df.loc['4', "precision"], end=',')
    print(p_df.loc['4', "recall"], end=',')
    print(p_df.loc['4', "f1-score"], end='\n')
    print(p_df.loc['5', "precision"], end=',')
    print(p_df.loc['5', "recall"], end=',')
    print(p_df.loc['5', "f1-score"], end='\n')
    print(p_df.loc['weighted avg', "precision"])

    #print(p_df)


    # ============================================================================================================
    #
    # Part 6: SVM
    #
    # ============================================================================================================

    svm_classifier = LinearSVC(random_state=727)
    svm_classifier.fit(tf_x_train, Y_train)
    y_test_pred = svm_classifier.predict(tf_x_test)


    report = classification_report(Y_test, y_test_pred, output_dict=True)
    svm_df = pd.DataFrame(report).transpose()
    print(svm_df.loc['1', "precision"], end=',')
    print(svm_df.loc['1', "recall"], end=',')
    print(svm_df.loc['1', "f1-score"], end='\n')
    print(svm_df.loc['2', "precision"], end=',')
    print(svm_df.loc['2', "recall"], end=',')
    print(svm_df.loc['2', "f1-score"], end='\n')
    print(svm_df.loc['3', "precision"], end=',')
    print(svm_df.loc['3', "recall"], end=',')
    print(svm_df.loc['3', "f1-score"], end='\n')
    print(svm_df.loc['4', "precision"], end=',')
    print(svm_df.loc['4', "recall"], end=',')
    print(svm_df.loc['4', "f1-score"], end='\n')
    print(svm_df.loc['5', "precision"], end=',')
    print(svm_df.loc['5', "recall"], end=',')
    print(svm_df.loc['5', "f1-score"], end='\n')
    print(svm_df.loc['weighted avg', "precision"])

    # ============================================================================================================
    #
    # Part 7: Logistic Regression
    #
    # ============================================================================================================

    lr_classifier = LogisticRegression(max_iter=1000, solver='saga', random_state=727)
    lr_classifier.fit(tf_x_train, Y_train)
    y_test_pred = lr_classifier.predict(tf_x_test)

    report = classification_report(Y_test, y_test_pred, output_dict=True)
    lr_df = pd.DataFrame(report).transpose()
    print(lr_df.loc['1', "precision"], end=',')
    print(lr_df.loc['1', "recall"], end=',')
    print(lr_df.loc['1', "f1-score"], end='\n')
    print(lr_df.loc['2', "precision"], end=',')
    print(lr_df.loc['2', "recall"], end=',')
    print(lr_df.loc['2', "f1-score"], end='\n')
    print(lr_df.loc['3', "precision"], end=',')
    print(lr_df.loc['3', "recall"], end=',')
    print(lr_df.loc['3', "f1-score"], end='\n')
    print(lr_df.loc['4', "precision"], end=',')
    print(lr_df.loc['4', "recall"], end=',')
    print(lr_df.loc['4', "f1-score"], end='\n')
    print(lr_df.loc['5', "precision"], end=',')
    print(lr_df.loc['5', "recall"], end=',')
    print(lr_df.loc['5', "f1-score"], end='\n')
    print(lr_df.loc['weighted avg', "precision"])

    # ============================================================================================================
    #
    # Part 8 Multinomial Naive Bayes
    #
    # ============================================================================================================

    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(tf_x_train, Y_train)
    y_test_pred = mnb_classifier.predict(tf_x_test)

    report = classification_report(Y_test, y_test_pred, output_dict=True)
    mnb_df = pd.DataFrame(report).transpose()
    print(mnb_df.loc['1', "precision"], end=',')
    print(mnb_df.loc['1', "recall"], end=',')
    print(mnb_df.loc['1', "f1-score"], end='\n')
    print(mnb_df.loc['2', "precision"], end=',')
    print(mnb_df.loc['2', "recall"], end=',')
    print(mnb_df.loc['2', "f1-score"], end='\n')
    print(mnb_df.loc['3', "precision"], end=',')
    print(mnb_df.loc['3', "recall"], end=',')
    print(mnb_df.loc['3', "f1-score"], end='\n')
    print(mnb_df.loc['4', "precision"], end=',')
    print(mnb_df.loc['4', "recall"], end=',')
    print(mnb_df.loc['4', "f1-score"], end='\n')
    print(mnb_df.loc['5', "precision"], end=',')
    print(mnb_df.loc['5', "recall"], end=',')
    print(mnb_df.loc['5', "f1-score"], end='\n')
    print(mnb_df.loc['weighted avg', "precision"])


# Helper Function to convert treebag POS to wordnet POS. No need for ADJ_SAT since we're gooing from POS to wordnet
def word_net_pos_converter(pos_tag):
    if pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN     # word_net defaults to Noun otherwise

# Helper function to print out average length of review before and after processing
def get_avg_review_len(col, step, flag, df):
    mean_len = df[col].apply(len).mean()
    return str(mean_len)

    # if flag == False:
    #     print("This is the average length of reviews before " + step + " " + str(mean_len))
    # else:
    #     print("This is the average length of reviews after " + step + " " + str(mean_len))


main()