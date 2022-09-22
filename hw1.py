import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

import ssl
import os
import contractions
here = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(here, 'amazon_reviews_us_Jewelry_v1_00.tsv')
output_file = "output.txt"
#nltk.download('wordnet')
import re
from bs4 import BeautifulSoup


def main():

    #============================================================================================================
    #
    # Part 1 & 2: Dataset Prep / Cleaning
    #
    # ============================================================================================================

    data = pd.read_table('/Users/ryanluu/git/CSCI544-Sentiment_Analysis/test.tsv', usecols=['star_rating', 'review_body'],
                        low_memory=False)

    data['review_body'].fillna('', inplace=True)

    sampled_amazon_df = data

    # data = pd.read_table(filename, usecols=['star_rating', 'review_body'], low_memory=False)
    #
    # # Fills in NaN's
    # data['review_body'].fillna('', inplace=True)
    #
    # sampled_amazon_df = data.sample(n=20000)

    # Prints Initial length of reviews before Data Cleaning
    #mean_len = sampled_amazon_df['review_body'].apply(len).mean()
    #print("This is the average length of reviews before cleaning: " + str(mean_len))

    # Lower Case
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].str.lower()

    # Fixes Contractions
    sampled_amazon_df['fix_contractions'] = sampled_amazon_df['review_body'].apply(lambda l: [contractions.fix(word) for word in l.split()])
    sampled_amazon_df['review_body'] = [' '.join(map(str, k)) for k in sampled_amazon_df['fix_contractions']]
    del sampled_amazon_df['fix_contractions']

    # Removes HTML, HTML Tags, and URL's
    sampled_amazon_df["review_body"] = sampled_amazon_df["review_body"].str.replace('<[^<]+?>', '', regex=True).str.strip()
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)

    # Removes all non-alphabetical chars
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].str.replace('[^a-zA-Z]', ' ', regex=True)

    # removes extra spaces
    sampled_amazon_df['review_body'] = sampled_amazon_df['review_body'].replace(r'\s+', ' ', regex=True)

    # Prints Initial length of reviews before Data Cleaning
    #mean_len = sampled_amazon_df['review_body'].apply(len).mean()
    #print("This is the average length of reviews after cleaning: " + str(mean_len))

    # ============================================================================================================
    #
    # Part 3: Preprocessing with NLTK
    #
    # ============================================================================================================

    stop_words = set(stopwords.words('english'))
    regexp = RegexpTokenizer('\w+')

    # Creates column of NLTK tokens
    sampled_amazon_df["nltk_tokens"] = sampled_amazon_df["review_body"].apply(regexp.tokenize)

    # Removes the stop words
    sampled_amazon_df['nltk_tokens'] = sampled_amazon_df['nltk_tokens'].apply(lambda x: [word for word in x if word not in stop_words])

    # Gets Parts of Speech for each word
    sampled_amazon_df['part_of_speech_tags'] = sampled_amazon_df['nltk_tokens'].apply(nltk.tag.pos_tag)






    sampled_amazon_df.to_csv("out.csv")
    print(sampled_amazon_df)



main()