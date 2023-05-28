Importing the Libraries
import pandas as pd
import numpy as np
import nltk
Reading the dataset
dataset = pd.read_csv("/Users/deepthitamma/Downloads/test2.csv", sep="\t",encoding =
'cp1252', header=None)
dataset.head()
dataset.columns=['Text']
dataset
dataset['Text']
Importing the Regular Expressions Library
import re
import string
Removing the punctuations
def remove_punct(text):
text_nopunct = "".join([char for char in text if char not in string.punctuation])
return text_nopunct
dataset['Text_clean'] = dataset['Text'].apply(lambda x: remove_punct(x))
dataset.head()
Tokenizing the Text
def tokenize(text):
tokens = re.split('\W+', text)
return tokens
dataset['Text_tokenized'] = dataset['Text_clean'].apply(lambda x: tokenize(x.lower()))
dataset.head()
Removing the Stopwords
stopword = nltk.corpus.stopwords.words('english')
stopword
def remove_stopwords(tokenized_list):
text = [word for word in tokenized_list if word not in stopword]
return text
dataset['Text_nostop'] = dataset['Text_tokenized'].apply(lambda x: remove_stopwords(x))
dataset.head()
dataset['Text_nostop']
Exporting the Cleaned dataframe to .csv file
dataset.to_csv (r'/Users/deepthitamma/Desktop/A6_converted.csv', index = False,
header=True)