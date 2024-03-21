import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Define list of lemmatised stop words
add_stopwords = ['wa', 'ha', 'doe', 'did', 've', 'ca', 'll', 'gon', 'don', 'wan', 'na']

# Combine sklearn's stop word list with the above list
stopwords = [x for x in text.ENGLISH_STOP_WORDS]
new_stopwords = stopwords + add_stopwords

# Define the lemmatizer
lemmatizer = WordNetLemmatizer()

def eda_mod(df, nos):
    # Tokenize and lemmatize the text
    df['lemmatized_body'] = df['body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(str(x).lower())]))
    
    # Instantiate a CountVectorizer object
    vectorizer = CountVectorizer(stop_words=new_stopwords, ngram_range=(nos, nos))
    
    # Fit and transform the count vectorizer on the lemmatized body text
    X = vectorizer.fit_transform(df['lemmatized_body'])

    # Convert to a dataframe and densify
    word_count_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())

    return word_count_df





