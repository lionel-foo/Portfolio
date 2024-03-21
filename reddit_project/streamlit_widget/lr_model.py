import pandas as pd
import dict
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords

# Load data:
reddit = pd.DataFrame(dict.dict)

# additional stop words to remove
additional_stop_words = ['wa', 'ha', 'doe', 'did', 've', 'ca', 'll', 'gon', 'don', 'wan', 'na']

# Combine native 'english' stop words with additional stop words
all_stop_words = list(set(ENGLISH_STOP_WORDS).union(additional_stop_words))

# Creating X (features) and y (target)
X = reddit['body']  # Features
y = reddit['subr-def_ai']  # Target

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Define a function to perform lemmatization on a text
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Apply lemmatization to all rows in X
X_lemmatized = X.apply(lemmatize_text)

# Perform train-test split with 20% test size and stratify with y
X_train, X_test, y_train, y_test = train_test_split(X_lemmatized, y, test_size=0.2, stratify=y, random_state=42)

pipeline_lr = Pipeline([
    ('cvec', CountVectorizer(lowercase=True, stop_words=all_stop_words,
                             max_df=0.4, max_features=5000,
                             min_df=3, ngram_range=(1, 3),
                             )),
    ('lr', LogisticRegression(max_iter=5000, random_state=42,
                              C=1, penalty='l2'))
])

pipeline_lr.fit(X_train, y_train)

def preds(df):
    return pipeline_lr.predict(df)
