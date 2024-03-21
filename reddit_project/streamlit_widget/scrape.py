# Importing libraries for data scraping
import pandas as pd
import praw

import re
from emoji import demojize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Unique identifier client_id and client_secret retrieved from personal application registered on Reddit.
'''
Note: You will have to fill in these identifier keys with your own set. Refer to below link:
https://praw.readthedocs.io/en/latest/getting_started/authentication.html
'''
reddit = praw.Reddit(user_agent="PRAW", 
                     client_id="", 
                     client_secret=""
                     )


# Define function to append post information to the dictionary
def dictapp(dict, post, op=False):
    if op:
        dict['is_op'].append(1)
        if post.selftext:
            dict['body'].append(post.title + ' ' + post.selftext)
        else:
            dict['body'].append(post.title)
    else:
        dict['is_op'].append(0)
        dict['body'].append(post.body)
    dict['author'].append(post.author)
    dict['num_comments'].append(replycnt(post, op))
    dict['upvotes'].append(post.score)
    dict['post_id'].append(post.id)


# Defining function to count replies to comment. This is used in dictapp() above.
def replycnt(comment, op):
    if op:
        reply_obj = comment.comments
    else:
        reply_obj = comment.replies
    count = 0
    for reply in reply_obj:
        count += 1
    return count

# Define list of known bots/automated posts between both subreddits
bot_list = ['DefendingAIArt-ModTeam',
            'AutoModerator',
            'WikiSummarizerBot',
            'BookFinderBot',
            'sneakpeekbot',
            'Anti-ThisBot-IB',
            'exclaim_bot',
            'of_patrol_bot',
            'AmputatorBot',
            'savevideobot',
            'RemindMeBot']

# Function to get subjectivity score using TextBlob
def get_subjectivity_score(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity

# Function to retrieve title of the thread
def threadtitle(link):
    submission = reddit.submission(url=link)
    return submission.title

# # Function to retrieve highest voted comment
# def mostvotes(df):
#     dataframe = df[df['is_op']==0]
#     topvote = dataframe.sort_values(by=['upvotes'], ascending=False).head(1).to_dict()
#     return('Most upvotes: ' + str(topvote['body'])
#            + '\n' + str(topvote['upvotes']) + ' upvotes')

'''
SCRAPER FUNCTION
'''
def scraper(link):
    # First defining/clearing the dictionary before the scraping process
    reddit_dict = {
                    'is_op': [],
                    'author': [],
                    'post_id': [],
                    'body': [],
                    'upvotes': [],
                    'num_comments': []}

    # Scraping process
    submission = reddit.submission(url=link)
    submission.comments.replace_more(limit=0)

    dictapp(reddit_dict, submission, op=True)
    for comment in submission.comments.list():
        dictapp(reddit_dict, comment)  

    # Creating dataframe and exporting to csv format
    reddit_df = pd.DataFrame(reddit_dict)

    '''
    CLEANING
    '''
    # Filtering out all rows where the the contents of the post are "[deleted]"
    reddit_df = reddit_df.loc[reddit_df['body']!='[deleted]',:].reset_index(drop=True)

    # Filtering out all rows where the the contents of the post are "[removed]"
    reddit_df = reddit_df.loc[reddit_df['body']!='[removed]',:].reset_index(drop=True)

    # Relabelling null entries under 'author' as "[deleted]"
    reddit_df['author'].fillna(value='[deleted]', inplace=True)

    # Filtering out the automated posts in our dataframe
    for bot in bot_list:
        reddit_df = reddit_df.loc[reddit_df['author']!=bot,:].reset_index(drop=True)

    # Define the string to search for
    string_to_search = 'this message was mass deleted/edited with redact.dev'

    # Filtering out the posts removed by redact.dev
    reddit_df = reddit_df[~reddit_df['body'].str.contains(re.escape(string_to_search))]

    # Checking updated dataframe
    reddit_df.info()

    # Using regex to search the 'body' and remove hyperlinks from posts
    reddit_df['body'] = reddit_df['body'].str.replace('http[^ ]*', '', regex=True)

    # Using regex to search the 'body' and remove giphy links from posts
    reddit_df['body'] = reddit_df['body'].str.replace('[!][[]gif[]][^ ]*', '', regex=True)

    # Applying the demojize function on 'body'
    reddit_df['body'] = reddit_df['body'].apply(demojize)

    # Replace NaN values with an empty string for consistency
    reddit_df['body'] = reddit_df['body'].fillna('')

    # Check if any rows in 'body' consist of only '0', '', or only newline character
    rows_with_zeros_or_newline = reddit_df[(reddit_df['body'].isin(['0', '', '\n', '\n\n', '\n\n\n', '\n\n\n\n']))]

    # Drop the rows with no actual text and reset the index
    reddit_df = reddit_df.drop(rows_with_zeros_or_newline.index).reset_index(drop=True)

    '''
    FEATURE ENGINEERING
    '''
    # Engineering feature for the character length of each post
    reddit_df['post_length'] = reddit_df['body'].str.len()

    # Engineering feature for the number of words in each post
    reddit_df['post_word_count'] = reddit_df['body'].str.split().str.len()

    # Instantiate Sentiment Intensity Analyzer
    sent = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to the 'body' column
    reddit_df['sentiment_scores'] = reddit_df['body'].apply(lambda x: sent.polarity_scores(x))

    # Expand the sentiment scores into separate columns
    sentiment_df = reddit_df['sentiment_scores'].apply(pd.Series)

    # Concatenate the sentiment scores DataFrame with the original DataFrame
    reddit_df = pd.concat([reddit_df, sentiment_df], axis=1)

    # Apply subjectivity analysis to the 'body' column
    reddit_df['subjectivity_score'] = reddit_df['body'].apply(get_subjectivity_score)

    # Drop unnecessary 'sentiment_scores' column after splitting
    reddit_df.drop(columns=['sentiment_scores'], inplace=True)

    return reddit_df