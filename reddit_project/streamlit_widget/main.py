# python -m streamlit run c:/Users/Aspire/Documents/GA/Faaiz-Khan/project_3/streamlit_widget/main.py

import streamlit as st
import scrape
import eda
import lr_model
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Remove annoying warning
st.set_option('deprecation.showPyplotGlobalUse', False)


# FUNCTIONS

# Defining function to generate wordclouds from url
def genwc(df, nos=1):
    eda_df = eda.eda_mod(df, nos)
    # Run prediction function
    pred = lr_model.preds(eda_df)
    # Sum the word occurrences and sort by frequency
    word_counts = eda_df.sum().sort_values(ascending=False)
    # create & generate the WordCloud object
    cloud = WordCloud(min_word_length =3, width=800, height=800,
                        background_color='white').generate_from_frequencies(word_counts)
    #plot
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot()
    return pred.mean()

# Defining function to display thread stats on the widget
def printstats(url):
    df = scrape.scraper(url)
    st.write('Total ' + str(len(df)) + ' posts scraped')
    st.write('Thread title: ' + scrape.threadtitle(url))
    return df

# Defining function to plot the wordclouds for each url on widget
def printwc(df):
    st.subheader('Most common words:')
    score = genwc(df)
    st.subheader('Most common 2-grams:')
    genwc(df, 2)
    return score


# MAIN CODE

st.header("AI-bro or AI-anti?")
st.markdown("Visualising the most common words in 2 Reddit threads and predicting the prevailing sentiment in each thread towards Generative AI technology.")
    
# Ask for text or text file
url1 = st.text_input('Enter 1st Reddit url here:')
url2 = st.text_input('Enter 2nd Reddit url here:')

# Add a button feature
if st.button("Generate Wordclouds"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("URL 1:")
        df1 = printstats(url1)
        # st.write(scrape.mostvotes(df1))
    with col2:
        st.subheader("URL 2:")
        df2 = printstats(url2)
        # st.write(scrape.mostvotes(df2))

    col1, col2 = st.columns(2)
    with col1:
        score1 = printwc(df1)
    with col2:
        score2 = printwc(df2)

    # Compare the 2 scores
    compare = {'URL 1': score1, 'URL 2': score2}
    st.subheader('Overall, posts from ' + max(compare, key=compare.get)
                + ' favour Generative AI art more than posts from '
                + min(compare, key=compare.get) + '.')
        
