import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
emotion_classifier_model = load_model('model_emotions.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the comments and music video transcripts
youtube_comments = pd.read_csv('youtube_comments_for_demo.csv')
youtube_mv_transcripts = pd.read_csv('youtube_mv_predicted_emotion.csv')

# Create a new column 'text_to_model' that replicates 'text'
youtube_comments['text_to_model'] = youtube_comments['text']

# Tokenize and pad sequences
youtube_comments['text_to_model'] = youtube_comments['text_to_model'].apply(lambda x: tokenizer.texts_to_sequences([x]))
youtube_comments['text_to_model'] = youtube_comments['text_to_model'].apply(lambda x: pad_sequences(x, maxlen=50))

# Predict labels and probabilities
predictions = emotion_classifier_model.predict(np.concatenate(youtube_comments['text_to_model'].values))
youtube_comments['label'] = predictions.argmax(axis=1)
youtube_comments['probability_sadness'] = predictions[:, 0]
youtube_comments['probability_joy'] = predictions[:, 1]
youtube_comments['probability_love'] = predictions[:, 2]
youtube_comments['probability_anger'] = predictions[:, 3]
youtube_comments['probability_fear'] = predictions[:, 4]

# Map labels to emotions
emotion_dict = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear'}
youtube_comments['emotion'] = youtube_comments['label'].map(emotion_dict)

# Streamlit application layout
st.title('Personalizing Music Video Recommendations with Emotional Intelligence')

# Let the user choose which row to use
row_index = st.selectbox('Demo', options=range(len(youtube_comments)), format_func=lambda x: f"Demo {x+1}")

# Display the comment and its emotion
selected_row = youtube_comments.iloc[row_index]
st.header("User makes comment after watching video")

# Display a different video based on the selected demo
if row_index == 0:
    st.video('https://www.youtube.com/watch?v=QCtEe-zsCtQ')
elif row_index == 1:
    st.video('https://www.youtube.com/watch?v=uS_y_65CcpA')

st.write(f"Username: {selected_row['author']}")
st.success(f"Comment: {selected_row['text']}")

st.markdown("---")

if st.button("Generate Emotion"):
    st.spinner()
    st.success(f"**Emotion: {selected_row['emotion']}**")
    if selected_row['emotion'] in ['Sadness', 'Anger', 'Fear']:
        user_emotion_prob = selected_row[['probability_sadness', 'probability_anger', 'probability_fear']].sum()
        st.info(f"**Sum of negative emotion probabilities: {user_emotion_prob}**")
    else:
        user_emotion_prob = selected_row['probability_' + selected_row['emotion'].lower()]
        st.info(f"**Emotion probability: {user_emotion_prob}**")

st.markdown("---")

# Recommend music videos based on the comment's emotion
def recommend_music_videos(user_comment, user_emotion, youtube_mv_transcripts, youtube_comments):
    # Define the emotions
    positive_emotions = ['Joy', 'Love']
    negative_emotions = ['Sadness', 'Anger', 'Fear']
    negative_probs = ['probability_' + emo.lower() for emo in negative_emotions]

    origin_video_id = user_comment['comment_origin_video_id']

    filtered_transcripts = youtube_mv_transcripts[youtube_mv_transcripts['video_id'] != origin_video_id]

    recommended_videos = None
    if user_emotion in positive_emotions:
        tree = KDTree(filtered_transcripts[['probability_' + user_emotion.lower()]])
        dist, idx = tree.query([[user_comment['probability_' + user_emotion.lower()]]], k=min(5, len(filtered_transcripts)))
        recommended_videos = filtered_transcripts.iloc[idx[0]]

    elif user_emotion in negative_emotions:
        # For negative emotions, calculate cosine similarity
        user_vector = user_comment[negative_probs].values.reshape(1, -1)
        
        # Filter out videos with sum of negative emotion probabilities that are not at least 0.1 than the user's comment
        filtered_transcripts = filtered_transcripts[
            (filtered_transcripts[negative_probs].sum(axis=1) < user_comment[negative_probs].sum() - 0.1)
        ]

        # Calculate cosine similarity
        filtered_transcripts['cosine_similarity'] = filtered_transcripts.apply(lambda row: cosine_similarity(user_vector, row[negative_probs].values.reshape(1, -1))[0][0], axis=1)

        # Find the 5 closest emotion probabilities
        if len(filtered_transcripts) > 0:
            recommended_videos = filtered_transcripts.nsmallest(5, 'cosine_similarity')
        else:
            st.write("There are currently no videos to recommend based on the user's current emotion level.")
            return

    # Sort the recommended videos by the probability of the user's emotion
    recommended_videos = recommended_videos.sort_values(by='probability_' + user_emotion.lower(), ascending=False)

    video_ids = recommended_videos['video_id'].tolist()
    st.subheader("Recommended Videos")
    for i, video_id in enumerate(video_ids):
        st.markdown(f"**Video ID {i+1}:** {video_id}")
        if user_emotion in negative_emotions:
            st.markdown(f"**Sum of negative emotion probabilities:** {recommended_videos.iloc[i][negative_probs].sum()}")
        else:
            st.markdown(f"**{user_emotion} Probability:** {recommended_videos.iloc[i]['probability_' + user_emotion.lower()]}")
        st.video(f'https://www.youtube.com/watch?v={video_id}')
        st.markdown("---")

def recommend_music_videos_by_index(index, youtube_mv_transcripts = youtube_mv_transcripts, youtube_comments = youtube_comments):
    # Get the user comment and emotion based on the index
    user_comment = youtube_comments.iloc[index]
    user_emotion = user_comment['emotion']
    # Call the original function with the fetched comment and emotion
    recommend_music_videos(user_comment, user_emotion, youtube_mv_transcripts, youtube_comments)

if st.button("Generate Music Video Recommendations"):
    st.spinner()
    recommend_music_videos_by_index(row_index)