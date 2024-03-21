import streamlit as st
from streamlit import session_state as ss
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For Image-Text OCR
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # Replace this path based on installation of Tesseract OCR
model = joblib.load('Model_02B_model.pkl')
tokenizer = joblib.load('Model_02B_tokenizer.pkl')

if 'analysed' not in ss:
    ss['analysed'] = False


def process_img(img):
    texts = pytesseract.image_to_string(img)

    return texts
    
def classify_text(text):
    text = pd.Series([text])
    max_len = 34
    # Tokenise as done for the model training
    text_processed = tokenizer.texts_to_sequences(text)
    text_padded = pad_sequences(text_processed, max_len)
    result = model.predict(text_padded)
    pred_class = (result>=0.5).astype(int)
    return pred_class

# Streamlit application layout
st.title('Hate/Offensive Speech Detection by CloverWorks')

tab1, tab2, tab3 = st.tabs(['Image-Text Detection', 'Find Hate/Offensive Users', 'Get Hate/Offensive Posts'])

with tab1:
    st.header('Classify Text from Image')
    image_file = st.file_uploader("Upload an image (png/jpg)", accept_multiple_files=False, type=['png', 'jpg'])
    if image_file is not None:
        with st.spinner("Processing the image..."):
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes,1)
            text_from_img = process_img(img)
            class_result = classify_text(text_from_img)
            class_list = ['not hateful/offensive', 'hateful/offensive']
            st.write(text_from_img)
            st.markdown("""---""") 
            st.write('The text from this image is ', class_list[int(class_result)])



with tab2:
    with st.form(key='analyse_form'):
        st.header('Subreddit Analyser')
        #subreddit_input = st.text_input("Enter Subreddit Name", "", key='sub_name')
        st.write('Get the 10 users with the most potentially hateful/offensive posts')
        analyse_submit = st.form_submit_button(label='Analyse')
        if analyse_submit:
            start_time = time.process_time()
            with st.spinner("Scraping the Subreddit"):
                # Load the data from csv
                df = pd.read_csv('clean_merged.csv').head(500)

            with st.spinner("Analysing Comments..."):
                # Run model prediction
                df['classification'] = df['comment_text'].apply(lambda x: classify_text(str(x)))
                
            with st.spinner("Visualising Results..."):
                df_class1 = pd.DataFrame(df[df['classification']==1]['author_name'].value_counts()).reset_index().rename(columns={'count':'hate_offensive'}).head(10)

                st.pyplot(df_class1.plot(kind='barh', x='author_name', stacked=True, figsize=(10, 6)).figure)
                ss['total'] = df_class1
                ss['ranked_users'] = df_class1.sort_values(by='hate_offensive', ascending=False)['author_name'].tolist()
                ss['df'] = df
                ss['analysed'] = True
            st.write('Time taken: ', time.process_time() - start_time)

with tab3:
    if ss['analysed']:
        with st.form('user_posts_form'):
            st.header('View Hate/Offensive Posts')
            user_picker = st.selectbox("Get hate/offensive posts from user", ss['ranked_users'])
            submitted = st.form_submit_button("Get posts")
            if submitted:
                #with st.spinner('Getting posts...'):
                #st.dataframe(ss['df'][ss['df']['author_name'] == user_picker])
                df_show = ss['df'][(ss['df']['author_name'] == user_picker) & ((ss['df']['classification'] == 0) | (ss['df']['classification'] == 1))]
                df_show = df_show[['classification', 'comment_text', 'comment_score']]
                df_show['classification'] = 'hateful/offensive'
                st.dataframe(
                    df_show,
                    column_config={
                        'classification': 'Classification',
                        'comment_text' : 'Comment',
                        'comment_score': 'Reddit Comment Karma'
                    },
                    hide_index=True,
                    use_container_width=True
                )

