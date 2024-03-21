# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 
Capstone Project: Personalizing Music Video Recommendations with Emotional Intelligence

> SG-DSI-41: Lionel Foo

---

### Problem Statement

My objective is to design an emotion-centric music video recommendation system. The proposed system will analyze YouTube users’ comments to discern their emotional state and subsequently recommend Youtube music videos. These recommendations aim to either resonate with the users’ current emotional state or aid them in navigating through negative emotions.

The system will prioritize user well-being by suggesting songs that harmonize with their current mood. It will evaluate users’ emotional needs based on their comments and recommend the most “beneficial” music videos instead of the most “matched” ones. A beneficial music video is defined by two criteria:
* It aligns with the user’s emotional state, thereby fulfilling their emotional needs.
* If the user’s overall mood is negative, the recommended music video should have the potential to uplift the user’s mood. This implies that the music should be slightly more positive than the user’s current mood. However, to prevent causing emotional discomfort or resistance to the recommendation, the degree of positivity should be delicately balanced.

Specifically, this project will have 2 parts:
#### Part 1: Build an emotion centric text based classifier to distinguish text betweem 5 different types of emotions - 
1. Sadness
2. Joy
3. Love
4. Anger
5. Fear

We will explore 5 different models specifically RNN with Glove Word Embeddings, Fine Tuning Distilbert, Naive Bayes, Logistic Regression and XG Boost.
Since we are placing equal importance in correctly identifying the correct emotion class, both the recall and precision scores are important to us. The F1-score is particularly useful if we want to have a balance between identifying as many instances of each emotion as possible (high recall) and keeping the number of incorrect predictions low (high precision). In addition, the overall accuracy of our detection model is of interest. As a result, our models will be evaluated on their (a) f1-score and (b) accuracy (c) efficiency

#### Part 2: Developing a rule-based recommender system that matches the emotions of each user comment to a list of 5 music videos. 

The rules for recommending the music videos differ based on whether the user is deemed to have positive or negative emotions based on their comments.

In terms of deliverables, we intend to have a youtube music video recommender system that will evaluate users’ emotional needs based on their comments and recommend the most “beneficial” music videos instead of the most “matched” ones. This approach ensures that the system is not just recommending music videos, but also contributing to the users’ emotional well-being.

### Organization

The code notebooks will be grouped into 5 main types based on their sequential order, as follows:

1. 01_EDA_&_Data_Preprocessing
2. 02A_Classification_Model(Multiclass Classification - DistilBert)
3. 02B_Classification_Model(Multiclass Classification - Naive Bayes, Logistic Regression, XGBoost)
4. 02C_Classification_Model(Multiclass Classification - RNN with GloVe Word Embeddings)
5. 03A_Scraping_Youtube_MV_Transcripts
6. 03B_Scraping_Youtube_Comments
7. 04_Recomender_System_Modeling
8. Demo

### Data

For the data, we will be using a dynamically generated Emotions dataset from Kaggle. This is a collection of English Twitter messages meticulously annotated with six fundamental emotions classifications:
1. Sadness
2. Joy
3. Love
4. Anger
5. Fear
6. Surprise

### Exploratory Data Analysis

The EDA process mainly focused on looking at substrings to remove through regex cleaning, looking at the most common words for each emotion and analysing the distribution of the emotion classes

For data cleaning process, performed the following:
1. Convert Contraction words
2. Removed Website urls related words
3. Removed unecessary white spaces
4. Drop rows with only 1 words
5. Drop rows with duplicate comments

In terms of the Word Cloud generated, each class of emotions had distinct common words used.

Based on our analysis of the distribution of emotion classes. Surprise emotion was significantly less common in the dataset than the other emotions. We decided to balance the dataset by removing all rows with Surprise class. This is due to the following reasons
* to address the significant class imbalance
* Surprise is a transient emotion
* Surprise can have any valence


### Modeling

### Emotion Centric Multiclass Classification 

For our study, we began with a dataset that had five classes: Sadness (Class 0), Joy (Class 1), Love (Class 2), Anger (Class 4) and Fear (Class 5). Disproportionately high number of "Sadness" (94522), "Joy" (107767), and "Anger" (43678) instances. Disproportionately low number of "Love" (23528) instances.

Approach to Addressing Class Imbalance
* Undersampling (oversampling) of the most (least) numerous class to attain a count equivalent to the "Fear" class (as it has a count between both classes), where we'll:
    * undersample: the most numerous classes ("Sadness", "Joy", and "Anger") 
    * oversample: the least numerous class ("Love")

* To prevent "training" data from leaking into the "testing" data - the class rebalancing procedure will be applied:
    * after the "train-test-split" step, and
    * applied to the "training" sample only

Our first approach, Model 02A, was a Fine Tuning Distilbert model. While the DistilBERT model, a transformer-based model, was expected to perform well due to its ability to understand the semantic meaning of words based on their context, it did not meet expectations, with the model performing very poorly on the F1 and accuracy metrics. The Model was also the most computationally intensive of all the models ran. Potential reasons could be the model’s complexity and the possibility of insufficient training.

Our second approach was to try out simpler and more traditional models like models like Naive Bayes, Logistic Regression, XGBoost. The 3 Models performed well in general with a high overall accuracy and F1-score. Out of the 3 approaches, Logistic Regression model seems to performed the best. It had the highest accuracy and F1-scores, and it’s also relatively efficient.

Our final approach was a Multinomial LSTM Keras Sequential Model. To improve our model’s performance, we incorporated GloVe word embeddings. This semantic word embedding approach retained stop words and did not perform lemmatization, as the GloVe repository contained stop words and words with similar lemmas that had different word vectors. The use of GloVe word embeddings resulted in a small noticeable improvement in model performance over the non-neural network models, demonstrating the effectiveness of semantic word embeddings in enhancing the model’s ability to understand and classify text. 

Hence the Multinomial LSTM Keras Sequential Model using GloVe embeddings was our chosen model. It had the best performance. Moreover, this model is also preferred because it combines the strengths of RNNs and GloVe embeddings.
- RNNs (Recurrent Neural Networks) are capable of capturing the sequential information present in the text data, which is crucial for understanding the context of words in a sentence. On the other hand, GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words. These embeddings can capture semantic relationships between words, which is beneficial for our task to predict emotions from Youtube comments and transcripts.
- However, it’s important to note that while GloVe embeddings capture semantic relationships between words, they are static and do not take into account the context in which a word appears. This means that the same word will have the same representation regardless of its context. For example, the word “apple” would have the same representation in “apple computers” and “apple juice”. Therefore, while the combination of RNNs and GloVe embeddings can capture some level of context, it may not fully capture context-driven semantic word vectorisation.

### Developing a rule-based recommender system that matches the emotions of each user comment to a list of 5 music videos.

We first scraped the transcripts(lyrics) of 100 popular music videos from the 2020s and 1,000 comments each from a list of 100 music videos. Using the pre-trained emotion classifier model we predicted the emotions expressed in the 100 music video lyrics and the approx 50,000 youtube comments.

The final output are 2 DataFrames:
* Youtube comments dataframe that includes the original YouTube comments along with the predicted emotion and the probabilities for each emotion class.
* Dataframe that includes the transcripts of the 100 music videos along with the predicted emotion and the average probabilities for each emotion class.

We used a the following **Recommendation Rules** to set up different rules for recommending music videos based on the user’s emotion:

- **Positive Emotion**: If the user’s emotion is positive, look for music videos that evoke a similar emotion. For example, if a user’s comment is classified as ‘Joy’ with a certain probability, recommend 5 music videos whose ‘Joy’ probabilities are closest to that of the user’s comment. This is done using a method called KDTree, which is a way of finding the closest points (in this case, emotion probabilities) in a space.
- **Negative Emotion**: If the user’s emotion is negative, look for music videos that can help uplift their mood. Form a ‘vector’ or a set of values for the comment, consisting of the probabilities of the three negative emotions. Do the same for all 100 music videos. Then, calculate the cosine similarity between the comment vector and each music video vector. Cosine similarity is a measure of how similar two vectors are, and in this case, it’s used to find music videos whose emotion probabilities are most similar to the user’s comment. However, the system will only consider those videos whose sum of negative emotion probabilities is at least 0.1 less than that of the user’s comment. This ensures that the recommended videos are slightly more positive than the user’s current mood. The 5 music videos with the smallest cosine similarity scores are recommended.

The final output of this recommender system is a list of 5 music videos that resonates with the users’ current emotional state or aid them in navigating through negative emotions.


### Evaluation 
The Emotion-centric recommender system I have build is functional but has room for improvement. The system’s ability to classify emotions in music videos and user comments is not always accurate. For instance, some music videos and comments that seem negative are classified as positive and vice versa. Furthermore, the system sometimes recommends music videos that don’t seem to match the user’s emotional state. For example, it might recommend a sad song to a user expressing joy or a sadder song to a user who is already feeling sad.

### Potential Weaknesses
1. The classification model was trained on a labeled dataset of tweets from Kaggle. This might not be representative of the language used in YouTube comments or music video lyrics.
2. Emotions can overlap and it can be challenging to discern emotions from text. A comment or a song could contain elements of multiple emotions, making it difficult for the model to classify accurately.
3. The system only analyzes the transcripts of the music videos, which might not fully capture the emotional content of the songs. The actual audio signals and the visuals of the music videos are not considered.
4. The GloVe word embeddings used in the model capture semantic relationships between words but are static and do not consider the context in which a word appears. This means the same word will have the same representation regardless of its context, which could limit the model’s ability to fully capture context-driven semantic word vectorization.
5. The DistilBERT model, a transformer-based model, was expected to perform well due to its ability to understand the semantic meaning of words based on their context. However, it performed poorly, possibly due to its complexity and insufficient training.

### Conclusion

1. Music holds a profound capacity to elicit human emotions. The field of music recommendation is particularly influenced by this emotional aspect, as individuals often gravitate towards music that mirrors their current emotional state.
2. However, existing social media video recommendation methods, such as those on YouTube, primarily rely on genre features. This approach may not fully cater to the emotional needs of users, as it overlooks the emotional connection that users often seek in music.
3. This project offers a opportunity to enhance these systems by incorporating emotion-based recommendations, leading to a more personalized and emotionally resonant user experience.
3. Our classification model predicts the kaggle twitter text dataset with good accuracy and f1-score.
4. Our Recommender system will evaluate users’ emotional needs based on their comments and recommend the most “beneficial” music videos instead of the most “matched” ones.

### Future Work
For future work and to further improve on this project:
1. Consider increasing the data pool to include labeled comments from other social media platforms. This could help the model better understand the language used in YouTube comments and music video lyrics.
2. Explore using unsupervised learning to cluster the emotions of the text. This could potentially improve the accuracy of emotion classification.
3. Consider incorporating analysis of the audio signals and visuals of the music videos. This could provide a more comprehensive understanding of the emotional content of the songs.
4. Consider using context-aware word embeddings that can capture the semantic meaning of words based on their context.
5. Future work could explore tuning the DistilBERT model or providing it with more training data to improve its performance.

