# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4: Detecting Hate Speech

> SG-DSI-41 Group 01: Lionel Foo, Joel Lim, Poon Wenzhe, Daryl Chia

---

### Problem Statement

We are a Cyber Wellness Non-Profit in Singapore and the goal of our project is to empower internet users and community leaders (moderators) with the means to block hateful/offensive comments and users. This is so that our stakeholders can have a healthy browsing experience.

Specifically, this project will explore the use of Recurrent Neural Networks (RNN) to distinguish between these types of speech:

1. Hate
2. Offensive
3. Neither (Neither Hate nor Offensive Speech)

Since we are placing equal importance in wrongly detecting hate/offensive speech and missing out actual hate/offensive speech, both the recall and precision scores are important to us. In addition, the overall accuracy of our detection model is of interest. As a result, our models will be evaluated on their (a) f1-score and (b) accuracy.

In terms of deliverables, we intend to both detect hate/offensive comments and flag users with a history of making such. This will allow us to empower our users and community leaders with the means to exercise discretion in censoring out hate/offensive comments.

### Organization

The code notebooks will be grouped into 3 main types based on their sequential order, as follows:

1. 01_EDA_&_Data_Cleaning
2. 02A-M1_Classification Model (Multiclass Classification - RNN w.out GloVe Word Embeddings)
3. 02A-M2_Classification Model (Multiclass Classification - RNN with GloVe Word Embeddings)
4. 02B_Classification Model (Binary Classification - RNN with GloVe Word Embeddings)
5. 03_Reddit_Scraping_Demo

### Data

For the data, we will be using a dynamically generated hate speech dataset from Kaggle, which had each tweet classified by a panel of people using a standardized definition. The dataset will consist of speech classifications for:
1. Hate Speech
2. Offensive Speech
3. Neither

### Exploratory Data Analysis

The EDA process mainly focused on looking at substrings to remove through regex cleaning and also looked at the top 1/2-gram words for each target class.

For data cleaning process, we removed:
1. Line and tab characters
2. Website urls
3. Hashtags and emojis
4. Tweet mentions
5. Reteweet strings
6. Numbers
7. Punctuations
8. Unecessary white spaces

In terms of the top 1/2-gram words, hate speech words unique had discriminatory words compared to the other classes, while both hate and offensive words both had profanities. As for the class "neither", it neither had profanities nor discrimnatory words.

### Modeling

For our study, we began with a dataset that had three classes: Hate Speech (Class 0), Offensive Language (Class 1), and Neither (Class 2). We observed a significant class imbalance, with fewer tweets categorized as “hate speech” (5.8%) and “neither” (16.8%).

Our first approach, Model 02A-M1, was a Multinomial LSTM Keras Sequential Model. To address class imbalance, we undersampled the most numerous class (“offensive language”) and oversampled the least numerous class (“hate speech”). However, this model struggled to accurately detect hate speech (Class 0), yielding low f1-scores.

To improve our model’s performance, we incorporated GloVe word embeddings in Model 02A-M2, a Multinomial LSTM Keras Sequential Model. This semantic word embedding approach retained stop words and did not perform lemmatization, as the GloVe repository contained stop words and words with similar lemmas that had different word vectors. The use of GloVe word embeddings in Model 02A-M2 resulted in a noticeable improvement in model performance compared to Model 02A-M1, demonstrating the effectiveness of semantic word embeddings in enhancing the model’s ability to understand and classify text. However, despite these improvements, the f-1 scores were still not ideal in relation to predicting hate speech (Class 0). This indicates that while GloVe word embeddings can enhance model performance, additional strategies may be needed to effectively classify hate speech.

Finally, in Model 02B, we performed a Binomial LSTM Keras Sequential Model with GloVe word embeddings. In this model, we combined hate speech (Class 0) and offensive language (Class 1) into a new class, Class 1, representing hurtful text. The original Class 2 (neither) was reclassified as Class 0, representing non-hurtful text. This reframing transformed our multi-class classification problem into a binary one, which simplified the task and improved the model’s performance.

Throughout this process, we made several adjustments to handle class imbalance and improve model performance, including undersampling, oversampling, class merging, and the incorporation of GloVe word embeddings.

All models followed a similar process. We implemented a Sequential model containing three Embedding Layers to learn featured vector representations of the input vectors, a Bidirectional LSTM layer to identify useful patterns in the sequence, and a fully connected layer. We included BatchNormalization layers to enable stable and fast training and a Dropout layer before the final layer to avoid overfitting. The final layer is the output layer which outputs soft probabilities for the classes. We used ‘softmax’ activation for the multinomial models and ‘sigmoid’ activation for the binomial models.

While compiling a model, we provided three essential parameters: optimizer, loss, and metrics. The optimizer helps to optimize the cost function by using gradient descent. We used binary_crossentropy for the binomial models and categorical_crossentropy for the multinomial models. The loss function monitors whether the model is improving with training or not. Metrics help to evaluate the model by predicting the training and the validation data.
 
To prevent model overfitting, we used L2 regularization, dropout, and callback. Callbacks are used to check whether the model is improving with each epoch or not. If not, then necessary steps are taken like ReduceLROnPlateau decreases learning rate further. Even then if model performance is not improving then training is stopped by EarlyStopping. For callback, we used monitor = ‘val_accuracy’, restore_best_weights = True, to ensure that the model would revert back to the optimal epoch with the best validation accuracy score.
We tweaked the hyperparameters to ensure we obtained the best performing model. This summarizes our process and the steps we took in our study.

### Demo (Streamlit)
A demo using the chosen model is available for this project. Additional libraries are required:

- `joblib`
- `streamlit`
- `cv2`
- `pytesseract`

#### Computer Vision - Image to Text
This demo contains an element of computer vision by extracting the text from a given image. This allows our model to be applied on images that potentially contains hate/offensive text.

`Tesseract` should also installed on your device for this demo. 

For Mac, run the command `brew install tesseract`.

For Windows, download the binary file from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki).

Update the variable `pytesseract.pytesseract.tesseract_cmd` in `demo.py` to reflect the file path where necessary.

#### Required Files
Data is scraped from Reddit for this demo. The following files are in the `demo` folder:

- `threads.csv`: data on scraped threads
- `comments.csv`: data on scraped comments
- `clean_merged.csv`: the final dataset that is cleaned and combined for use in the demo
- `n-oreo-offensive.png`: an image containing hateful/offensive text for the demo
- `Model_02B_model.pkl`: the file containing the fitted chosen model
- `Model_02B_tokenizer.pkl`: the file containing the tokenizer fitted to use with GloVe embedding

#### Running the Demo
1) Use a terminal such as command prompt or Anaconda prompt depending on the installation path of Python, change the directory to point the the `demo` folder
2) Run the line on the terminal `streamlit run demo.py`
3) When the streamlit demo is successfully running, the terminal will display the Local URL, which shows the localhost port used on the machine, and the Network URL

To allow other machines to connect to the Network URL, go to your machine's Firewall settings and allow `python.exe` to access networks. The other machines should be connected to the same network as the machine you are using to host the streamlit demo.

### Conclusion & Recommendations

1. Hate speech results in many negative impacts to individual mental health, divides communities, and has wider economic implications.
2. This project offers a viable solution by using a LSTM model with word embeddings to classify if a given text is hateful/offensive.
3. Our model predicts the combined class of hate/offensive texts with good accuracy and f1-score.
4. Our solution empowers users to mitigate the impact of online hate speech by detecting hate/offensive comments which can then be incorporated into a browser extension to censor such comments on social media platforms.

### Future Work
For future work and to further improve on this project:
1. Obtain a localised dataset containing local vocabulary and text patterns.
2. The localised sample may contain words that are foreign to the word embedding used in the current model; we should explore using localised word embeddings if available.
3. Obtain data from various social media platforms.
4. The testing of the model should be done with a more representative sample of social media comments as the current sampled dataset is already biased to more offensive posts whereas the ground truth may have a much different ratio of classified posts. 
5. For recognising hate/offensive images, more comprehensive computer vision models could be used, and these might involve incorporation of other means to contextualise and classify images (instead of using the same text model).