# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

**DSI-41 Group 2**: Muhammad Faaiz Khan, Lionel Foo, Gabriel Tan

## **Project title**: Generative AI and Art - understanding and predicting chatter from online communities

## 1. Introduction and Background
The rapid development of generative Artificial Intelligence (AI) in the art space has created significant buzz. While [there is much opportunity for the growth of this market](https://www.coherentmarketinsights.com/industry-reports/digital-artwork-market), the potential for polarising opinions over certain issues have emerged. Top issues are AI's impact on [work for existing artists](https://www.businessinsider.com/ai-taking-jobs-fears-artists-say-already-happening-2023-10), independent human creativity and thought, and [broader questions about ownership/ copyright](https://lawgazette.com.sg/feature/generative-ai/). 

In today's digital age where (online) culture wars can be easily imported, it would be important for Singapore's policymakers to to get a quick sensing, understand and quantify which way the discourse may be heading. This would better help authorities frame engagement efforts with stakeholders in the arts community and refine regulatory frameworks in line with any prevailing concerns.

With the above in mind, this project selected two online communities (subreddits) for analysis of their social medias posts:
- [**r\DefendingAIArt**](https://www.reddit.com/r/DefendingAIArt). i.e. the **'pro'** camp. Describes itself as a _"space for pro-AI activism"_ and is aimed at _"fighting misinformation and attempts at legislation against AI generated artwork"_
- [**r\ArtistHate**](https://www.reddit.com/r/ArtistHate) i.e. the **'anti'** camp. Named as such for Redditors to discuss what is being observed as _"increasing hate against artist and art hobbyists"_ when their interests should be better protected amidst AI

## 2. Problem Statement

**Problem Statement and Objective**: <br>
This is a binary classication problem that aims to use Natural Language Processing (NLP) to accurately predict which of the two subreddits a given post comes from, and by extension, to help policy-makers understand key differences in 'pro-AI' and 'anti-AI' discourse and sentiments.

**Models tested**: 
- _Multinomial Naive Bayes_ (commonly used in NLP), as well as other classification algorithms:
- _Logistic Regression_
- _k-Nearest Neighbours_
- _Random Forest_
- _Adaboost Classifier_

_Multinomial Naive Bayes_ was chosen to be our final model of choice (see ['Findings and Evaluations'](#4-findings-and-evaluation)).

**Evaluation**: <br>
Key metrics used for evaluating success and for selecting the final model:

- _Accuracy score_ - _overall_ percentage of predictions ('pro-AI' and 'anti-AI') that were correct
- _Recall score_ - percentage of instances _out of all actual cases for one class (e.g. 'pro-AI')_ that the model correctly predicted
- _Precision score_ - percentage of predictions the model could get correct _whenever it is predicting for one class (e.g. 'pro-AI')_
- _F1 score_ - combined average (harmonic) of recall and precision scores
- _Computational cost_ - relative amount of time taken to run a model versus others

Reason for appraising these metrics collectively: 
- It is deemed equally important that stakeholders be able to distinguish social media chatter from either camp to get a quick ground sensing and therefore better frame engagement efforts (i.e. 'accuracy')
- Yet, concerns from the 'anti' camp should not be misclassified as being pro-AI (e.g. 'precision'), so that legitimate issues can be addressed by the authorities instead of falling through the gaps

**Scope**: <br>All English subreddit posts (inclusive of original posts and replies/comments) retrieved from [**r\DefendingAIArt**](https://www.reddit.com/r/DefendingAIArt) and [**r\ArtistHate**](https://www.reddit.com/r/ArtistHate), analysed using NLP and other supervised learning modelling methods. Both subreddits were created in 2023 and therefore the content of posts reflect roughly the same recency and time period of discourse.

**Stakeholders**:
- _Primary stakeholders_: Government and public sector policymakers, such as those involved in: 
    - furthering of related industry and AI tech initiatives (e.g. _InfoComms Media Authority of Singapore_),
    - developing the local arts scene and engaging with artists/art institutions/the general public (e.g. _National Arts Council_)
    - establishing relevant guidelines and regulatory frameworks to keep up with the pace of AI development (e.g. _Personal Data Protection Commision; Intellectual Property Office of Singapore_) 
- _Secondary stakeholders_: Industry stakeholders such as artists, or local arts institutions and [interest groups](https://artshouselimited.sg/artandai)


## 3. Data Collection
Data was scraped from both subreddits on 18 January 2024 via the Python package _'Python Reddit API Wrapper'_ (PRAW), which allows for simple access to Reddit's API. Each row of data was tagged as being from either subreddit to train and test the potential models. After text cleaning, pre-processing and balancing of classes in the data by subreddit source, this made for a final 6,000 posts approximately. 

The final data dictionary is as below:

**Data Dictionary**:

|Feature|Type|Description|
|---|---|---|
|**From subreddit source:**|||
|`subr-def_ai`|int|Boolean, whether the post is from _**r/DefendingAIArt**_ ('1') or _**r/ArtistHate**_ ('0')|
|`is_op`|int|Boolean, whether the post is the original post/OP ('1') or a comment ('0') below the original post |
|`author`|str|Reddit user name|
|`post_id`|str|Unique identifier for each post|
|`body`|str|Text content or body of the post|
|`upvotes`|int|Number of upvotes for the post|
|`num_comments`|int|Number of comments/responses to the post|
|**Engineered features:**|||
|`body_length`|int|Number of characters in the post|
|`body_word_count`|int|Number of words in the post|
|`sentiment_scores`*|object| Overall sentiment dictionary with `neg`, `neu`, `pos` and `compound` scores|
|`neg`*|float|Proportion of text in the post rated as 'negative' in sentiment|
|`neu`*|float|Proportion of text in the post rated as 'neutral' in sentiment|
|`pos`*|float|Proportion of text in the post rated as 'positive' in sentiment|
|`compound`*|float|'Overall' sentiment score between -1 and 1 where: <br> score >= 0.05 if positive,<br> score between -0.05 and 0.05 if neutral,<br> score <= -0.05 if negative|
|`subjectivity_score`^|float|Subjectivity score between 0 and 1, where 0 indicates a very objective text and 1 represents a very subjective text|

\* Based on VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool
<br>^ Based on TextBlob (Python NLP library)

## 4. Findings and Evaluation

To answer our problem statement, our modelling was run using NLP-processed text from `'body'` as our predictor variables and `'subr-def_ai'` as our target variable.

**_4.1 Model scores (with tuning)_**

| Model                   | Training accuracy | Testing accuracy | Precision | Recall | F1-score | Computational time<br>(relative to Naive Bayes)|
|-------------------------|-------------------|------------------|-----------|--------|----------|----------|
| Multinomial Naive Bayes | 0.83              | 0.68             | 0.67      | 0.73   | 0.70     |1 x|
| Logistic Regression     | 0.96              | 0.71             | 0.73      | 0.67   | 0.70     |2 x|
| K-Nearest Neighbors     | 1.0               | 0.55             | 0.70      | 0.17   | 0.28     |2 x|
| Random Forest           | 0.93              | 0.65             | 0.63      | 0.71   | 0.67     |up to 12 x|
| Adaboost Classifier     | 0.79              | 0.67             | 0.70      | 0.61   | 0.65     |5 x|

Based on the above, we selected the **Multinomial Naive Bayes** as our final model algorithm, given its predictive accuracy of close to 70% on unseen data and the smallest difference from our training accuracy, suggesting reasonable generalizability of this model. It is also computationally the cheapest option, offering our stakeholders for a quick ground sensing of online discourse.
<br><br>
**_4.2 Sentiment analysis_**

| subreddit      | Negative  | Neutral | Positive | Compound score | Subjectivity score |
|----------------|----------|---------|----------|----------------|--------------------|
| **r\DefendingAIArt** | 0.096 | 0.793 | 0.111 | 0.073 | 0.492 |
| **r\ArtistHate**     | 0.092 | 0.778 | 0.129 | 0.113 | 0.463 |

For both subreddits, sentiments and subjectivity scores reflect slightly positive sentiments _(compound scores over 0.05)_, and on average a balance between subjective opinions and objective discussions _(subjectivity scores around 0.5)_. This is encouraging as it suggests the current discourse is not strongly polarising or emotional; however the overall positive sentiment inferred has its shortcomings as humour or sarcasm may not be picked up by the current algorithm.

<br><br>
**_4.3 Keywords/topics discussed by Redditors and other observations_**
![Top bigram frequencies by subreddit](image.png)
Top words by frequency counts on both camps did not differ very much, suggesting that either camp may simply be discussing different perspectives on the same AI-specific issues, e.g. **AI's ability to generate what is 'real' art or not** (with terms such as _'look like'/'feel like'/'just like'_) and **text-to-image capabilities** (e.g. _'Stable Diffusion'_). There is also some indication of concern about the use of **_'training data'_ for AI algorithms** in **r\ArtistHate**

There are also interesting observations that suggest the nature of debate and conversations may be dominated by a select few in one subreddit over the other:
- **r\DefendingAIArt**: almost twice the number of unique users (1096) compared to ArtistHate (603)
- **r\ArtistHate**: similarly much higher proportion of posts (31.95%) contributed by the top 20 users compared to DefendingAIArt (15.42%)



## 6. Conclusion and recommendations

The topic of generative AI art is a relatively new topic and our data likely captures only a fraction of the online chatter that is likely to build up over time. Nonetheless the predictive accuracy of 70% by our model offers our stakeholders the opportunity for a quick categorical sensing over the baseline chance of 50%.

As next steps, we recommend the following:

**1. Devote greater funding in these areas**:
- Extension of our project into further research by collecting/scraping more data across other social media platforms, or other methods such as in-depth interview/ focus groups/ surveys with artists and the public to get more nuanced sentiments etc.
- Programmes to promote human artist-AI system collaboration, not replacement, given emerging tools such as Stable Diffusion
- Educational efforts to raise awareness about AI’s impact among artists and other professionals such as curators, art-related tech developers etc.

**2. Review potential concerns raised, i.e. review regulatory frameworks and establish ethical guidelines (Code of Practice):**
- Better clarity around intellectual property ownership of AI-generated artworks 
- Safeguard individuals’ privacy and  training data being fed into AI models
- Encourage ethical data sourcing practices by artists and developers
- Requiring disclosure in the use of AI for artworks

**3. More public engagement to understand ground perspectives and offer reassurance over concerns:**
- Industry stakeholders: artists, technology developers, art institutions
- Involve the public in decision-making processes




## 7. References and external research
1. [Digital Artwork Market Size and Share Analysis - Growth Trends and Forecasts (2023 - 2030)](https://www.coherentmarketinsights.com/industry-reports/digital-artwork-market)
2. [Workers are worried about AI taking their jobs. Artists say it's already happening.](https://www.businessinsider.com/ai-taking-jobs-fears-artists-say-already-happening-2023-10)
3. [Generative AI - An evaluation of the current solutions to address the intellectual property challenges it generates](https://lawgazette.com.sg/feature/generative-ai/)
4. [Singapore Digital Art Market: Complete Guide](https://dxagroup.io/singapore-digital-art-market-complete-guide/)
