# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Singapore Housing Data and Kaggle Challenge

> SG-DSI-41 Group 01: Daryl Chia, Germaine Choo, Lionel Foo

---

**Problem Statement**
As young couples are eager to move into new HDB Build-to-Order (BTO) homes while often having limited buying power, we aim to offer a second opinion/ advise on looking at HDB resale flats instead. In particular, we are focusing on what a suitable market price is and what key flat features to look out for; to ensure that newly-weds are not overspending, but snatching a great deal with more confidence.
Specifically, we will design and evaluate different linear regression models to best predict house price and also provide information on the main house features that are driving flat price across the board. This will position young couples with the know-how and benchmark price to negotiate successfully.
Models to explore:
1. Model 1: Linear regression: LASSO (l1) and Ridge (l2) regularization
2. Model 2: Model 1 with polynomial features: LASSO (l1) and Ridge (l2) regularization
3. Model 3: Model 2 (drop multicollinear features again): LASSO (l1) and Ridge (l2) regularization
4. Model 4: Model 3 (drop features with low correlation to the target variable): LASSO (l1) and Ridge (l2) regularization

In selecting an ideal model, the success will be generally evaluated based on what the Root Mean Squared Error (RMSE) metric is.

**Steps**
1. Load train.csv in notebook 01_EDA_&_Data_Cleaning.ipynb
2. Load hdb_train_new.csv into notebook 02_Feature_Eng_&_Preprocessing.ipynb
3. Load resultant output files into notebook 03_Modelling.ipynb
4. Load train.csv into notebook 04_Data_Visualisation.ipynb to generate slide visuals
5. Load test.csv into notebook 05_Kaggle.ipynb
6. Load model_3.csv, hdb_test_new.csv, test.csv into notebook 06_Kaggle_Predictions.ipynb to run model and generate Kaggle test results

**Data Cleaning and Exploratory Data Analysis (EDA)**

The data cleaning and exploratory data analysis (EDA) involved an intial EDA to deal with wrong data types, null values, and variables that need to be redefined. This was followed by the necessary data cleaning, such as dropping rows with null values.

In the next round of EDA, the features were grouped together according to related features, such as those for public transport. This was so the analysis could be done on a more manageable scale.
For each group, numerical features were analysed for their correlation (to each other and with the target variable) and distribution, while categorical features were analysed for their association (to each other using ratios), different averages for the target variable (catplots), disitribution, and count of unique values. In addition, there were other analyses, such as whether certain variables were redundant.

Dealing with Multicollinearity:
Where numerical features were collinear to each other, one was kept and the rest were dropped. This was also the case with categorical variables that are highly or perfectly associated with one another.

Dealing with Potential Overfitting:
Where categorical features had around more than 100 unique values, they would be dropped as the sheer amount of dummies generated in a linear regression model would not only be minimally interpretive but likely cause significant overfitting.

Once multicollinearity and overfitting was thoroughly dealt with, we can be confident that the data can be fitted in the linear regression models to generate accurate and informative predictions.

**Preprocessing and Modeling**

We conducted comprehensive feature engineering, resulting in the creation of four distinct datasets tailored for regression modeling. These datasets, namely:
Model 1: Exclusively utilises non-polynomial features as predictor variables, serving as a baseline for performance comparison. This allows us to gauge the impact of introducing polynomial features on model performance.

Model 2: Incorporates polynomial features as predictor variables to capture non-linear relationships in real-world data. The addition of higher-degree and interaction terms aims to improve model performance, reflected in metrics such as R-squared and RMSE. Interaction terms enhance our understanding of feature interactions, providing valuable insights into underlying processes.
Model 3: Further refines Model 2 by addressing multicollinearity. Correlation analysis is employed to identify and eliminate highly correlated polynomial features (correlation of > 0.8), enhancing the model’s robustness.

Model 4: Following the identification of multicollinearity in Model 3, we conducted an additional evaluation on the remaining polynomial features. Specifically, we analysed the correlation between each feature and the target variable. Features with a correlation of less than 0.1 with the target variable were dropped in this model.
In the preprocessing phase, categorical data across all four datasets were converted into a numerical binary format suitable for regression analysis using one-hot encoding. Subsequently, standard scaling was applied to normalize numerical features in all datasets.

After preprocessing, a train-test split was performed on each of the four processed datasets. Linear regression, Lasso, and Ridge models were then applied to evaluate the effectiveness of each model in predicting HDB resale prices.

**Evaluation and Conceptual Understanding**

Models to explore:
1. Model 1: Linear regression: LASSO (l1) and Ridge (l2) regularization
2. Model 2: Model 1 with polynomial features: LASSO (l1) and Ridge (l2) regularization
3. Model 3: Model 2 (drop multicollinear features again): LASSO (l1) and Ridge (l2) regularization
4. Model 4: Model 3 (drop features with low correlation to the target variable): LASSO (l1) and Ridge (l2) 

| Model | Regularization Method | Training R2 Score | Testing R2 Score | Training RMSE | Testing RMSE |
|-------|-------------|-------------------|------------------|---------------|--------------|
| 1     | Lasso  | 0.8963            | 0.8932           | 46142.30      | 46728.75     |
| 1     | Ridge  | 0.9026            | 0.9004           | 44721.66      | 45140.20     |
| 2     | Lasso  | 0.9222            | 0.9203           | 39956.41      | 40381.66     |
| 2     | Ridge  | 0.9330            | 0.9320           | 37076.91      | 37297.88     |
| 3     | Lasso  | 0.9134            | 0.9117           | 42148.57      | 42492.25     |
| 3     | Ridge  | 0.9218            | 0.9208           | 40060.81      | 40255.52     |
| 4     | Lasso  | 0.9093            | 0.9071           | 43153.32      | 43588.56     |
| 4     | Ridge  | 0.9161            | 0.9145           | 41493.88      | 41806.84     |

R2 score calculates percentage of the error in the data can be explained by a model, while Root Mean Squared Error is the average error per data point. Both metrics are inversely related, with a high R2 score implying a lower RMSE value. Based on the project objectives, RMSE score was chosen as the metric to evaluate out models on.
**Model 1** was the baseline model where polynomial features were not yet incorporated.

Based on the R2 scores and RMSE values, **Model 2 (Ridge)** appears to be the best model. It has the highest testing R2 score (0.9320), indicating that it explains the most variance in the target variable. Additionally, it has the lowest testing RMSE (37297.88), suggesting that its predictions are, on average, closest to the actual values. However, it’s important to consider the complexity of the model and ensure it’s not overfitting the training data.

However, in **Model 2 (Ridge)**, multicollinearity as a result of exhaustively generating polynomial features could lead to unstable coefficients, difficulty in interpretation, and overfitting. Therefore, to addess this, **Model 3 (Ridge)** was chosen because of its stability, interpretability, and next lowest RMSE score.
**Model 4** was developed to manually drop features with low correlation to the target variable. However, the RMSE score turned out to be worse. Furthermore, regularization methods like Ridge and Lasso are capable of feature selection by shrinking the coefficients of less important features towards zero. This automated form of feature selection can often be more effective and less biased than manual feature removal based on correlation with the target variable. Therefore, **Model 3 (Ridge)** was a better choice.

To conclude, we have determined **Model 3 (Ridge)** is the best model overall, given its balance of performance, interpretability, feature selection, and low RMSE score.

**Conclusion and Recommendations**

Choosing model 3 as our most optimal model, where young couples have potential set of flats considered for purchase, we are able to input the key features identified and generate a range of resale prices with the RMSE.

In addition, apart from having this value gauge, newly weds are also better-informed of the key features which would significantly shift the needle of resale prices more than others. For instance, apart from focusing on the typical features such as age and floor area of flats, its location and flat model are highly important. In particular, areas located in central region such as Bukit Timah, or Marine Parade could expect much higher resale prices; being one of the top features with the highest absolute coefficient values. Also, flats of Terrace models should see much higher resale prices, being another top feature. It is also critical to note that couples should not be fooled by the argument that close proximity to a bustop or mrt station would have significantly raise resale flats over other more impactful features.

While it is intuitive that flats would cost more if located close to amenities, it is crucial to note that having a higher number of hawker food stalls within 5km of the flat would also significantly inflate its price. This particular feature has observed to create a larger impact in shifting flat costs other amentities' characteristics such as proximity of malls.
Given couples' limited spending power and understanding the weightages of features in shifing the price, they will be able to better weigh their priorities with respect to the extent of price difference from these specific features. They would also be better guarded against arugments that they should pay a higher price given certain 'logical' flat features, which are actually not shifting much of the needle given our analysis.

Also, sspecially now with volatile flat prices, the predicted prices that the model provides (with RMSE) would be place newly weds in a better position with stronger bargaining power and confidence in ensuring that they are not overspending on their desired flat, while also identifying a great deal.
Moving forward, understanding the impact of cooling measures and future developments on resale prices, we aim to include these factors into the regression model for a more holistic comprehensive analysis. Accounting for supply and prices of BTO flats would also be done as BTO and resale flats are often intertwined closely.