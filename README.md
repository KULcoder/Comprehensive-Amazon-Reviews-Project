# Comprehensive-Amazon-Reviews-Project

*Some idea of this project is inspired from a deep learning course and a recommender system course in University of California, San Diego. All rights reserved.*



This is a multi-task projects (analysis, recommender system, NLP) on the Amazon Reviews Dataset (Electronics Part)

Data Source: https://s3.amazonaws.com/amazon-reviews-pds/readme.html

Dataset size: 3091024



**This project is designed for me to better understand data science**: What is the purpose of data science?

- *The core is to extract and utilize useful information from data*



Different notebook is in different folders. There might be additional information provided in md file in each folder.



## Part 1 Exploratory Data Analysis

This part we performed some data preprocess, and perform some univariate and multivariate data analysis (majorly visualized in notebook).

**Dataset Columns**:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.

### Data Preprocess

Following preprocess is performed:

- missing values: since only very few missing values is in the data, we directly drop the rows of missing values
- duplicate data: drop any duplicate data
- irrelevant data: drop the following irrelevant columns: 'marketplace', 'product_parent', 'product_category'
- correct data type: change review_date column from string data type into datetime data type
- transform data: we create some new features for better understand the data

### Univariate Analysis

Investigate the distribution and values in each features.

Some interesting observations: 

- Review data in more on weekdays than weekends

- data is majorly clustered after 2013

### Multivariate Analysis

Investigate the relationship between two variables.

Some interesting observations:

- Star Ratings do significantly influence the distribution of other features (review week days, verified purchase, review length...)
- different category of product display different average ratings


---


## Part 2 Recommender System

This is an application of explicit feedbacks recommender system with multiple models. Supported by `surprise` and `pytorch`, `pytorch lightning` libraries.

### Task

If we only have the data of userID, productID, and ratings. How is it possible for us to **make predictions with only that two ids**? This is related to an concept of latent factor: *know the user-product pairs might be enough to obtain information for recommender system*

### Models and Performance

We use Mean Squared Error & rounded accuracy to evaluate the result. (Notice that we are doing a regression task and the rounded accuracy: round prediction to nearest integer is just for reference)

#### Baseline Model: Similarity-Based Rating Estimation

We try to predict rating of a user to a product by the weighted average of its ratings on other products (with similarity between products as weights)

Performance: 

- MSE: **1.9474959814844819**
- Accuracy: **0.21011289462650565**

#### Latent Factor Model

Approximate the user and the product by a length k vector with unknown values, and then train to learn those values as the model. 

Performance:

- MSE: **1.6888076658234337**
- Accuracy: **0.2831695904011098**

#### Neural Collaborative Filtering

This is an idea built on Latent Factor Model, if we are guessing those vectors, why not directly let a neural net to learn the similarities? We express user by products, products by users, then encoded them and concatenate them into two fully connected linear layers and an output layer.

Performance:

- MSE: **0.7011991391584224**
- Accuracy: **0.5145615363213437**



### Issues Discovered

#### Test Set Construction

When we are constructing the test set, it is not practical to predict past values by future data. A <u>more practical way</u> will be to use a  <u>leave-one-out method</u> (only choose the newest interaction of a user) to construct the test set.

#### Cold Start & Sparsity Problem

One of the major issue of naïve recommender system is its training data is sparse: a lots of user only leave very few interactions in the dataset, therefore, it might be hard to obtain information for those users. One solution can be the <u>embedded layer</u> can help reduce this issue, or another way can be <u>increasing trainable dataset</u>.

#### Implicit vs. Explicit Feedback

Closely related to the previous problem, one idea will be not to predict the ratings but whether to predict if one user will interact with a product or not. This greatly increase the amount of data we have since <u>implicit feedback</u> can be gathered more than reviews dataset. This can includes the data of *purchase, view, search....*

---

## Part 3 NLP 

This part of the project targeting in NLP classification task: can we predict the ratings based on raw review text data? This is a meaningful task since language is possibly one of the most generated data type by humans, if we can extract ratings, we can extract sentiments and more information! Support by `sklearn`, `PyTorch`, `PyTorch Lightning`, and `Transformers` in this part.

### Task

Predicted ratings by review texts on a sampled balanced dataset (equal amount of 1, 2, 3, 4, 5 star ratings).



### Models and Performance

We use accuracy and confusion matrix to evaluate the result. Also I transformed the result into positive (4, 5) vs. negative (1, 2) that can be viewed as a simulation of sentiment analysis, we will also evaluate accuracy and confusion matrix on that.

#### Baseline: Tf-idf with Logistic Regression

The Tf-idf vectorizer can transform a sentence into useful vectors based on importance of words. We use a logistic regression on those vectors to train our model.

Performance:

- Basic Accuracy: **0.5057333333333334**
- ![baseline model confusion matrix](https://github.com/KULcoder/Comprehensive-Amazon-Reviews-Project/blob/main/Part3_NLP/images/baseline_cm.png)
- Negative vs Positive Accuracy: **0.9195728349741364**
- ![baseline model positive vs negative confusion matrix](https://github.com/KULcoder/Comprehensive-Amazon-Reviews-Project/blob/main/Part3_NLP/images/baseline_np_cm.png)

#### Fine Tuned BERT Model

BERT model, the encoder part of transformer (decoder part is the famous GPT model), can be used to transform a sentence into useful vector with **attention**. I placed one fully connected layer and an output layer on the top of the BERT model to modify the pre-trained model for our task (transfer learning).

Performance:

- Basic Accuracy: **0.6261333333333333**
- ![BERT model confusion matrix](https://github.com/KULcoder/Comprehensive-Amazon-Reviews-Project/blob/main/Part3_NLP/images/bert_cm.png)
- Negative vs Positive Accuracy: **0.9834095141357712**
- ![BERT model positive vs negative confusion matrix](https://github.com/KULcoder/Comprehensive-Amazon-Reviews-Project/blob/main/Part3_NLP/images/bert_np_cm.png)



### Result

The improvement of BERT model might not be as high as expected. However, I do find that the BERT model is not only making more correct prediction, but also making less severe false predictions. Which brings me to think:

- Is it possible to obtain the exact ratings by only the review text? Can a human do this? Humans might be like the model that can properly predict is a review text positive or negative, too.
- There might be other strong pre-trained model (like lots of them on huggingface.co) can improve the performance of the model.
- There might be better way of preprocessing the data that filter / transform the data to improve the performacne of the model.

(the training process graph is provided in md under 'Part3_NLP' folder)

---

## Reference

https://s3.amazonaws.com/amazon-reviews-pds/readme.html

https://arxiv.org/abs/1708.05031

https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e





Waited to be present & organized

## To Do

1. Add some graphs of training / validation
2. Add some analysis on review body
3. organize into different folders
