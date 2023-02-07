# Project Outline

## Data Source

Amazon Customer Reviews Dataset: https://s3.amazonaws.com/amazon-reviews-pds/readme.html



## Part 1 Basic Data Analysis

Continue to think about the content

## Part 2 Recommender System with PyTorch

### Issues to recognize

**The core in ML world**: frame and change the abstract problem into practical machine learning task

1. Cold-start problem

   Our dataset is sparse as most of practical problem is, therefore, what is a good way to predict labels for them?

2. Test Set problem

   There are actually time differences between values. It might be <u>data leakage with a look-ahead bias</u> when we put future data in train set and past data in test set

3. Explicit vs. Implicit

   We are still trying to find explicit feedbacks. However, explicit feedback is rare while implicit feedback is abundant. The implicit method might be more practical:

   - But it also make us think about what are recommender system for: is that ultimately the goal to recommend the content that the user will interact with?
   - Similarly, we might  

## Part 3 NLP Classification with Transformers



## Reference

https://arxiv.org/abs/1708.05031

https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e