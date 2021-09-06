The Quora Question Pair Similarity Problem
==========================================

A beginner’s journey through the various life cycles of a problem on Kaggle.
----------------------------------------------------------------------------

[![Sanjay Chouhan](https://miro.medium.com/fit/c/56/56/1*tLOOCkAOaetwQ_E4zwz0Ww.jpeg)](https://sanjayc.medium.com/?source=post_page-----3598477af172--------------------------------)[

Sanjay Chouhan

](https://sanjayc.medium.com/?source=post_page-----3598477af172--------------------------------)[

5 days ago·11 min read

](https://medium.com/the-quora-question-pair-similarity-problem-3598477af172?source=post_page-----3598477af172--------------------------------)

![](https://miro.medium.com/max/8000/1*ZOjoppdWw-zPqO1gJ5Aaeg.png)Photo by [Me](https://www.instagram.com/sanjaychouhansc/).

This is my first case study, so you can expect a beginner-friendly data analysis and model building. I have used only the classical machine learning models for this problem. However, working on this case study was a great learning experience for me. And in this blog, I will try to share with you as much as possible.

In the blog, I will write only the summary. You can view the full notebook [here](https://nbviewer.jupyter.org/github/scsanjay/case-studies/blob/main/01.%20Quora%20Quesion%20Pair%20Similarity/quora-duplicate-questions.ipynb) and you can view the code on [github](https://github.com/scsanjay/case-studies/tree/main/01.%20Quora%20Quesion%20Pair%20Similarity).

And to all the experienced folks out there, I would love your feedback for future case studies. 🤝🤓

Table of contents
=================

1.  Introduction
2.  Business Objectives and Constraints
3.  Data Overview
4.  Business Metrics
5.  Basic EDA
6.  Data Cleaning
7.  Feature Extraction
8.  EDA with Features
9.  Featurization with SentenceBERT  
    i. EDA on new features related to SentenceBERT
10.  Data Pre-processing
11.  Training Models  
    i. Support Vector Classifier  
    ii. Random Forest  
    iii. XGBoost  
    iv. Another XGBoost 🏆
12.  Final Thoughts
13.  References

Introduction
============

Quora is a platform for Q&A, just like StackOverflow. But quora is more of a general-purpose Q&A platform that means there is not much code like in StackOverflow.

One of the many problems that quora face is the duplication of questions. Duplication of question ruins the experience for both the questioner and the answerer. Since the questioner is asking a duplicate question, we can just show him/her the answers to the previous question. And the answerer doesn’t have to repeat his/her answer for essentially the same questions.

For example, we have a question like “How can I be a good geologist?” and there are some answers to that question. Later someone else asks another question like “What should I do to be a great geologist?”.  
We can see that both the questions are asking the same thing. Even though the wordings for the question are different, the intention of both questions is the same.  
So the answers will be the same for both questions. That means we can just show the answers to the first question. That way the person who is asking the question will get the answers immediately and people who have answered already the first question don’t have to repeat themselves.

This problem is available on Kaggle as a competition. [https://www.kaggle.com/c/quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs)

So given two questions, our main objective is to find whether they are similar. So let’s do some magic with ML. 🪄

Business Objectives and Constraints
===================================

*   There is no strict latency requirement.
*   We would like to have interpretability but it is not absolutely mandatory.
*   The cost of misclassification is medium.
*   Both classes (duplicate or not) are equally important.

Data Overview
=============

Available Columns: **id, qid1, qid2, question1, question2, is\_duplicate**  
Class labels: **0, 1**  
Total training data / No. of rows: **404290**  
No. of columns: **6**  
**is\_duplicate** is the dependent variable.  
No. of non-duplicate data points is **255027**  
No. of duplicate data points is **149263**

We have **404290** training data points. And only **36.92%** are positive. That means it is an imbalanced dataset.

Business Metrics
================

It is a binary classification.

*   We need to minimize the log loss for this challenge.

Basic EDA
=========

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*qmAxfC775YvYVuuQ" width="700" height="467" srcSet="https://miro.medium.com/max/552/0\*qmAxfC775YvYVuuQ 276w, https://miro.medium.com/max/1104/0\*qmAxfC775YvYVuuQ 552w, https://miro.medium.com/max/1280/0\*qmAxfC775YvYVuuQ 640w, https://miro.medium.com/max/1400/0\*qmAxfC775YvYVuuQ 700w" sizes="700px" role="presentation"/>

Photo by [Andrew Neel](https://unsplash.com/@andrewtneel?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Test data don’t have question ids. So the independent variables are **question1**, **question2** and the dependent variable is **is\_duplicate**.

3 rows had null values. So We removed them and now We have **404287** question pairs for training.

*   **36.92%** of question pairs are duplicates and **63.08%** of questions pair non-duplicate.
*   Out of **808574** total questions (including both question1 and question2), **537929** are unique.
*   Most of the questions are repeated very few times. Only a few of them are repeated multiple times.
*   One question is repeated **157** times which is the max number of repetitions.

There are some questions with very few characters, which does not make sense. It will be taken care of later with Data Cleaning.

Data Cleaning
=============

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*CVcfGd\_POFSXouO0" width="700" height="467" srcSet="https://miro.medium.com/max/552/0\*CVcfGd\_POFSXouO0 276w, https://miro.medium.com/max/1104/0\*CVcfGd\_POFSXouO0 552w, https://miro.medium.com/max/1280/0\*CVcfGd\_POFSXouO0 640w, https://miro.medium.com/max/1400/0\*CVcfGd\_POFSXouO0 700w" sizes="700px" role="presentation"/>

Photo by [Pille R. Priske](https://unsplash.com/@pillepriske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

*   We have converted everything to lower case.
*   We have removed contractions.
*   We have replaced currency symbols with currency names.
*   We have also removed hyperlinks.
*   We have removed non-alphanumeric characters.
*   We have removed inflections with word lemmatizer.
*   We have also removed HTML tags.

Feature Extraction
==================

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*rYFoBHN4goDzRwR\_" width="700" height="467" srcSet="https://miro.medium.com/max/552/0\*rYFoBHN4goDzRwR\_ 276w, https://miro.medium.com/max/1104/0\*rYFoBHN4goDzRwR\_ 552w, https://miro.medium.com/max/1280/0\*rYFoBHN4goDzRwR\_ 640w, https://miro.medium.com/max/1400/0\*rYFoBHN4goDzRwR\_ 700w" sizes="700px" role="presentation"/>

Photo by [Stefan Rodriguez](https://unsplash.com/@stefantakespictures?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

We have created **23** features from the questions.

*   We have created features q1\_char\_num, q2\_char\_num with count of characters for both questions.
*   We have created features q1\_word\_num, q2\_word\_num with count of characters for both questions.
*   We have created total\_word\_num feature which is equal to sum of q1\_word\_num and q2\_word\_num.
*   We have created differ\_word\_num feature which is absolute difference between q1\_word\_num and q2\_word\_num.
*   We have created same\_first\_word feature which is 1 if both questions have same first word otherwise 0.
*   We have created same\_last\_word feature which is 1 if both questions have same last word otherwise 0.
*   We have created total\_unique\_word\_num feature which is equal to total number of unique words in both questions.
*   We have created total\_unique\_word\_withoutstopword\_num feature which is equal to total number of unique words in both questions without the stop words.
*   The total\_unique\_word\_num\_ratio is equal to total\_unique\_word\_num divided by total\_word\_num.
*   We have created common\_word\_num feature which is count of total common words in both questions.
*   The common\_word\_ratio feature is equal to common\_word\_num divided by total\_unique\_word\_num.
*   The common\_word\_ratio\_min is equal to common\_word\_num divided by minimum number of words between question 1 and question 2.
*   The common\_word\_ratio\_max is equal to common\_word\_num divided by maximum number of words between question 1 and question 2.
*   We have created common\_word\_withoutstopword\_num feature which is count of total common words in both questions excluding the stopwords.
*   The common\_word\_withoutstopword\_ratio feature is equal to common\_word\_withoutstopword\_num divided by total\_unique\_word\_withoutstopword\_num.
*   The common\_word\_withoutstopword\_ratio\_min is equal to common\_word\_withoutstopword\_num divided by minimum number of words between question 1 and question 2 excluding the stopwords.
*   The common\_word\_withoutstopword\_ratio\_max is equal to common\_word\_withoutstopword\_num divided by maximum number of words between question 1 and question 2 excluding the stopwords.
*   Then we have extracted fuzz\_ratio, fuzz\_partial\_ratio, fuzz\_token\_set\_ratio and fuzz\_token\_sort\_ratio features with fuzzywuzzy string matching tool. Reference: [https://github.com/seatgeek/fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)

EDA with Features
=================

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*yNpeMlNby1eavvAK" width="700" height="467" srcSet="https://miro.medium.com/max/552/0\*yNpeMlNby1eavvAK 276w, https://miro.medium.com/max/1104/0\*yNpeMlNby1eavvAK 552w, https://miro.medium.com/max/1280/0\*yNpeMlNby1eavvAK 640w, https://miro.medium.com/max/1400/0\*yNpeMlNby1eavvAK 700w" sizes="700px" role="presentation"/>

Photo by [Isaac Smith](https://unsplash.com/@isaacmsmith?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

*   If First word or Last word is the same then there is a high chance that the question pairs are duplicates.

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*ru-kvaSw53yiunQA-rrKVA.png" width="700" height="264" srcSet="https://miro.medium.com/max/552/1\*ru-kvaSw53yiunQA-rrKVA.png 276w, https://miro.medium.com/max/1104/1\*ru-kvaSw53yiunQA-rrKVA.png 552w, https://miro.medium.com/max/1280/1\*ru-kvaSw53yiunQA-rrKVA.png 640w, https://miro.medium.com/max/1400/1\*ru-kvaSw53yiunQA-rrKVA.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*c\_Dt34mDzOXPBSJkaQPSAw.png" width="700" height="267" srcSet="https://miro.medium.com/max/552/1\*c\_Dt34mDzOXPBSJkaQPSAw.png 276w, https://miro.medium.com/max/1104/1\*c\_Dt34mDzOXPBSJkaQPSAw.png 552w, https://miro.medium.com/max/1280/1\*c\_Dt34mDzOXPBSJkaQPSAw.png 640w, https://miro.medium.com/max/1400/1\*c\_Dt34mDzOXPBSJkaQPSAw.png 700w" sizes="700px" role="presentation"/>

*   The number of total unique words (q1 and q2 both combined) with and without stopwords is less if question pairs are duplicate.

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*-dQbmgg1emAFW9QiMAaWpw.png" width="700" height="260" srcSet="https://miro.medium.com/max/552/1\*-dQbmgg1emAFW9QiMAaWpw.png 276w, https://miro.medium.com/max/1104/1\*-dQbmgg1emAFW9QiMAaWpw.png 552w, https://miro.medium.com/max/1280/1\*-dQbmgg1emAFW9QiMAaWpw.png 640w, https://miro.medium.com/max/1400/1\*-dQbmgg1emAFW9QiMAaWpw.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*jybKeGlO7yxr6Clsao7pjw.png" width="700" height="262" srcSet="https://miro.medium.com/max/552/1\*jybKeGlO7yxr6Clsao7pjw.png 276w, https://miro.medium.com/max/1104/1\*jybKeGlO7yxr6Clsao7pjw.png 552w, https://miro.medium.com/max/1280/1\*jybKeGlO7yxr6Clsao7pjw.png 640w, https://miro.medium.com/max/1400/1\*jybKeGlO7yxr6Clsao7pjw.png 700w" sizes="700px" role="presentation"/>

*   For duplicate question pairs, the total unique words to total words ratio is generally smaller.

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*jCwZaWpbL37EHbSlOOqIjQ.png" width="700" height="262" srcSet="https://miro.medium.com/max/552/1\*jCwZaWpbL37EHbSlOOqIjQ.png 276w, https://miro.medium.com/max/1104/1\*jCwZaWpbL37EHbSlOOqIjQ.png 552w, https://miro.medium.com/max/1280/1\*jCwZaWpbL37EHbSlOOqIjQ.png 640w, https://miro.medium.com/max/1400/1\*jCwZaWpbL37EHbSlOOqIjQ.png 700w" sizes="700px" role="presentation"/>

*   Duplicate question pairs tend to have more common words between both the questions. Hence extracted features related to common words are also showing differences in distributions.

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*57FikO1mSmPLXAfiS48c-w.png" width="700" height="260" srcSet="https://miro.medium.com/max/552/1\*57FikO1mSmPLXAfiS48c-w.png 276w, https://miro.medium.com/max/1104/1\*57FikO1mSmPLXAfiS48c-w.png 552w, https://miro.medium.com/max/1280/1\*57FikO1mSmPLXAfiS48c-w.png 640w, https://miro.medium.com/max/1400/1\*57FikO1mSmPLXAfiS48c-w.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*lvjg3UtkjH0fZTW8qFU7uA.png" width="700" height="254" srcSet="https://miro.medium.com/max/552/1\*lvjg3UtkjH0fZTW8qFU7uA.png 276w, https://miro.medium.com/max/1104/1\*lvjg3UtkjH0fZTW8qFU7uA.png 552w, https://miro.medium.com/max/1280/1\*lvjg3UtkjH0fZTW8qFU7uA.png 640w, https://miro.medium.com/max/1400/1\*lvjg3UtkjH0fZTW8qFU7uA.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*tyjjZKi-vjjyz4t5pnXY8Q.png" width="700" height="255" srcSet="https://miro.medium.com/max/552/1\*tyjjZKi-vjjyz4t5pnXY8Q.png 276w, https://miro.medium.com/max/1104/1\*tyjjZKi-vjjyz4t5pnXY8Q.png 552w, https://miro.medium.com/max/1280/1\*tyjjZKi-vjjyz4t5pnXY8Q.png 640w, https://miro.medium.com/max/1400/1\*tyjjZKi-vjjyz4t5pnXY8Q.png 700w" sizes="700px" role="presentation"/>

*   The fuzz ratios tend to be generally higher for duplicate question pairs.

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*tg7TNhx5ACexPNoEmDc0Fg.png" width="700" height="258" srcSet="https://miro.medium.com/max/552/1\*tg7TNhx5ACexPNoEmDc0Fg.png 276w, https://miro.medium.com/max/1104/1\*tg7TNhx5ACexPNoEmDc0Fg.png 552w, https://miro.medium.com/max/1280/1\*tg7TNhx5ACexPNoEmDc0Fg.png 640w, https://miro.medium.com/max/1400/1\*tg7TNhx5ACexPNoEmDc0Fg.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*uPiopOLjUu9r3zsTPp\_Tnw.png" width="700" height="260" srcSet="https://miro.medium.com/max/552/1\*uPiopOLjUu9r3zsTPp\_Tnw.png 276w, https://miro.medium.com/max/1104/1\*uPiopOLjUu9r3zsTPp\_Tnw.png 552w, https://miro.medium.com/max/1280/1\*uPiopOLjUu9r3zsTPp\_Tnw.png 640w, https://miro.medium.com/max/1400/1\*uPiopOLjUu9r3zsTPp\_Tnw.png 700w" sizes="700px" role="presentation"/>

Featurization with SentenceBERT
===============================

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*gt5yGUHGZyIh2B4M" width="700" height="468" srcSet="https://miro.medium.com/max/552/0\*gt5yGUHGZyIh2B4M 276w, https://miro.medium.com/max/1104/0\*gt5yGUHGZyIh2B4M 552w, https://miro.medium.com/max/1280/0\*gt5yGUHGZyIh2B4M 640w, https://miro.medium.com/max/1400/0\*gt5yGUHGZyIh2B4M 700w" sizes="700px" role="presentation"/>

Photo by [Mika Baumeister](https://unsplash.com/@mbaumi?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

We need to convert the questions to some numeric form to apply machine learning models. There are various options from basic like Bag of Words to Universal Sentence Encoder.

I tried InferSent sentence embeddings. But it returns 4096 dimension representation. And after applying it the train data became huge. So I discarded it. And I chose SentenceBERT for this problem.

SentenceBERT is a BERT based sentence embedding technique. We will use pre-trained SentenceBERT model _paraphrase-mpnet-base-v2_, which is recommended for best quality. The SentenceBERT produces an output of 768 dimensions. [https://www.sbert.net/](https://www.sbert.net/)

We created two more features **cosine\_simlarity\_bert** and **euclidean\_distance\_bert** which measures similarity and distance between both pairs of questions with SentenceBert representation.

The total number of features till now is **25**.

EDA on new features related toSentenceBERT
------------------------------------------

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*M0UJDA70w2cb7\_DBugXhHg.png" width="700" height="253" srcSet="https://miro.medium.com/max/552/1\*M0UJDA70w2cb7\_DBugXhHg.png 276w, https://miro.medium.com/max/1104/1\*M0UJDA70w2cb7\_DBugXhHg.png 552w, https://miro.medium.com/max/1280/1\*M0UJDA70w2cb7\_DBugXhHg.png 640w, https://miro.medium.com/max/1400/1\*M0UJDA70w2cb7\_DBugXhHg.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1000/1\*xopO6BzMshmvuldDDtZ5Kg.png" width="500" height="262" srcSet="https://miro.medium.com/max/552/1\*xopO6BzMshmvuldDDtZ5Kg.png 276w, https://miro.medium.com/max/1000/1\*xopO6BzMshmvuldDDtZ5Kg.png 500w" sizes="500px" role="presentation"/>

*   **Cosine Similarity** is larger for duplicate pairs.
*   80% of non-duplicate question pairs and only 20% of duplicate question pairs have cosine similarity of <= .815

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/1\*3zxx\_rKCKlqv3Rw-MMw\_TQ.png" width="700" height="259" srcSet="https://miro.medium.com/max/552/1\*3zxx\_rKCKlqv3Rw-MMw\_TQ.png 276w, https://miro.medium.com/max/1104/1\*3zxx\_rKCKlqv3Rw-MMw\_TQ.png 552w, https://miro.medium.com/max/1280/1\*3zxx\_rKCKlqv3Rw-MMw\_TQ.png 640w, https://miro.medium.com/max/1400/1\*3zxx\_rKCKlqv3Rw-MMw\_TQ.png 700w" sizes="700px" role="presentation"/>

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1000/1\*WmFT1rvf-FB2eQIIyTC4mA.png" width="500" height="256" srcSet="https://miro.medium.com/max/552/1\*WmFT1rvf-FB2eQIIyTC4mA.png 276w, https://miro.medium.com/max/1000/1\*WmFT1rvf-FB2eQIIyTC4mA.png 500w" sizes="500px" role="presentation"/>

*   **Euclidean Distance** is smaller for duplicate pairs.
*   20% of non-duplicate question pairs and approx 80% of duplicate question pairs have euclidean distance of <= 2.

It is showing the Pareto Principle (80–20 rule).

Data Pre-processing
===================

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*v1gvhdSK3Y7YNmgi" width="700" height="467" srcSet="https://miro.medium.com/max/552/0\*v1gvhdSK3Y7YNmgi 276w, https://miro.medium.com/max/1104/0\*v1gvhdSK3Y7YNmgi 552w, https://miro.medium.com/max/1280/0\*v1gvhdSK3Y7YNmgi 640w, https://miro.medium.com/max/1400/0\*v1gvhdSK3Y7YNmgi 700w" sizes="700px" role="presentation"/>

Photo by [Braden Collum](https://unsplash.com/@bradencollum?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

We normalized (min-max scaling) the extracted features. We have not normalized the embeddings because it is not recommended.

We have **1561** features (25 + 768 + 768).

*   **25** are extracted features.
*   **768+768** for sentence embedding of question 1 and question 2.

Since the dataset was imbalanced. We did **oversample** by sampling from the minority class.  
Now we have **510048** data points for training. **255024** from each class.

Note that I have not set aside any data for testing locally. Because our main goal is to get a good score on Kaggle.

Training Models
===============

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*WXiK1FtQPFQzc1B6" width="700" height="468" srcSet="https://miro.medium.com/max/552/0\*WXiK1FtQPFQzc1B6 276w, https://miro.medium.com/max/1104/0\*WXiK1FtQPFQzc1B6 552w, https://miro.medium.com/max/1280/0\*WXiK1FtQPFQzc1B6 640w, https://miro.medium.com/max/1400/0\*WXiK1FtQPFQzc1B6 700w" sizes="700px" role="presentation"/>

Photo by [Karsten Winegeart](https://unsplash.com/@karsten116?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Support Vector Classifier
-------------------------

While training Halving Grid Search CV with param grid,

```
svc\_param\_grid = {‘C’:\[1e-2, 1e-1, 1e0, 1e1, 1e2\]}
```

We have used LinearSVC because it is recommended for large datasets. We have used the L2 penalty and the loss function is squared of hinge loss. Also, it is recommended to use primal formulation for large datasets. For some values of C it was not conversing so I increased max\_iter to 3000.

```
svc\_clf = LinearSVC(penalty='l2', loss='squared\_hinge', dual=False, max\_iter=3000)
```

For cross-validation in halving grid search cv, I have used 1 shuffle split with a 70:30 split. Also, the scoring for selection is accuracy.

```
svc\_clf\_search = HalvingGridSearchCV(svc\_clf, svc\_param\_grid, cv=splits, factor=2, scoring='accuracy', verbose=3)
```

The halving grid search cv found **C=100** to be the best param. And the best accuracy is **85.79%**. So the best estimator looks like,

```
LinearSVC(C=100.0, dual=False, max\_iter=3000)
```

Now since we need to minimize log loss for the competition. We would want a good predicted probability. Calibrated Classifier can be used to get a good predicted probability.

```
svc\_calibrated = CalibratedClassifierCV(base\_estimator=svc\_clf\_model, method="sigmoid", cv=splits)
```

After calibration of the model for probabilities. I predicted probabilities of test data and submitted on Kaggle. The public leader board score for the Kaggle submission is **0.36980**.  It is very good considering that the model assumes linear separability.

Random Forest
-------------

You know Quora itself usage Random Forest for this problem. Or at least they did when they first posted the competition on Kaggle in June 2017.

Same as before we are using halving grid search cv with following param grid,

```
rf\_param\_grid = { 'n\_estimators':\[200, 500, 800\], 'min\_samples\_split':\[5, 15\], 'max\_depth': \[70, 150, None\]}
```

And the rest of the params are the default for the Random Forest Classifier.

```
rf\_clf = RandomForestClassifier()
```

We have used the very similar halving grid search cv as before,

```
rf\_clf\_search = HalvingGridSearchCV(rf\_clf, rf\_param\_grid, cv=splits, factor=2, scoring='accuracy', verbose=3)
```

The halving grid search cv found **{‘max\_depth’: 150, ‘min\_samples\_split’: 5, ‘n\_estimators’: 800}** to be the best params. And the best accuracy is **90.53%**. So the accuracy has increased by 5% as compared to SVM. The best estimator looks like,

```
RandomForestClassifier(max\_depth=150, min\_samples\_split=5, n\_estimators=800)
```

Now at this point, I should have used calibration but because it has already taken a lot of time I skipped it. I should have used Bayesian Optimisation technique 😞.

The public leader board score for the Kaggle submission is **0.32372**, which slightly better than SVC. I was expecting a little less logloss but remember we have not done calibration (due to time constraints). We will try to better with XGBoost — the holy grail of ml models for the Kaggle competition.

XGBoost
-------

Due to time and system configuration constrained, I decided to use 200000 data points to estimate a few of the params.  
At first, I was using Optuna for hyperparameter tuning but it had some issues because of which it was not releasing memory after the trials. So the system was crash after few trials.  
Later on, I decided to use HyperOpt for the tuning.

With HyperOpt, I tuned only **max\_depth** and **learning\_rate**. It was not a fine-tune because I used only 5 trials. But it gave a rough idea.

Finally, I choose the following params for training the model on whole data,

```
params = dict( objective = "binary:logistic", eval\_metric = "logloss", booster = "gbtree", tree\_method = "hist", grow\_policy = "lossguide", max\_depth = 4, eta = 0.14)
```

The **objective = “binary:logistic”** because we are trying to get probabilities. I have used **tree\_method = “hist”** for faster training. **grow\_policy = “lossguide”** is inspired from LightGBM for better accuracy.

The **num\_boost\_round** is set to 600 with **early\_stopping\_rounds** as 20.

The public leader board score for the Kaggle submission is **0.32105**, which slightly better than the other models. I was expecting a better result than this. Which is possible with more fine-tuning the hyperparameters. XGBoost have tons of hyperparameters [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)

Another XGBoost
---------------

I was not happy with the result of the XGBoost model so I decided to tune the parameters with gut feeling.

The first thing I did is that I got rid of oversampled data by removing the duplicate rows.

This time I added a few more parameters to generalize better,

```
params = dict( objective = "binary:logistic", eval\_metric = "logloss", booster = "gbtree", tree\_method = "hist", grow\_policy = "lossguide", max\_depth = 4, eta = 0.15, subsample = .8, colsample\_bytree = .8, reg\_lambda = 1, reg\_alpha = 1)
```

Also, I decreased the number of boosting round to 500.

<img alt="" class="es ff fb fl v" src="https://miro.medium.com/max/1400/0\*sqk8L10gLSJVvNda" width="700" height="394" srcSet="https://miro.medium.com/max/552/0\*sqk8L10gLSJVvNda 276w, https://miro.medium.com/max/1104/0\*sqk8L10gLSJVvNda 552w, https://miro.medium.com/max/1280/0\*sqk8L10gLSJVvNda 640w, https://miro.medium.com/max/1400/0\*sqk8L10gLSJVvNda 700w" sizes="700px" role="presentation"/>

Photo by [Fauzan Saari](https://unsplash.com/@fznsr_?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

🥁 Voila! We have a winner. **This submission resulted in public LB score of 0.28170**.  
This seems a very good result.

Final Thoughts
==============

I learned a lot from this case study. I took some shortcuts either because of system configuration constraints or some time constraints.

I also experienced firsthand that machine learning is not all about model building but steps before that take more time. The hyperparameter tuning can be automated but things like feature extraction or deciding on what featurization to use need to be done manually.

I spent almost two weeks 😅 and half of that time I was waiting for some execution to complete. So I think it’s a good idea to use things like Amazon SageMaker if you have resource-intensive tasks.

In the future, we can try some deep learning-based models.

References
==========

a. [https://appliedroots.com/](https://appliedroots.com/)
