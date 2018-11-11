# recommender-systems-ml-ai

## Getting Started

### Install Anaconda, Course Materials and Create Movie Recommendations

- Tensorflow

- DSSTNE

- SageMaker

- Apache Spark

**Recommender Systems - Getting Set Up**

- Install Anaconda for Python 3 - https://www.anaconda.com/download/

- Create an Environment - Open Anaconda Navigator, select “Environments,” and create a new “RecSys” environment for Python 3.6 or newer

- Install Scikit-Surprise - From the RecSys environment you just made, click the arrow next to it and select “Open Terminal.” In the terminal, run:

```
conda install -c conda-forge scikit-surprise
```

Hit ‘y’ to proceed if prompted.

- Download Course Materials 

- Other Resources - https://www.amazon.com/gp/product/B07GCV5JCZ/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B07GCV5JCZ&linkCode=as2&tag=sundog07-20&linkId=0aa54fda30b476b37e5ea02feb2c53d3

**Singular Value Decomposition (SVD)**

- 100,000 movie ratings with just about 60 lines of code

- It starts by choosing an arbitrary user *whom we're going to get to know very well* and summarizing the movies that he loved and the movies he hated, so you can get a sense of his taste

But it's hard to tell if a movie you've never heard of is a good recommendation or not

Just defining what makes a good recommendation is huge problem that's really central to the field of Recommender Systems

In many ways, building Recommender Systems is more Art than Science
You're trying to get inside people's heads and build models of their preferences

### Course Roadmap

- Getting Started

- Intro to Python

- Evaluating Recommender Systems

- Building a Recommendation Engine

- Content-Based Filtering

- Neighborhood-Based Collaborative Filtering

- Model-Based Methods

- Intro to Deep Learning

- Recommendations with Deep Learning

- Scaling it Up

- Challenges of Recommender Systems

- Case Studies

- Hybrid Solutions

- More

### Type of Recommenders

**Many Flavors of Recommenders**

- Things

- Content

- Music

- People (Online Dating)

- Search Results

### Undestanding You through Implicit and Explicit Ratings

- Understanding You

- Understanding You ... Explicitly (Feedback)

- Understanding You ... Implicitly (Looking at the things you do and interpreting them)

A big part of why Amazon's recommendations are so good is because they have so much purchase data to work with and it's such a great signal for someone's interests

They don't even need great recommendation algorithms, because the data they're working with is so great to begin with

### Top-N Recommender Architecture

The job of the Top-N Recommender Systems is to produce a finite list of the best things to present to a given person

Customers don't want to see your ability to predict their rating for an item

They just want to see things they're likely to love

**GOAL:** Put the Best Content we can find in front of users in the form of a top-N list

**(One) Anatomy of a top-N Recommender**

- Individual Interest - Thing they bough in the part - Usually a Big Distributed NoSQL like Cassandra, MongoDB or memcached; ideally, this interest data is normalized using techniques such as Mean Centering or Z Scores to ensure that the Data is comparable between users

- Candidate Generation - Items we think might be interestinig to the user 

- Item Similarities - Items that are similar

- Candidate Ranking - Many Candidates will appear more than once and need to be combined together in some way 

- Filtering - Take the N best recommendations

**Another way to do it - Theorical Approach**

- Candidate Generation 

- Rating Predictions

- Candidate Ranking

- Filtering

### Review the basics of Recommender Systems

- Purchasing, viewing, and clicking are all **Implicit Ratings**. We can infer you like something because you chose to interact with it somehow

- Star Reviews are **Explicit Ratings**. They provide very good data for Recommender Systems, they require extra work from your users

- Examples: Netflix's Home Page, Google Search, Amazon's "people who also bought...", Pandora, YouTube


## Introduction to Python [Optional]

  ### The Basics of Python
  
  ### Data Structures in Python
  
  ### Functions in Python
  
  ### Booleans, loops and hands-on challenge

## Evaluating Recommender Systems

### Train/Test and Cross Validation

**Train/Test - Methodology for Testing Recommender Systems Offline**

- Full Data Set (Movie Ratings, etc.)

- Training Set (80-90 %)

- Machine Learning

- Predictions

- Test Set

**K-Fold Cross-Validation**

- Full Data Set (Movie Ratings, etc.)

- Fold 1 -> Machine Learning -> Measure Accuracy

- Fold 2 -> Machine Learning -> Measure Accuracy

- Test Set

- Fold k-1 -> Machine Learning -> Measure Accuracy

- Take Average of Measured Accuracies

### Accuracy Metrics (RMSE, MAE)

**Mean Absolute Error (MAE)**

The most straightforward metric is Mean Absolute Error or MAE:

- Let's say we have **n** ratings in our test set that we want to evaluate

- For each rating we call the rating or system predicts **y**

- And the rating the user actually gave **x**

- Just take the absolute value of the difference between the two, to measure the error for that rating prediction

It's literally just the difference between the predicted rating and the actual rating

We sum those errors up across all n ratings in our test set, and divide by n to get the average, or mean

So mean absolute error is exactly that, the mean or average absolute values of each error in rating predictions 

**Root Mean Square Error (RMSE)**

This is a more popular metric for a few reasons, but one is that it penalizes you more when your rating prediction is way off, and penalizes you less when you are reasonably close

The differences is that instead of summing up the absolute values of each rating prediction error, we sum up the squares of the rating prediction errors instead

Taking the square we ensures we end up with positive numbers like absolute values do and it also inflates the penalty for larger errors

When we're done we take the square root to get back to a number that makes sense

### Top-N Hit Rate - Many Ways

**Evaluating Top-N Recommenders**

- **Hit Rate** - Generate Top End Recommendations for all of the users in your test set. If one of the recommendations in a users top-end recommendations is something they actually rated, you consider that a hit. You actually managed to show the user something they found interesting enough to watch on their own already so we'll consider that a success **hits/users**. Just add up all the hits in your top-end recommendations for every user in your test set, divide by the number of users, and that's your hit rate

- **Leave-One-Out Cross Validation** - Compute Top-End Recommendations for each user in our Training Data and intentionally remove one of those items from that users training data. We then test our recommenders system's ability to recommend that item that was left out in the Top-End Results it creates for that user in the testing phase. So we measure our ability to recommend an item in a top-end list for each user that was left our from the training data. That's why it's called **"leave-one-out"**. The trouble is it's a lot harder to get one specific movie right while testing than to just get one of the recommendations. So **"hit rate"** with **"leave-one-out"** tends to be very small and difficult to measure unless you have a very large data set to work with. But it's a much more *user focused metric* when you know your recommender system will be producing Top-End Lists in the real world, which most of them do

- **Average Reciprocal Hit Rate (ARHR)** - A variation on **"hit rate"**. This metric is just like **"hit rate"** but it accounts for where in the Top-End list your hits appear. So you end up getting more credit for successfully recommendinng an item in the top slot tha in the bottom slot. Again, this is a more *user focused metric* since users tend to focus on the beginning of lists. The only difference is that instead of summing up the number of hits we sum up the reciprocal rank of each hit. So if we successfully predict our recommedation in slot three, that only counts as 1/3. But a hit in slot one of our *Top-End Recommendations* receives the full weight of **1.0**. Whether this metric makes sense for you depends a lot on how your *Top-End Recommendations* are displayed. If the user has to scroll or paginate to see the lower items in your *Top-End List* then it makes sense to penalize good recommendations that appear too low in the list where the user has to work to find them

- **Cumulative Hit Rate (cHR)** - We throw away hits if our predicted rating is below some threshold. The idea is that we shouldn't get credit for recommending items to a user that we think they won't actually enjoy

- **Rating Hit Rate (rHR)** - Break it down by predicted rating score. It can be a good way to get an idea of the distribution of how good your algorithm thinks recommended movies are that actually get a hit. Ideally you want to recommend movies that they actually liked and breaking down the distribution gives you some sense of how well you're doing in more detail

Those are all different ways to measure the effectiveness of *Top-End Recommendations* Offline

The World of *Recommender Systems* would be probably be a little bit different if Netflix awarded the Netflix prize on hit rate instead of RMSC. It turns out that small improvements in RMSC can actually result in large improvements to hit rate which is what really matters. But it also turns out that you can build *Recommender Systems* with great *Hit Rates* but poor RMSC scores and we'll see some of those later 

So RMSC and hit rate aren't always related

### Covarage, Diversity, and Novelty

### Churn, Responsiveness, and A/B Tests

### Review ways to measure your recommender

### Walkthrough of RecommenderMetrics.py

### Walkthrough of TestMetrics.py

### Measure the Performance of SVD Recommendations


  
## A Recommender Engine Framework

  ### Our Recommender Engine Architecture
  
  ### Recommender Engine Walkthrough, Part 1
  
  ### Recommender Engine Walkthrough, Part 2
  
  ### Review the Results of our Algorithm Evaluation
  
  
  
  

## Content-Based Filtering

  ### Content-Based Recommendations, and the Cosine Similarity Metric
  
  ### K-Nearest-Neighbors and Content Recs
  
  ### Producing and Evaluating Content-Based Movie Recommendations
  
  ### Bleeding Edge Alert! Mise en Scene
  
  ### Dive Deeper into Content-Based Recommendations

## Neighborhood-Based Collaborative Filtering}

  ### Measuring Similarity, and Sparsity
  
  ### Similarity Metrics
  
  ### User-based Collaborative Filtering
  
  ### User-based Collaborative Filtering, Hands-On
  
  ### Item-based Collaborative Filtering
  
  ### Item-based Collaborative Filtering, Hands-On
  
  ### Tuning Collaborative Filtering Algorithms
  
  ### Evaluating Collaborative Filtering Systems Offline
  
  ### Measure the Hit Rate of Item-Based Collaborative Filtering
  
  ### KNN Recommenders
  
  ### Running User and Item-Based KNN on MovieLens
  
  ### Experiment with different KNN parameters
  
  ### Bleeding Edge Alert! Translation-Based Recommendations

## Matrix Factorization Methods

  ### Principal Component Analysis (PCA)
  
  ### Singular Value Decomposition
  
  ### Running SVD and SVD++ on MovieLens
  
  ### Improving on SVD
  
  ### Tune the hyperparameters on SVD
  
  ### Bleeding Edge Alert! Sparse Linear Methods (SLIM)

## Introduction to Deep Learning [Optional]

  ### Deep Learning Introduction
  
  ### Deep Learning Pre-Requisites
  
  ### History of Artificial Neural Networks
  
  ### Playing with Tensorflow
  
  ### Training Neural Networks
  
  ### Tuning Neural Networks
  
  ### Introduction to Tensorflow
  
  ### Handwriting Recognition with Tensorflow, part 1
  
  ### Handwriting Recognition with Tensorflow, part 2
  
  ### Introduction to Keras
  
  ### Handwriting Recognition with Keras
  
  ### Classifier Patterns with Keras
  
  ### Predict Political Parties of Politicians with Keras
  
  ### Intro to Convolutional Neural Networks (CNN's)
  
  ### CNN Architectures
  
  ### Handwriting Recognition with Convolutional Neural Networks (CNNs)
  
  ### Training Recurrent Neural Networks
  
  ### Sentiment Analysis of Movie Reviews using RNN's and Keras

## Deep Learning for Recommeder Systems

  ### Intro to Deep Learning for Recommenders
  
  ### Restricted Boltzmann Machines (RBM's)
  
  ### Recommendations with RBM's, part 1
  
  ### Recommendations with RBM's, part 2
  
  ### Evaluating the RBM Recommender
  
  ### Tuning Restricted Boltzmann Machines
  
  ### Exercise Results: Tuning a RBM Recommender
  
  ### Auto-Encoders for Recommendations: Deep Learning for Recs
  
  ### Recommendations with Deep Neural Networks
  
  ### Clickstream Recommendations with RNN's
  
  ### Get GRU4Rec Working on your Desktop
  
  ### Exercise Results: GRU4Rec in Action
  
  ### Bleeding Edge Alert! Deep Factorization Machines
  
  ### More Emerging Tech to Watch

## Scaling it Up

  ### Introduction and Installation of Apache Spark
  
  ### Apache Spark Architecture
  
  ### Movie Recommendations with Spark, Matrix Factorization, and ALS
  
  ### Recommendations from 20 millions ratings with Spark
  
  ### Amazon DSSTNE
  
  ### DSSTNE in Action
  
  ### Scaling Up DSSTNE
  
  ### AWS SageMaker and Factorization Machines
  
  ### SageMaker in Action: Factorization Machines on one million ratings, in the cloud

## Real-World Challenges of Recommender Systems

  ### The Cold Start Problem (and solutions)
  
  ### Implement Random Exploration
  
  ### Exercise Solution: Random Exploration
  
  ### Stoplists
  
  ### Implement a Stoplist
  
  ### Exercise Solution: Implement a Stoplist
  
  ### Filter Bubbles, Trust, and Outliers
  
  ### Identify and Eliminate Outlier Users
  
  ### Exercise Solution: Outlier Removal
  
  ### Fraud, The Perils of Clickstream, and International Concerns
  
  ### Temporal Effects, and Value-Aware Recommendations

## Case Studies

  ### Case Study: YouTube, Part 1
  
  ### Case Study: YouTube, Part 2
  
  ### Case Study: Netflix, Part 1
  
  ### Case Study: Netflix, Part 2

## Hybrid Approaches

  ### Hybrid Recommenders and Exercise
  
  ### Exercise Solution: Hybrid Recommenders

## Wrapping Up

  ### More to Explore
  
  ### Bonus Lecture: Companion Book and More Courses 
