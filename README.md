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

### Understanding You through Implicit and Explicit Ratings

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

- **Coverage** - % of <user, item> pairs that can be predicted. That's just the percentage of possible recommendations that your system is able to provide. If your enforce a *Higher Quality Threshold* on the Recommendations you make, then you might improve your accuracy at the expense of coverage. Finding the balance of where exactly you're better off recommending nothing at all can be delicate. **Coverage** can also be important to watch, beacuse it gives you a sense of how quickly new items in your catalog will start to appear in recommendations (e.g. When a new book comes out on Amazon, it won't appear in recommendations until at least a few people but it, therefore establishing patterns with the purchase of other items. Until those patterns exist, that new book will reduce Amazon's coverage metric

- **Diversity (1 - S)** - *S = avg similarity between recommendation pairs*. You can think of this as a measure of how broad a variety of items your Recommender System is putting in front of People. (e.g. Low diversity would be a Recommender System that just recommends the next books in a series that you've started reading, but doesn't recommend books from different authors, or movies related to what you've read. This may seem like a subjective thing, but it is measurable). Many Recommender Systems start by computing some sort of similarity metric between items, so we can use these similarity scores to measure diversity. If we look at the similarity scores of every possible pair in a list of **Top-N** Recommendations, we can average them to get a measure of how similar the recommended items in the list are tot each other. We can call that measure **S**. **Diversity** is basically the opposite of average similarity, so we subtract it from 1 to get a number associated with **Diversity**. It's important to realize that **Diversity**, at least in the context of Recommender Systems, isn't always a good thing. You can achieve very high **Diversity** by just recommending completely random items. But those aren't good recommendations by any stretch of the imagination. Unusually high **Diversity** scores mean that you just have bad recommendations more often than not. You always need to look at **Diversity** alongside Metrics that measure the quality of the recommendations as well.

- **Novelty** - *Mean Popularity Rank of Recommended Items*. Similarly, novelty sounds like a good thing, but often it isn't. Novelty is a measure of how popular the items are that you're recommending. And again, just recommending random stuff would yield very high **Novelty** scores since that vast majority ot items are not Top Sellers. Although novelty is measurable, what to do with it is in many subjective. There's a concept of **User Trust** in a Recommender System. People want to see at least a few familiar items in their recommendations that make them say, *"Yeah, that's a good recommendation for me. This system seems good"*. If you only recommend things people have never heard of, they may conclude that your system doesn't really know them, and they may **engage** less with your recommendations as a result. Also, popular items are usually popular for a reason. They're enjoyable by a large segment of the population, so you would expect them to be good recommendations for a large segment of the population who hasn't read or watched them yet. If you're not recommending some popular items, you should probably question whether your Recommender System is really working as it should. This is an important point. You need to strike a balance between familiar, popular items and what we call serendipitous discovery of new items the user has never heard of before. The familiar items estabish trust with the user, and the new ones allow the user tot discover entirely new things that they might love. 

- **The Long Tail** - **Novelty** is important, though, because the whole point of recommender systems is to surface items inn what we call **"The Long Tail"**. Imagine this is a plot of the sales of items of every item in your catalog, sorted by sales. So the number of sales, or popularity, is on the Y axis, and all the products are along the X axis. You almost always see an exponential distribution like this. Most sales come from a very small number of items, but taken together, the **"Long Tail"** makes up a large amount of sales as well. Items in that long tail, the yellow part in the graph, are items that cater to people with unique niche interests. Recommender systems can help people discover those items in the long tail that are relevant to their own unique niche interests. If you can do that successfully, then the recommendations your system makes can help new authors get discovered, can help people explore their own passions, and make money for whoever you're building the system for as well. Everybody wins. When done right, recommender systems with good **Novelty** scores can actually make the world a better place. But again, you need to strike a balance between **Novelty** and **Trust**. Building Recommender Systems is a bit of an art, and this is an example of why.

### Churn, Responsiveness, and A/B Tests

- **Churn** - *How Often Do Recommendations Change?* - In part, **Churn** can measure how sensitive your recommender system is to new user behavior. If a user rates a new movie, does that substantially change their recommendations? If so, then your **Churn** Score will be high. Maybe just showing someone the same recommendations too many times is a bad idea in itself. If a user keeps seeing the same recommendation but doesn't clik on it, at some point should you just stop trying to recommend it and show the user something else instead? Sometimes a little bit of randomization in your **Top-N** Recommendations can keep them looking fresh, and expose your users to more items than they would have seen otherwise. But, just like **Diversity** and **Novelty**, high **Churn** is not itself a good thing. You could maximize your **Churn** metric by just recommending items completely at random, and of course, those would not be good recommendations. All of these metrics need to be looked at together, and you need to understand the trade-offs between them. 

- **Responsiveness** - *How Quickly does New User Behavior Influence your Recommendations* - If you rate a new movie, does it affect your Recommendations immediately or does it only affect your Recommendations the next day after some Nightly Job runs? More **Responsiveness** would always seem to be a good thing, but in the world of Business you have to decide how **Responsive** your Recommender really needs to be, since Recommender Systems that have instantaneous **Responsiveness** are complex, difficult to maintain, and expensive to build. You need to strike your own balance between **Responsiveness** and **Simplicity**. 

- **What's Important?** - We've covered a lot of different ways to evaluate your recommender system:
  * **MAE**
  * **RMSE**
  * **Hit Rate (in various forms)**
  * **Coverage**
  * **Diversity**
  * **Novelty**
  * **Churn**
  * **Responsiveness**
  * How do you know what to focus on? Well, that answer is that it depends. It may even depend on **Cultural Factors**. Some cultures may **want** more **Diversity** and **Novelty** in their Recommendations than others, while other cultures may want to stick with things that are familiar with them. It also depends on **what you're trying to achieve** as a Business. And usually, a business is just trying to make money which leads to one more way to evaluate Recommender Systems that is arguably the most important of all. 
  
- **Online A/B tests!** - To Tune your Recommender System using your real customers, and measuring how they react to your recommendations. You can put recommendations from different algorithms in front of different sets of users, and measure if they actually buy, watch, or otherwise indicate interest in the recommendations you've presented. By **always testing changes** to your Recommender System using conntrolled, online experiments, you can see if they actually cause people to discover and **purchase more new things** than they would have otherwise. **That's ultimately what matters** to your business, and it's ultimately what matters to your users, too. None of the metrics we've discussed matter more than how real customers react to the recommendations you produce in the real world. You can have the most accurate rating predictions in the world, but if customers can't find new items to buy or watch from your system, it will be worthless from a practical standpoint. If you test a new algorithm and it's more complex than the one it replaced, then you should discard it if it doesn not result in a **measurable improvement** in users interacting with the recommedations you present. **Online tests** can help you avoid introducing **complexity** that adds no value, and remember, **complex systems are difficult to maintain**. So remember, offline metrics such as accuracy, diversity, and novelty can all be inndicators you can look at while developing Recommender Systems Offline, but **you should never declare victory until you've measured a real impact on real users from your work**. Systems that look good in an offline setting often fail to have any impact in an online setting, that is, in the real world. **User behavior is the ultimate test of your work**. There is a real effect where often accuracy metrics tell you that an algorithm is great, only to have it do horribly in an online test. YouTube studied this, and calls it the **"surrogate problem"**. Accurately predicted ratings don't necessarily make for good video recommendations. YouTube said in one of their papers, and I quote, *"There is more art than science in selecting the surrogate problem for recommendations"* - YouTube. What they mean is that you can't always use **accuracy** as a surrogate for good recommendations. Netflix came to the same conclusion, which is why they aren't really using the results of that one million dollar prize for accuracy they paid out. At the end of the day, the results of online **A/B tests** are the only evaluation that matters for your Recommender System.

- **Perceived Quality** - Another thing you can do is just straight up ask your users if they think specific recommendations are good. Just like you can ask for explicit feedback on items with ratings, you can ask users to rate your recommendations, too. This is called measuring **"Perceived Quality"** and it seems like a good idea on paper, since, as you've learned, defining what makes a "good" recommendation is by no means clear. In practice though, it's a tough thing to do. Users will probably be confused over whether you're asking them to rate the item or rate the recommendation, so you won't really know how to interpret this data. It also requires extra work from your customers with no clear payoff for them, so you're unlikely to get enough ratings on your recommendations to be useful. It's best to just stick with the **Online A/B tests**, and measure how your customers **vote with their wallets** on the **Quality** of your Recommendations.

### Review ways to measure your recommender - Q&A's

**Which Metric was used to Evaluate the Netflix Prize?**

- *R: Root mean Squared Error (RSME)*

**What's a Metric for Top-N recommenders that accounts for the Rank of predicted items?**

- *R: Average Reciprocal Hit Rank (ARHR)*

**Which Metric measures how Popular or Obscure your recommendations are?**

- *R: Novelty*

**Which Metric would tell us if we're recommending the same types of things all the time?**

- *R: Diversity*

**Which Metric matters more than anything?**

- *R: The Results of Online A/B Tests*

### Walkthrough of RecommenderMetrics.py

- **Surprise** - A Python scikit for recommender systems - http://surpriselib.com/

Think of it as **Test Driven Development**, we're going to write our tests before we write any actual Recommender Systems, and that's generally a good idea so that you focus on the results you want to achieve. We're going to use an Open Source Python Library called Surprise to make life easier. 

**Surprise** is built around measuring the **accuracy** of Recommender Systems, and although I've said repeatedly that this is the wrong thing to focus on, it's really the best we can do without access to a real, large-scale website of our own. And it can give us some information about the properties of the algorithms that we're going to work with. It also saves us a lot of work (e.g. it can compute things like MAE and RMSE for us)

Start Spyder in our RecSys Environment

Code > Evaluating > `RecommenderMetrics.py`

```python
import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)


        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n

```

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
