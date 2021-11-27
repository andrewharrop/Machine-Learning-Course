# Classification

> ## Simple linear classifiers

 ### Logistic regression:
- One of the most used classifiers
- Extensive applications
- Similrily to what we looked at in the introduction, underlying algorithams are applied to optimize parameters
- Example: Determine whether a review is positive or negative

Decision boundries are the point at which above one decision is made and below another is. 

There can be more than two coefficients. With two coefficients, the decision boundry is a line. With three, it is a plane. With more than three, it is a hyperplane. Decision boundries typically take complicated shapes.

    prediction=Sign(Score(x))

Where:

Score(x<sup>(i)</sup>) = &theta;<sup>T</sup>x

prediction = Sign(&theta;<sup>T</sup>x)

> #### Applying probability 

Now that we have a classification model, how can we improve it? 

We can add probabilities to it. What is the probability that our classification is right. That is, given input x, what is the probability that output y=+1

    P(y=+1|x)
    Where P(A|B) = P(AnB)/P(B)

We can apply this probability to class predicitons

But how do we implement probabilities to classifications? 

We can apply certain functions to standardize the score.
One example is the sigmoid scoring funcition. It maps any real number to a value between 0 and 1 by applying this function:

Sigmoid(x) = 1/(1+e<sup>-x</sup>)

It maintains transitive properties, although there could be problems with disproportionate scaling, especially with high feature variance. Restated for our classification algoritham, it looks like this:

Sigmoid(f<sub>&theta;</sub>(x)) = 1/(1+e<sup>-f<sub>&theta;</sub>(x)</sup>)

Applied:
P(y=+1|x, &theta;)  = 1/(1+e<sup>-&theta;<sup>T</sup>x</sup>)

We have an idea of how to create classification models, and we know that we can effectively apply certain functions to determine reasonable probabilities.

> ### How can we determine the relationship between features?

Because this is supervised learning, we have access to a dataset! One way of coming up with the regression coefficients is to split up the datasets. We can split it up into a finite number of classes by a unique discrete outcome.




In linear classification, we are looking for a  relationship between features and observations, but we have discrete outcomes. To work around this, we can break up the data, sorted by outcomes, and then apply methods similar to linear regression model fitting. 

Then when fitting the model, we find &theta;'s such that they will represent the (probabilistic) relationship between features and outcomes.


> An example:

A dataset has two features, **top speed**, and **engine volume**. The set of possible observations are {Syundai, Shevrolet, Shrysler}. 

We can split up the dataset into three parts. One part is where all observations are Syundai, another is Shevrolet, and another is Shrysler. Now, we try to find the coefficients that relate **top speed** and **engine volume** to the possible outcomes. 

We want to pick the regression coefficients that most accurately describe the relationship, and as we saw in part 1, we can apply loss functions to determine the best fits.

> ### Likelihood

The likelihood function measures the quality of the fit for a model with parameters &theta;... 

The maximum likliehood seeks to find parameters &theta;...  that model the relationship the most accurately.

If you are trying to predict a +1 outcome on testing features, your model if very good if:

- P(y=+1|x, &theta;) ~ 1
- P(y=-1|x, &theta;) ~ 0

If you are trying to predict a -1 outcome on testing features, your model if very good if:

- P(y=+1|x, &theta;) ~ 0
- P(y=-1|x, &theta;) ~ 1

Before we go into maximum likliehood, think about this: Why must we use maximum likliehood? Why not use OLS? 

Because the classification error is not like a regression error. In regression, an error is determined from a (mathamatical) comparison between predicted and observed values. 

How can we mathamatically observe the relationship between two discrete values? There is no way to model this difference by only using the predictions/observations. 

We apply the log likliehood to the sigmoid function, or whichever you use for standardization. You can set the derivative of the sum of these probabilities to zero to get the best parameters.

With multiple parameters, same idea but with gradients

> ##  Criteria


We can set thresholds based on the standardization functions or probability requirements, suchb as ``0.5`` in the sigmoid funciton.
The threshold becomes the decision boundry.

We can rank the classifications too, usually by organizing the observations.


> Using probabilities

We want to hevaly penalize misclassifications with high confidence. To do this, we can use ``cross entropy``


> ##  Label based criteria

We used label based criteria to obtain easy to interpret information about classifications.

An example is accuracy. It is:

    # Correct classifications / Total # of classifications

Not very useful if we dont know much about the distribution


We can use confusion matricies to comprehensively represent the classification errors

There is precision, recall, F-score, sensitivity, specificity, balanced accuracy, among many other measures.

> ## Rank based classifiers

If we move the threshold, we change the outsome from ranking classifiers. 

The receiver operating charachteristic tries all possible thresholds.

An ROC curve plots the TPR-FPR distribution over the course of the algoritham.

A very good classifier has an area under the ROC curve of 1. This is also known as the AUROC. A AUROC of 0.5 is bad.

Precision recall curve is the same idea.

> ## Multiclass classification

We have K classes. We apply the same practices to each permutation. Apply log likleyhood this same way

We can use softmax in this instance. Softmax converts a vector of real numbers to a vector of probabilities. It is a generalization of the sigmoid.

> # Training versus testing

This subject was briefly elaborates in the introduction.

For supervised learning, we can split up the dataset we have into different parts. We use these different parts for training and testing. There are different ways to do this. We can split the dataset in half, or we can split it into k differnt parts and apply k different train-test model fitting iterations. 


