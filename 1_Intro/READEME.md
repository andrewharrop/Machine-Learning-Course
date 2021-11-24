# **Introduction**

>## Terms

Regression is a statistical tool
- Features are the data on which predictions are made
- We may be analyzing many features
- Observations/responses are what we are trying to predict

Supervised learning
- We are provided a dataset
- We are trying to predict more "correct" responses

Classification: Discrete outcomes
Regression: Continuous outcomes

>## Linear Regression

Overview:

- Understand the impact variables excert on other variables
- Develop a simple functional relatonship between these variables
- Usually, we approximate this relationship with some simple function such as a polynomial
- We create a model to track the relationship
    * The input can be divided into training and testing data
    * We can apply feature engineering to increase usefulness of features
    * We use training observations and training features to train a machine learning model
    * We input the testing features to the machine learning model, and compare the model predictions with the testing observations

Training data

- Because we are using supervised learning, we must posses information we know is correct
- This is called training data, and we use it to train our model
- Example: Predict energy consumption with historical energy consumption patterns. The historical energy consumption patterns are the training data

Feature extraction
- This is the process of determining what data we need, and what data we don't need for our model
- Example: Predict the probability of a cruise ship sinking. The weather on the ships route is probably relavant, but the colours of passenger shirts probably is not important.
- The more features in a model, the more complex it will be, so it is best practice to review the dataset.

Quality metrics are used to compare the difference between the testing data observations and the predicted observations based on the features related to testing data observations

> ## Linear regression applied

The most simple linear regression model is on a first degree polynomial function with only one feature. The format of this function is:

y=mx+b

But in machine learning, we rewrite it:

f(x) = xθ<sub>1</sub>+θ<sub>0</sub>

θ<sub>1</sub>, θ<sub>0</sub> are regression coefficients

The error this type of model arises when there is a difference between a predicted value and an observed value. To account for that, we add it to the model like this:

f(x) = x<sup>(i)</sup>θ<sub>1</sub>+θ<sub>0</sub>+ε<sup>(i)</sup>

The <sup>(i)</sup> was added to show that we are applying the model to a single feature

> ## Assesing the fit of a line

To asses the fit of a line, we should define the cost of it. One method is the residual sum of squares (RSS) The function to compute it is:
    
RSS(θ<sub>0</sub>, θ<sub>1</sub>) = &Sigma;[y<sup>(j)</sup>- (θ<sub>0</sub>+θ<sub>1</sub>x<sup>(j)</sup>)]<sup>2</sup> <sub>j=1 -> m</sub>

###### *Note: a residual is the difference between predicted values and the observation*
###### Residual = y<sup>(j)</sup>- (θ<sub>0</sub>+θ<sub>1</sub>x<sup>(j)</sup>)

We want to minimize this. Because it is the squared difference between the predicted and actual value.

The best possible fit is Min(RSS). We can apply calculus to find this value, or employ a hill climbing algoritham.


# **Linear regression with multiple features**

> ## Linear regression predictons with multiple features

In the previous discussions, we have only considered linear regression with single features. We can also use multiple features in our models.

Mathamatically this is:

f(x) = x<sub>n</sub>θ<sub>n</sub>+...+x<sub>2</sub>θ<sub>2</sub>+x<sub>1</sub>θ<sub>1</sub>+θ<sub>0</sub>

The i<sup>th</sup> observation is:

y<sup>(i)</sup> = &Sigma; (&theta;<sub>j</sub>x<sub>j</sub><sup>(i)</sup>) + &theta;<sub>0</sub> + &epsilon;<sup>(i)</sup> from <sub>j=1 ->n</sub>

There is one more regression parameter than features, to account for an intercept. The set of n features at the i<sup>th</sup> observation still map to only one observations, so we need n+1 regression coefficients and m sets of features with n features each.

The matrix representation of this is:

y<sup>i</sup> = (x<sup>(i)</sup>)<sup>T</sup>&theta; + &epsilon;<sup>(i)</sup>

> ## RSS for linear regression with multiple features

The RSS for linear regression with multiple features is:

RSS(&theta;) = &Sigma;(y<sup>i</sup> - (x<sup>(i)</sup>)<sup>T</sup>&theta;)<sup>2</sup> from <sub>i=1->m</sub>

Again iterating over each prediction, and comparing it to the observation.

> ## Gradiants

We can apply gradients to minimize the RSS. This means we will take the partial derivative of each coefficient in an attempt to minimize them. 

The gradiants are usually computed by machine learning libraries under the hood. 

One method to apply gradients is the gradient descent approach. 

> ## Gradient descent

The cost function applied withing the gradient descent algoritham measuers how accurate a fit is.
Gradient descent applies a learning rate like hill climbing. You do not need to find the best value for each parameter, but you find a value that satisfies a threshold, leaving you with a parameter that is "close enough".

# Assesing performance

> ## Loss functions

To asses performance, we must apply a loss function to quantify the difference between predictions and observations. The loss function is effective when used over an entire set of predictions.

Some examples of loss functions:


- Absolute Error Loss Function (L<sub>1</sub> Norm):
    - L(y, f(x)) = |y-f(x)|
    - Minimizing this is the Least Absolute Deviation (LAD).
    - More robust to outliers




- Squared error loss function (Squared L<sub>2</sub> Norm):
    - L(y, f(x)) = (y-f(x))<sup>2</sup>
    - Minimizing this is the Ordinary Least Squares (OLS).
    - OLS is the most common loss function.
    
Training error is the average error of a loss function
The Root Mean Squared Error (RMSE) is the square root of the training error of the L<sub>2</sub> norm.

The coefficient of determination, R<sup>2</sup>, is 1-(RSS/TSS), where TSS is the RSS function using the mean of the observations as the prediction. The coefficient of determination will always fall between 0 and 1, with 1 being the best possible fit, and 0 being the worst. It is easy to interpret.

Some pseudocode for related functions
    
Predictions from a 1 x n matrix of coefficients and a m x n matrix 

    linearModelPredict(coefficients, features)
        pred = ceofficients * features.transpose()
        return predictions

Apply RSS to predictions. In practice, there should be a column of 1's in the features to account for the first regression coefficient (intercept)

    linearModelLossRSS(coefficients, features, observations)
        pred = linearModelPredict(coefficients, features)
        residual = observations - pred
        rss = sum(residual**2)
        gradient = -2*[sum(residual), 
                       sum(residual*features)]
        return rss, gradient

We can call an optimizer from an outside library. An example is the optimizer offered by scipy.optimize in python.

    import scipy.optimize as so
    start_values = [0,0] # Arbitrary coefficient estimates
    minimized_coefs = so.minimize(linearModelLossRSS,
                                 start_values,
                                 args=(X,y),
                                 jac=True)

The official scipy [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for more information.

**What did we just do?** We build a linear regression prediction function ``linearModelPredict``, then we created a loss function, specifically the L<sub>2</sub> norm with the ``linearModelLossRSS``. Then we minimized it and computed the best coefficients for the given loss function.

We can apply the same process for the L<sub>1</sub> norm:

    linearModelLossLAD(coefficients, features, observations)
        pred = linearModelPredict(coefficients, features)
        residual = observations - pred
        
        lad = sum(abs(residual))
        gradient = -sgn(res)*features 
        return lad, gradient

This works if we add a column of 1's in features to account for the intercept. We would use the same scipy function to minimize the gradients.

> ## Concluding notes

Linear regression is the applicaiton of estimating the relationship between features and responses to those features. 
There is no set way to determine the most accurate effective relationship. 
The application of loss functions assists in finding parameters that can effectively describe the relationship between features and observations.

When computing gradients, it is generally not feasable to reach ``gradient=0``, so gradient decent is an effective solution.