# Feature construction

Specifically for supervised learning.

What is useful in our dataset? What can we omit, what should be changed? Is there some standardizing process, modification, or correlative relationship we need to exploit. 

We can modify features beyond linearity, and we can apply this to select features or all features. This introduces multifeature correlation.

We can any type of functions to our features as long as there is a real mapping.

> Why feature selection?

We may have too many features, or suspect that there are correlative properties between features.

We may want to convert one feature into a different format.

> ## Processes

Subset selection: This is dropping some of the features.

Shrinkage / Regularization: This involves reducing the value of predictors. Has the effect of reducing variance.

Dimension reduction: I.e feature combonation

### Subset seleciton

We can iterate over every possibility, or we can apply forward/backwards stepwise selection.

### Regularization

Keep all features, shrink relative to other features

There are two types of regularization

Ridge regression
Lasso regression

Ridge regression: RSS+squared betas

Lasso: RSS+abs betas

Both are for parameter reduction, Lasso can reduce coefs to zero.

Elastic net is a combonation of both.

Cross validation helps us pick the tuning parameter. 