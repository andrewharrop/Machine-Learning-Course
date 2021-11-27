# Quantifying uncertainty

## Parameter uncertainty, estimation, and sampling distribution

Bias is the expected difference between an estimated parameter and the generative parameter.

Variance is the expected squared difference between the estimators and the mean.

## Confidence intervals via central limit theorem

For many different distributions of RV X, the sampling distribution X<sub>n</sub> is approximately normal if n is big enough

Central limit theorem consequence

95% of a normal dist is contained within:

&mu; +- 1.96*&sigma;<sub>n</sub>/sqrt(n)

## Uncertainty of test error

Test error is the sum of the loss function applied to test data divided by the number of datapoints apploed.

## Bootstrap

Draw data from an initial sample according to some distribution that represents the relationship between the data. 

This creates many bootstrapped samples. 