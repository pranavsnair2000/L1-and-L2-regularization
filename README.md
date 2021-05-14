# L1-and-L2-regularization
A small experiment to assess the importance of regularization in a machine learning model, and understanding the differnece between L1 and L2 regularization.


## Problem statement
A fabrication plant produces microchips. They go through two test to ensure to ensure correct functionality. The dataset contains the test scores (integers) of these two sets of past microchips along with whether or not they’ve been accepted. The objective here is to build a regression model to determine whether a new microchip should be accepted or rejected, given its test scores. And also to assess how well both types of regularization improves test set accuracy.

## Files
Provided below are short descriptions of each function, along with explanation of the dataset.
### Data.txt
This is the text file containing 3 columns separated by commas and 118 rows.
Each row represents a past microchip.
The first two columns have the scores on test 1 and 2 for these microchips and the third column takes values either 1 or 0 which stand for accepted or rejected.

### Project_fin.m
This is the main file, we first load the data, plot it to visualize and split it into train and test data.
Then *mapFeature.m* is used to map features to the 6th degree.

The rest of the code can be divided into 3 parts, based on what kind of regularization is used for the model:    
  1. Without Regularization
  2. With L1 Regularization
  3. With L2 Regularization

In each part, fminunc function is used to optimize the fitting of theta. To this function, we pass the cost function (separate for each case), initial theta and “options”, which contain preferences like the maximum number of iterations to try.

**Note**: fminunc is used instead of regular gradient descent as it is a much more optimal way to solve our problem, as octave has many inbuilt libraries that are extremely optimized for this kind of problem.
### Cost functions
There are separate cost functions, *costFunctionRegL1.m* and *costFunctionRegL2.m* for each type of regularization.
And since without regularization essentially means regularization with lambda set as 0, *CostFunctionRegL1.m* is used with lambda set to 0.
### Predict.m
This function uses the theta that’s been fit to predict the result values for given feature mapped test score matrix.
