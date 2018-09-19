# Mercari Price Suggestion Challenge

1.	Problem statement

A Regression problem: Construct a trained model with optimized algorithm, which can automatically predict the most accurate price of a given product based on the specific features/information provided about this product.  

2.	Background

Several online trading companies or online shopping websites which sell a wide range of items need to suggest the prices automatically to their items. This process need to be automated because the items are just so many to be done by human beings and also getting it done by human being is very slow and expensive. So Artificial Intelligence and Machine Learning has a big application here. The Biggest Challenge here is that the price of items is dependent on the type of item and there are tens of thousands of different types of item. Added to this complexity prices of similar items can vary a lot based on their brand, condition of item, specifications of the item and so on. 
The most useful variables for predicting the price (item name, brand name, category, and description) all of them are unstructured text data and the target variable is a continuous numerical variable. Thus, due to this complication this problem becomes even more challenging. So, to deal with this here I have used the text analytics tools to derive the useful term frequency related new variables from this unstructured text data which can help us predict the prices more accurately and also the training and prediction steps can be done in a time and computation efficient manner. The text data where the complete text is important (e.g. brand name) they were label encoded so that the processing and computation is faster with these new label encoded variables.  After basic pre-processing steps many different algorithms, ranging from basic statistical method such as generalized linear model to advanced neural networks, which were suitable for regressions were employed and tested. The algorithm selection was done using n-fold cross-validation method to obtain the minimum bias and minimum variance of the trained algorithm. Then tuning of different parameters for the selected model was performed to select the most optimum parameters. Finally the most optimum model with best performance was build and tested on the given test dataset.

3.	Evaluation metric or Error metric 

Since it is a regression problem, we will have a squared error related metrics as the error metric to minimize for making accurate predictions. As our product price has a very broad range from tens to thousands, we have to use a transformed version of squared error metric so that we do not get biased for accuracy on the products with high price. Thus, I have used the “Root Mean Squared Logarithmic Error” (RMSLE) as the error metric to minimize (also suggested at the kaggle competition page) for this particular project. The mathematical formula for this error metric is is same as for the RMSE except for the fact that logrithm of target varible is used instead of actual values.





