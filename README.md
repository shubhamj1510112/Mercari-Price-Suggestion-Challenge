## Mercari Price Suggestion Challenge

# 1.	Problem statement

A Regression problem: Construct a trained model with optimized algorithm, which can automatically predict the most accurate price of a given product based on the specific features/information provided about this product.  

2.	Background

Several online trading companies or online shopping websites which sell a wide range of items need to suggest the prices automatically to their items. This process need to be automated because the items are just so many to be done by human beings and also getting it done by human being is very slow and expensive. So Artificial Intelligence and Machine Learning has a big application here. The Biggest Challenge here is that the price of items is dependent on the type of item and there are tens of thousands of different types of item. Added to this complexity prices of similar items can vary a lot based on their brand, condition of item, specifications of the item and so on. 
The most useful variables for predicting the price (item name, brand name, category, and description) all of them are unstructured text data and the target variable is a continuous numerical variable. Thus, due to this complication this problem becomes even more challenging. So, to deal with this here I have used the text analytics tools to derive the useful term frequency related new variables from this unstructured text data which can help us predict the prices more accurately and also the training and prediction steps can be done in a time and computation efficient manner. The text data where the complete text is important (e.g. brand name) they were label encoded so that the processing and computation is faster with these new label encoded variables.  After basic pre-processing steps many different algorithms, ranging from basic statistical method such as generalized linear model to advanced neural networks, which were suitable for regressions were employed and tested. The algorithm selection was done using n-fold cross-validation method to obtain the minimum bias and minimum variance of the trained algorithm. Then tuning of different parameters for the selected model was performed to select the most optimum parameters. Finally the most optimum model with best performance was build and tested on the given test dataset.

3.	Evaluation metric or Error metric 

Since it is a regression problem, we will have a squared error related metrics as the error metric to minimize for making accurate predictions. As our product price has a very broad range from tens to thousands, we have to use a transformed version of squared error metric so that we do not get biased for accuracy on the products with high price. Thus, I have used the “Root Mean Squared Logarithmic Error” (RMSLE) as the error metric to minimize (also suggested at the kaggle competition page) for this particular project. The mathematical formula for this error metric is is same as for the RMSE except for the fact that logrithm of target varible is used instead of actual values.

Please Note that the complete r code, figures, and tables are uploaded as separate files and are referened at appropriate places in this report

4.	Variable encoding

Note: The complete r code for this section is provided as the variable_encoding.R file. So after setting up the path you can run the complete r code but it may take some time to run (please check for r version or package availability related errors as they vary from system to system anyways the used versions are specified in the comments of r script). 

In this particular dataset I had four columns with unstructured text data like name, category, brand name, and item description. Moreover, these columns also had a lot of special characters which needed to be taken care. Also almost all of the variables had the missing values. Thus, to resolve these issues this particular step of label encoding was performed. Here I used many tools and techniques from text analytics and dummy variable encoding. Now let’s see this step in detail for each of the variable.
 
Figure 1: Percent missing values plotted for each variable. Brand name has highest amount of missing values (>40%)

Table 1: Missing value summary for each variable. I observed that every predictor variable has missing values therefore; proper missing value treatment is required.

In data there was an identifier number for each training observation (train_id). As this has no predictive power so it was just kept there for reference but not used in modelling.

4.1 First variable: Item Name (predictor variable)

This variable had the name of the item written in the text format. This variable is very important as the name of the item is one of the main factors in deciding its price. For example a computer will have price Rs. 35,000 or more whereas a pen will have the price of Rs. 10 or more, so the name is deciding the price. So I had to use this variable for prediction and since it is a categorical variable I needed to perform the dummy encoding for it so that it can be used in regression modelling. This is because most regression algorithm has this requirement that they perform very well on numeric data (except a few like decision trees). 
The problem with this variable was that many item name appeared only one or very few times and in many cases the same item was given different names like pen, stationary pen, PEN and so on. Also the name had many special characters which may produce error at the time of training. So first I removed all the special characters using “gsub” command in r than I applied the text analytics tools to perform other preprocessing like covert to lower case, stemming, and lemmatization. Here I did not remove number because it has significance in predicting price like 16GB pen drive cost less than 32 GB pen drive. Then I used “term frequency” metric to extract the item names most frequent in the dataset and they were dummy encoded. Due to the computational limitations only top 50 item names were selected for now but same code can be used to select more or any desired number. This step was useful as now this will make our algorithm more focused on accurately predicting the prices of the items which are most frequent in the business/market.
As a result of this step now we have 50 variables in our dataset for now: 
50 (item name dummy encoded) = 50

4.2 Second variable: item condition (predictor variable)

This variable tells about the condition of the item available for purchase and rates it from 1-5. This variable is very useful in predicting the price as the item in good condition will have more price in comparison to item with poor condition. This variable was already very well encoded and was in numeric data format, although it was kind of a factor ordinal variable. The only problem with this variable was missing values: >4000 observations had missing values for this variable. I used the imputation method as I cannot throw out this high amount of observations. I compared different methods like k-nearest neighbours, random forest, mean, median, and mode on the sample data. I found that slower methods like knn and random forest were performing similar to faster methods like mean, median, and mode. So I compared only the faster methods like the mean, median, and mode for the complete data and found that median imputation was performing better. Therefore, I imputed all the missing values for this variable with the median value of “2” item condition rating. Now this variable is ready to be used for modelling.  
As a result of this step now we have 51 variables in our dataset for now: 
50 (item name dummy encoded) + 1 (item condition)= 51

4.3 Third variable: category name (predictor variable)

This variable defined the category for each item or observation. This had three different levels of categories and they were in the hierarchical order. Therefore, I could create three variables (one for each hierarchy) from this one variable. I named them as category-1 (cat1 in scripts), category-2 (cat2 in scripts), and category-3 (cat3 in scripts). This splitting of one column into three columns was performed using the “reshape2” library in r. Also the missing category values were imputed with the “unknown” value.
Now in for cat1 I performed the dummy variable encoding for this variable. After that as many categories had interactions which needed to be encoded for example in summers the winter clothes prices will be low for both men’s and women’s. Also I needed to control the number of predictor variables because if I use to many then training of our model will be very tedious and computationally expensive. Also many times having to many variables is not good for making accurate predictions. So to tackle all these problems I performed the Principal Component Analysis (PCA) on the cat1 dummy encoded variables. PCA will generate new variables from the input variables and good thing about PCA is that these new variables (called PCs in the script) are linearly independent and orthogonal and also encode the interaction between input variables. Usually we select the number of PCs using which we can explain the >80% of variance of our complete data, from the scree plot (pasted below) I found that eighth PCs were enough to explain >80% of variance, so I selected eight PCs for further analysis. 
 
Figure 2: Principal Components (PCs) were plotted against the cumulative variance of data explained. I observed that eight PCs are enough to explain the >80% variance present in the data

As a result of this step now we have 60 variables in our dataset for now: 
1 (train_id) + 50 (item name dummy encoded) + 1 (item condition) + 8 (PCs encoding cat1)= 60

Now in for cat2 I could not perform the dichotomous dummy variable encoding because it had too many categories, so for this variable I used the label encoding strategy.  In this label encoding strategy each category value was assigned a numeric rank. To assign this rank for each category value the mean price value was calculated for each category and then sorted numerically. So you can say that these category levels were numerically encoded based on their mean price value.
This same strategy was also applied to label encode the cat3 variable because this also had too many categories to be dummy coded.
As a result of this step now we have 61 variables in our dataset for now: 
50 (item name dummy encoded) + 1 (item condition) + 8 (PCs encoding cat1) + 1 (label encoded cat2) + 1 (label encoded cat3) = 61

4.4 Fourth variable: brand name (predictor variable)

This variable defines the brand name for the items. This is also a text variable or categorical/nominal variable. The major problem with this variable is that it has a lot of missing values (>600,000 observations or >40% of training dataset). This is also expected as many items are not branded and hence, will not have brand name. So, all the missing values were imputed to “unknown” brand name. This variable had a lot of levels due to a lot of different brand name for different items, because of this dummy variable encoding was not possible for this variable. Thus, the label encoding strategy (as also used for cat2 and cat3) was utilized to encode this particular variable. In this label encoding strategy the brands were ranked based on the mean price values of their items and now these ranks were used as a numeric vector or a numeric predictor variable for predicting the price. 
As a result of this step now we have 62 variables in our dataset for now: 
50 (item name dummy encoded) + 1 (item condition) + 8 (PCs encoding cat1) + 1 (label encoded cat2) + 1 (label encoded cat3) +1 (label encoded brand_name)  = 62

4.5 Fifth variable: shipping (predictor variable)

This variable was easy to handle as this has only two possible values, one shipping cost from seller, and the other, shipping cost from buyer. This variable was already kind of dummy coded so no extra efforts were required. However, it had a lot of missing values (>4000). All these missing values were imputed with the most common shipping mode which was shipping cost paid by buyer. 
As a result of this step now we have 63 variables in our dataset for now: 
50 (item name dummy encoded) + 1 (item condition) + 8 (PCs encoding cat1) + 1 (label encoded cat2) + 1 (label encoded cat3) +1 (label encoded brand_name) + 1 (shipping) = 63

4.6 Sixth variable: item description (predictor variable)

This variable was most difficult to handle, as this was the most unstructured form of text data. This also contained a lot special characters (including some emoji characters). So at first all the special characters were removed. Then text preprocessing tools were used to covert complete text into lower case and to remove punctuation marks and whitespaces. Then stemming and lemmatization was performed. Further, most important words which describe the quality of item, such as great condition, awesome, brand new, and so on, were extracted. Now term frequency of these selected words was used to derive the novel dummy coded variables for each of these words. In total, fifty most frequent words were selected so I had fifty new dummy coded variables extracted from this one item description variable.
As a result of this step now we have 113 variables in our dataset for now: 
50 (item name dummy encoded) + 1 (item condition) + 8 (PCs encoding cat1) + 1 (label encoded cat2) + 1 (label encoded cat3) +1 (label encoded brand_name) + 1 (shipping) + 50 (item description dummy encoded) = 113

4.7 Seventh variable: Price (target variable)

This is our target variable which we want that our final trained model should be able to predict with very high accuracy. As this is a continuous variable I have perform the regression modelling or this is a regression problem in machine learning terms. This variable had >4000 observations with missing values. As this is our target variable I could not impute it at the training stage, so I removed all the observations/items with missing price value. 
A
s a result of this step now we have 114 variables in our dataset for now: 
50 (item name dummy encoded) + 1 (item condition) + 8 (PCs encoding cat1) + 1 (label encoded cat2) + 1 (label encoded cat3) +1 (label encoded brand_name) + 1 (shipping) + 50 (item description dummy encoded) + 1 (Price) = 114

After this variable encoding stage in the final file we have 113 predictor variables (all numeric as either dummy encoded or label encoded) and 1 target variable.




