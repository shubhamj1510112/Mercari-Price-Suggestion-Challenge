# set working directory
setwd("C:/Users/user/Desktop/pr_ed/final_upload/Preprocess_AlgorithmSelectionTuning_Training/supporting_files_patt")
getwd()
rm(list=ls())

# read input file from previous variable encoding stage 
df = read.table(file="Final_for_modeling.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")

#####################################################################################
#####################################################################################
################### Data preprocessing ##############################################

#####################################################################################
################### divide into train and test ######################################

# divide data into train and test dataset
# using random sampling without replacemnt method
train = df[sample(nrow(df),  886270, replace=F),]
# here this test dataset can be considered as blind or real dataset
# as this data will never be seen by the algorithm at the time of training
# not even at the 5-fold cross validation stage
test  = df[!(1:nrow(df)) %in% as.numeric(row.names(train)), ]
#write.table(train, file="train.csv", row.names=F, col.names=T, sep="\t", quote=F)
#write.table(test, file="test.csv", row.names=F, col.names=T, sep="\t", quote=F)
#train = read.table(file="train.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
#test = read.table(file="test.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")

#####################################################################################
################### remove near zero/ useless predictors ############################

# check variable types for the loaded train dataset
table(sapply(1:ncol(train), function(x) class(train[,x])))
# load library for parallel processing
require(foreach)
# identify useless or not varying variables
nzv <- nearZeroVar(train, freqCut=95/5, uniqueCut=10, allowParallel=T)
# check nzv and modify so that at least one representative of all of the initial variables 
# (name, item_codition, category, brand name, shipping, and descrition) we can keep 
remove = c(2,3,4,52)
nzv = nzv[! nzv %in% remove]
# remove the useless variables from the training dataset
train_nzv <- train[, -nzv]
# cleaning
rm(nzv, remove)

#####################################################################################
################### remove multicollinearity ########################################

# take subset of data for correlation study
train_corr = subset(train_nzv, select = -c(50))
train_target = train_nzv[,50]
# remove highly correlated variables
descrCor <- cor(train_corr)
# replace missing values where correlation could not be calculated due to zero variance
descrCor[is.na(descrCor)] = 0
highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
# we did not get any two predictor highly correlated (>0.85 cut-off for correlation coefficient) 
# so we did not remove any predictor variable at this stage
# remove linear dependencies as this is also a special type of multicollinearity
comboInfo <- findLinearCombos(train_corr)
# list of variables where one is linear combination of other
comboInfo$remove
# we see that no columns are linear combination of each other  
# so we did not remove any predictor variable at this stage also
# as we did not remove any predicor variable at this stage our output file is 
# same as input file after this stage of removing multicollinearity
train_nzv_mul = train_nzv
#write.table(train_nzv_mul, file="train_nzv_mul.csv", row.names=F, col.names=T, sep="\t", quote=F)
# cleaning
rm(descrCor, highlyCorDescr, comboInfo, train_corr, train_target)

#######################################################################################
################### Visualization: correlogram ########################################

#check the correlation using correlogram
require(corrplot)
corr_mat = cor(train_nzv)
head(round(corr_mat, 2))
# it got some NA because of near zero standard deviation of few variables
# so we will impite the NA with zero correlation as the standard deviation near zero
# indicates that these is no evidence of correlation and null hypothesis for # significant correlation can not be rejected, hence correlation stands zero
corr_mat[is.na(corr_mat)] = 0
jpeg(file="corrlogram_before_multicol.jpeg", height = 6000, width = 5500, res=300)
# par(mar=c(40, 4.1,40,2.1)+0.1)
corrplot(corr_mat, method='color', tl.cex=1.5, cl.cex=2) 
dev.off()
# although correlogram do show some positive and negative correlations
# but none of them is of strength greater than 0.85  
# as we also observed in the multicollinearity removal section

#####################################################################################
####################### Outlier Treatment ##########################################

# function to detect outliers
outdet = function(data) {
  outvec = numeric()
  for (i in 1:ncol(data)) {
    var = as.numeric(data[,i])
    cutoff_up = quantile(var, 0.75) + 3*(IQR(var))
    cutoff_low = quantile(var, 0.25) - 3*(IQR(var))
    l_up = length(var[var>cutoff_up])
    l_low = length(var[var<cutoff_low])
    if(l_up > 0 || l_low > 0) {
      colna = colnames(data)
      colname = colna[i]
      print(paste0("Iteration:", i, "There are outliers in: ", colname))
      outvec = c(outvec, i)
    }
    else {
      print (paste0("Iteration:", i, "There are no outliers"))
    }
  }
  return(outvec)
}
outvec1 = outdet(train_nzv_mul)    
# our 43 predictor variables had outliers so we need to treat them
# we will not perform outlier treatment for brand_name variable as this is the original variable which is label encoded
# also we will not perform outlier treatment for price variable as this is the target variable
remove = c(16, 50)
outvec1 = outvec1[! outvec1 %in% remove]
# treatment for outlier
# for that we create a new variables for the variable which has outlier
# this new variable will have outlier values replaced with boundary values example upper fence outlier with upper fence boundary values
# separate the predictor and target variable
train_nzv_mul_target = train_nzv_mul[,50]
train_nzv_mul_predictor = train_nzv_mul[,1:49]
# function for outlier treatment
outtre = function(data, i) {
    colname_out = numeric()
    mx = quantile(data[,i], 0.75) + 3*(IQR(data[,i]))
    mn = quantile(data[,i], 0.25) - 3*(IQR(data[,i]))
    for (j in 1:length(data[,i])) {
      if ( data[,i][j] < mn) {
        colname_out = c(colname_out, mn)
      } else if ( data[,i][j] > mx ) {
        colname_out = c(colname_out, mx)
      } else {
        colname_out = c(colname_out, data[,i][j])
      }
    }
    data[,i] = colname_out
    return(data)		
}
# run function over the list of variables with toutliers
# be patient this step may take some time
for (i in c(1:length(outvec1))) {
train_nzv_mul_predictor = outtre(train_nzv_mul_predictor, outvec1[i])
print (paste0("iteration:", i))
}
# combine the predictor and target variables
train_nzv_mul_out = cbind(train_nzv_mul_predictor, train_nzv_mul_target)
# write.table(train_nzv_mul_out, file="train_nzv_mul_out.csv", row.names=F, col.names=T, sep="\t", quote=F)

#####################################################################################
####################### variable importance ##########################################

############ perform this step on cluster #########
#df = read.table(file="train_nzv_mul_out.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
#df_work = df[,2:50]
#variable importance calculation with random forest
#df_work = train_nzv_mul_out[,2:50]
#library(parallel)
#library(doParallel)
#require(caret)
#cluster=makeCluster(10)
#registerDoParallel(cluster)
#fitControl = trainControl(allowParallel=TRUE, method="none")
#rf = train(Price ~ ., method="rf", ntree=100, data=df_work, importance=T, trControl = fitControl)
#stopCluster(cluster)
#registerDpSEQ()
#save(rf, file="rf_100.RData")
###################################################
# load the rf model created on cluster
forest = get(load("rf_100.RData"))
# create table of variable importance
require(caret)
var_imp = data.frame(varImp(forest)$importance)
var_imp$variable = row.names(var_imp)
var_imp = data.frame(var_imp[order(-var_imp$Overall),])
var_imp = var_imp[,c(2,1)]
#write.table(var_imp, file="var_imp.csv", row.names=F, col.names=T, sep="\t", quote=F)
# create figure of variable importance
jpeg(file="var_imp.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))
ggplot(var_imp, aes_string(x = reorder(as.factor(var_imp[,1]), -var_imp[,2]), y=var_imp[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()
# select important variable for next steps 
df = train_nzv_mul_out
df_predictor = df[,1:49]
Price = df[,50]
train_id = df[,1]
# other than important variables we will keep some other variables for analysis 
# which we think could be important for correctly predicting the price variable
sel_var = as.character(var_imp$variable[c(1:18, 20, 29, 39, 47)])
df_varimp = df_predictor[,sel_var]
df_tosave = cbind(train_id, df_varimp, Price)
dim(df_tosave)
names(df_tosave)
write.table(df_tosave, file="train_nzv_mul_out_varimp.csv", row.names=F, col.names=T, sep="\t", quote=F)


####################################################################################
####################################################################################
################### regression modeling ############################################

####################################################################################
############################# different algorithms comparison ######################

####################### data preparation ############################################
# as this a algorithm comparison step and this can be done with a sample data
# moreover running this comparison on complete data is computationally very expensive
# so we take randomly sampled 5000 observations and use this for making this comparison 
# read complete data
df = read.table(file="train_nzv_mul_out_varimp.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# randomly shuffle the data so that any bias in data spliting is gone
df<-df[sample(nrow(df)),]
# create sample data
samp = df[sample(nrow(df),  5000, replace=F),]
# this comparison will be done with 5-fold cross validation 
# manually create data partitions for 5-fold cross validation
# this is done so that our customized performance metric (RMSLE)can be evaluated at each fold
# therefore can have clear idea about the bias and variance of different algorithms
#Create 5 equally size folds
folds <- cut(seq(1,nrow(samp)),breaks=5,labels=FALSE)
# create 5 different train test dataset pairs
for(i in 1:5){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test <- samp[testIndexes, ]
    train <- samp[-testIndexes, ]
    write.table(train, file=paste0("samp_train_cv_", i, ".csv"), row.names=F, col.names=T, sep="\t", quote=F)
    write.table(test, file=paste0("samp_test_cv_", i, ".csv"), row.names=F, col.names=T, sep="\t", quote=F)
}

############################# define error metric function ##########################
# define function to calculate the error metric RMSLE
rmsle = function(pred, test) {
	sub = as.numeric()
	for (i in c(1:length(test))) {
		p = pred[i] + 1
		if (p < 0) { 
		p1 = 0
		}
		else {
		p1 = log(p)
		}
		a = test[i] + 1
		a1 = log(a)
		s = (p1 - a1)^2
		sub = c(sub, s)	
	}
	sum1 = sum(sub)
	div = sum1 / length(test)
	fin = sqrt(div)
	return(fin)
}

# compare different algorithms which found suitable for this particular problem of regression

####################### random forest ############################################
rmsle_vec_rf = as.numeric()
for (i in c(1:5)) { 
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none")
	set.seed(1234)
	rf = train(Price ~ ., method="rf", data=train_ip, ntree=100, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(rf, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_rf = c(rmsle_vec_rf, ram)
}
rmsle_vec_rf

####################### eXtreme Gradient Boosting ################
rmsle_vec_xgbTree = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(xgboost)
	library(readr)
	require(caret)
	require(stringr)
	require(car)
	set.seed(1234)
	xgbTree = xgboost(data=as.matrix(train_ip[1:22]), label=train_ip[,23], nrounds=100, nthread=5)
	pred = predict(xgbTree, as.matrix(test_ip))
	ram = rmsle(pred, test_check)
	rmsle_vec_xgbTree = c(rmsle_vec_xgbTree, ram)
}
rmsle_vec_xgbTree

####################### Relaxed Lasso ################################
rmsle_vec_relaxo = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	relaxo = train(Price ~ ., method="relaxo", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(relaxo, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_relaxo = c(rmsle_vec_relaxo, ram)
}
rmsle_vec_relaxo

####################### Support Vector Machines with Polynomial Kernel #############
rmsle_vec_svmPoly = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	svmPoly = train(Price ~ ., method="svmPoly", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(svmPoly, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_svmPoly = c(rmsle_vec_svmPoly, ram)
}
rmsle_vec_svmPoly

####################### CART #########################
rmsle_vec_rpart = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	rpart = train(Price ~ ., method="rpart", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(rpart, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_rpart = c(rmsle_vec_rpart, ram)
}
rmsle_vec_rpart


####################### k-Nearest Neighbors ################
rmsle_vec_kknn = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	kknn = train(Price ~ ., method="kknn", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(kknn, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_kknn = c(rmsle_vec_kknn, ram)
}
rmsle_vec_kknn



####################### Bayesian Additive Regression Trees ##########
rmsle_vec_bartMachine = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	bartMachine = train(Price ~ ., method="bartMachine", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(bartMachine, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_bartMachine = c(rmsle_vec_bartMachine, ram)
}
rmsle_vec_bartMachine


####################### Boosted Generalized Linear Model #################
rmsle_vec_glmboost = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	glmboost = train(Price ~ ., method="glmboost", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(glmboost, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_glmboost = c(rmsle_vec_glmboost, ram)
}
rmsle_vec_glmboost



####################### Neural Network ##########################
rmsle_vec_monmlp = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	monmlp = train(Price ~ ., method="monmlp", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(monmlp, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_monmlp = c(rmsle_vec_monmlp, ram)
}
rmsle_vec_monmlp

####################### Stochastic Gradient Boosting machines ############
rmsle_vec_gbm = as.numeric()
for (i in c(1:5)) {
	# this cross validation to check bias and variance of model w.r.t. rmsle error metric
	train = read.table(file=paste0("samp_train_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	train_ip = train[,2:24]
	test = read.table(file=paste0("samp_test_cv_", i, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
	test_ip = test[,2:23]
	test_check = test[,24]
	library(parallel)
	library(doParallel)
	require(caret)
	cluster=makeCluster(5)
	registerDoParallel(cluster)
	fitControl = trainControl(allowParallel=T, method="none") # this cross validation is for best parameter identification and is based on RMSE
	set.seed(1234)
	gbm = train(Price ~ ., method="gbm", data=train_ip, trControl = fitControl)
	stopCluster(cluster)
	pred = predict(gbm, newdata = test_ip)
	ram = rmsle(pred, test_check)
	rmsle_vec_gbm = c(rmsle_vec_gbm, ram)
}
rmsle_vec_gbm

algo_comp = data.frame(cbind(rmsle_vec_rf, rmsle_vec_xgbTree, rmsle_vec_relaxo, rmsle_vec_svmPoly, rmsle_vec_rpart, rmsle_vec_kknn, rmsle_vec_bartMachine, rmsle_vec_glmboost, rmsle_vec_monmlp, rmsle_vec_gbm))
row.names(algo_comp) = c("cv1", "cv2", "cv3", "cv4", "cv5")
write.table(algo_comp, file="algo_comp.csv", row.names=T, col.names=T, sep="\t", quote=F)
colnames(algo_comp) = c("Random_forest","xgBoost","Relaxed_Lasso","SVM_Polynomial","CART","KNN","bartMachine","glmboost","Neural_Network","gbm")
algo_comp_stack = data.frame(stack(algo_comp[,1:10]))
colnames(algo_comp_stack) = c("value", "algorithm")
#write.table(algo_comp_stack, file="algo_comp_stack.csv", row.names=T, col.names=T, sep="\t", quote=F)

# create figure of algorithm selection
jpeg(file="algo_comp.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))
ggplot(data=algo_comp_stack, aes_string(x = algo_comp_stack[,2], y=algo_comp_stack[,1])) +
  geom_boxplot(color="red", outlier.size=5) +
  theme_bw() + 
  xlab("Algorithms") + ylab("Root Mean Square Log Error") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()

# so the model with low bias and low variance was svmPoly but it could not be trained on complete data due to computational limitations
# so the next best model with low bias and low variance xgBoost was selected for further optimizations

##############################################################################################
############################# algorithm tuning ###############################################
# this step os done to identify the most optimal parameters for this algorithm
# so that we can achieve maximum performce with low bias and low variance
rmsle_vec_xgboost_cv5 = as.numeric()
for (i in c(25, 50, 75, 100)) {
	for (j in c(3,5,15)) {
		for (k in c(1:5)) {
			# this cross validation to check bias and variance of model w.r.t. rmsle error metric
			train = read.table(file=paste0("samp_train_cv_", k, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
			train_ip = train[,2:24]
			test = read.table(file=paste0("samp_test_cv_", k, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
			test_ip = test[,2:23]
			test_check = test[,24]
			library(xgboost)
			library(readr)
			require(caret)
			require(stringr)
			require(car)
			set.seed(1234)
			xgbTree = xgboost(data=as.matrix(train_ip[1:22]), label=train_ip[,23], nrounds = i, max_depth = j, nthread=5)
			pred = predict(xgbTree, as.matrix(test_ip))
			ram = rmsle(pred, test_check)
			rmsle_vec_xgboost_cv5 = c(rmsle_vec_xgboost_cv5, ram)
		}
	}
}
rmsle_vec_xgboost_cv5
name_vec = as.character(c(rep("3_25", 5), rep("5_25", 5), rep("15_25", 5), rep("3_50", 5), rep("5_50", 5), rep("15_50", 5), rep("3_75", 5), rep("5_75", 5), rep("15_75", 5), rep("3_100", 5), rep("5_100", 5), rep("15_100", 5)))
tune_comp = data.frame(cbind(rmsle_vec_xgboost_cv5, name_vec))
colnames(tune_comp) = c("value", "Parameter")
#write.table(tune_comp, file="tune_comp.csv", row.names=T, col.names=T, sep="\t", quote=F)
# create figure of algorithm selection
jpeg(file="tune_comp.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))
ggplot(data=tune_comp, aes_string(x = tune_comp[,2], y=as.numeric(as.character(tune_comp[,1])))) +
  geom_boxplot(color="blue", outlier.size=5) +
  theme_bw() + 
  xlab("Parameters") + ylab("Root Mean Square Log Error") +
  scale_y_continuous(limits = c(0.55, 0.85)) +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()
# from the comparison it is apparent that max_depth value of 5 and nrounds of 25 gives the best results in terms of RMSLE values and speed
# therefore max_depth of 5 and nrounds of 25 were selected for further analysis
 

######################################################################################################
############################# Final model construction ###############################################

# load complete train data
df = read.table(file="train_nzv_mul_out_varimp.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
train = df[,2:24]
# training of xgBoost algorithm on optimized parameters
library(xgboost)
library(readr)
require(caret)
require(stringr)
require(car)
set.seed(1234)
xgbTree = xgboost(data=as.matrix(train[1:22]), label=train[,23], nrounds = 25, max_depth = 5, nthread=20)
save(xgbTree, file="Final_model.RData")
xgbTree = get(load("Final_model.RData"))

####################################################################################
############################  Model Evaluation #####################################

# prediction on the blind dataset with a subset of 10000 observations
# for this blind data we have price value so we can get an idea about our rmsle error metric and performance of our model

############### preprocess blind dataset #############################
test = read.table(file="test.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
col_nzv_mul = read.table(file="colnames_train_nzv_mul.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
test_nzv_mul <- test[names(test) %in% col_nzv_mul$V1]
names(test_nzv_mul)
test_nzv_mul = test_nzv_mul[1:10000,]
# separate the predictor and target variable
test_nzv_mul_target = test_nzv_mul[,50]
test_nzv_mul_predictor = test_nzv_mul[,1:49]
# reestimate the outlier values the way we have done for training dataset
outtre = function(data, i) {
    colname_out = numeric()
    mx = quantile(data[,i], 0.75) + 3*(IQR(data[,i]))
    mn = quantile(data[,i], 0.25) - 3*(IQR(data[,i]))
    for (j in 1:length(data[,i])) {
      if ( data[,i][j] < mn) {
        colname_out = c(colname_out, mn)
      } else if ( data[,i][j] > mx ) {
        colname_out = c(colname_out, mx)
      } else {
        colname_out = c(colname_out, data[,i][j])
      }
    }
    data[,i] = colname_out
    return(data)		
}
# run the function
outvec1 = read.table(file="outvec1.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
for (i in c(1:length(outvec1$V1))) {
test_nzv_mul_predictor = outtre(test_nzv_mul_predictor, outvec1$V1[i])
print (paste0("iteration:", i))
}
# combine the predictor and target variables
test_nzv_mul_out = cbind(test_nzv_mul_predictor, test_nzv_mul_target)
# load variable importance file
var_imp = read.table(file="var_imp.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# select important variable for next steps 
test_predictor = test_nzv_mul_out[,1:49]
Price = test_nzv_mul_out[,50]
train_id = test_nzv_mul_out[,1]
# other than important variables we will keep some other variables for analysis which we think could be important for 
# correctly predicting the price variable
sel_var = as.character(var_imp$variable[c(1:18, 20, 29, 39, 47)])
test_varimp = test_predictor[,sel_var]
test_final = cbind(train_id, test_varimp, Price)
dim(test_final)
names(test_final)

################## predict on blind dataset ###################
xgbTree = get(load("Final_model.RData"))
pred = predict(xgbTree, as.matrix(test_final[2:23]))

########### performance check of our model on blind dataset  #####
# define function to calculate the error metric RMSLE
rmsle = function(pred, test) {
	sub = as.numeric()
	for (i in c(1:length(test))) {
		p = pred[i] + 1
		if (p < 0) { 
		p1 = 0
		}
		else {
		p1 = log(p)
		}
		a = test[i] + 1
		a1 = log(a)
		s = (p1 - a1)^2
		sub = c(sub, s)	
	}
	sum1 = sum(sub)
	div = sum1 / length(test)
	fin = sqrt(div)
	return(fin)
}
ram = rmsle(pred, test_final[,24])
ram

############ Wow! final RMSLE value we obtained is =  0.6236  ###########################
##########################################################################################################
This I achieved by using just 60% of our given training data (of stage1). And also my model can be trained in less than one minute
and it can predict the price for more than 100,000 observations within 2 minutes so its deployment (online or offline) is very easy.
Now if I use the complete dataset of stage one and stage two I need to run the whole calculation on cluster, however 
the initial results of this calculation on cluster gave the RMSLE value of 0.401. This shows that my strategy was really good and fast 
and if I would have participated in the challenge I would have got Silver medel.
