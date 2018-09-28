######################################################################################################
####### Make Predictions with the actual kaggle test data ###########################################

###### for the sake of fast running of r code here predictions are made for 10,000 observations ######
#### but the same script can be used to making predictions on any number of test observations #######

setwd("C:/Users/user/Desktop/pr_ed/final_upload/prediction_kaggle_test_data/supporting_files_testkag")
getwd()
rm(list=ls())

# load the test data from kaggle (test.tsv)
co = c('integer', 'character', 'integer', 'character', 'character', 'integer', 'character')
t = read.table(file="test.tsv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "", colClasses = co)
t = t[1:10000,]
t_reg = t[,1]

# we will apply the same prerocessing as we applied on our training set
# so that our algorithm can predict on this new data
# this code is very customized and any amount of raw data can be precessed for making predictions with our model

######################################################################
################ first feature: name preprocessing ###################
# load libraries ; try with R version 3.4.4 or above
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
t$name= gsub("\\[rm\\]", "", t$name) # to remove [rm] values from data
t$name= gsub("[^a-zA-Z0-9 ]", "", t$name)
# create corpus so that preprocessing methods can be applied
postCorpus = Corpus(VectorSource(t$name))
#case folding
postCorpus = tm_map(postCorpus, tolower)
writeLines(as.character(postCorpus[[2]]))
#remove stop words
postCorpus = tm_map(postCorpus, removeWords, stopwords('english'))
writeLines(as.character(postCorpus[[1]]))
#remove punctuation marks
postCorpus = tm_map(postCorpus, removePunctuation)
writeLines(as.character(postCorpus[[1]]))
#remove unnecesary spaces
postCorpus = tm_map(postCorpus, stripWhitespace)
writeLines(as.character(postCorpus[[1]]))
# stemming
postCorpus  = tm_map(postCorpus, stemDocument, language="english")
writeLines(as.character(postCorpus[[5]]))
# convert to string vector again
col1 = data.frame(text=sapply(postCorpus, identity))
#lemmatization
require(textstem)
col1 = lemmatize_words(col1)
sel_terms = read.table(file="sel_terms_var1.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# use below code for creating dichotomous variaables for first 50 terms selected at the time of training
fea_var1 = data.frame(train_id = t[,1])
for (i in 1:50) {
  vec = as.integer()
  for (j in c(1:length(col1$text))) {
    c = as.integer(str_count(as.character(col1$text[j]), pattern=as.character(sel_terms$x[i])))
    vec = c(vec, c)
  } 
  fea_var1 = cbind(fea_var1, vec)
  print(paste0("iteration:", i))
} 
rm(c, vec)
dim(fea_var1)
fea_var1 = fea_var1[,2:51]
mod_var1_terms = paste("var1", sel_terms$x[1:50], sep="_")
colnames(fea_var1) = mod_var1_terms
t_reg = cbind(t_reg, fea_var1)

######################################################################################
################ second feature: item_condition_id preprocessing ######################

# meadian item condition id will be used to impute any missing values
sum(is.na(t[,3]))
median_1 = 2
t[,3][is.na(t[,3])] = median_1 
sum(is.na(t[,3]))
var2_item_condition = t[,3]
t_reg = cbind(t_reg, var2_item_condition)

############################################################################################################
################ third feature: category_name preprocessing ###################################################

# impute missing category name as unknown
t[,4][is.na(t[,4])] = "unknown/unknown/unknown"
sum(is.na(t[,4]))
#remove special character such as & except / 
# because we want to create three variables (cat1, cat2, cat3) from this one variable
t[,4] = gsub("\\[rm\\]", "", t[,4]) # to remove [rm] values from data
t[,4] = gsub("[^a-zA-Z0-9 /]", "", t[,4])
# split columns
require("reshape2")
many_var3 = as.data.frame(colsplit(t[,4], "/", c('cat1', 'cat2', 'cat3')))
length(unique(many_var3$cat1))
length(unique(many_var3$cat2))
length(unique(many_var3$cat3))

####### dummy encoding for cat1 #################
cat1_terms = read.table(file="cat1_terms.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
fea_cat1 = data.frame(train_id = t[,1])
for (i in c(1:length(cat1_terms$x))) {
  vec = as.integer()
  for (j in c(1:length(many_var3$cat1))) {
    c = as.integer(str_count(as.character(many_var3$cat1[j]), pattern=as.character(cat1_terms$x[i])))
    vec = c(vec, c)
  } 
  fea_cat1 = cbind(fea_cat1, vec)
  print(paste0("iteration:", i))
} 
rm(vec, c)
dim(fea_cat1)
fea_cat1 = fea_cat1[,2:12]
mod_cat1_terms = paste("cat1", cat1_terms$x, sep="_")
colnames(fea_cat1) = mod_cat1_terms
dim(fea_cat1)
# covert to PCs
require(dplyr)
require(tibble)
require(ggfortify)
forest = get(load("mypc.RData"))
cat1_pc_sel_test = tbl_df(predict(mypc, newdata = fea_cat1)) %>% select(PC1:PC8)
cat1_pc_colname = paste("cat1", colnames(cat1_pc_sel_test), sep="_")
colnames(cat1_pc_sel_test) = cat1_pc_colname
View(head(cat1_pc_sel_test, 10))
dim(cat1_pc_sel_test)

# append to regression file
t_reg = cbind(t_reg, cat1_pc_sel_test)
# cleaning
rm(cat1, cat1_coded, cat1_pc_colname, cat1_pc_sel, cat1_terms, co_name, mod_cat1_terms, mypc, pr_var, prop_varex, std_dev)

# create new data frame for cat1 and cat2 label encoding 
t1 = as.data.frame(cbind(as.numeric(t[,1]), as.character(many_var3$cat1), as.character(many_var3$cat2), as.character(many_var3$cat3), as.numeric(t[,6])))
t1$V1 = as.numeric(as.character(t1$V1))
t1$V5 = as.numeric(as.character(t1$V5))

####### label encoding for cat2 ########################
cat2_mean_df = read.table(file="cat2_mean_df.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
cat2_coded = as.numeric()
rm(a)
for (i in c(1:nrow(t1))) {
  a = match(t1[i,3], cat2_mean_df$cat2_uniq)
  cat2_coded = c(cat2_coded, a)
}
cat2_coded[is.na(cat2_coded) == TRUE] = 74 # which is the label encoding for unknown cat2 values (which were not there in training dataset)
cat2_label_encoded = cat2_coded
# append to regression file
t_reg = cbind(t_reg, cat2_label_encoded)
#cleaning
rm(a, cat2, cat2_coded, cat2_label_encoded, cat2_mean, cat2_mean_df, cat2_uniq, i, num,)

####### label encoding for cat3 ########################
cat3_mean_df = read.table(file="cat3_mean_df.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
cat3_coded = as.numeric()
rm(a)
for (i in c(1:nrow(t1))) {
  a = match(t1[i,4], cat3_mean_df$cat3_uniq)
  cat3_coded = c(cat3_coded, a)
}
cat3_coded[is.na(cat3_coded) == TRUE] = 572 # which is the label encoding for unknown cat3 values (which were not there in training dataset)
cat3_label_encoded = cat3_coded
# append to regression file
t_reg = cbind(t_reg, cat3_label_encoded)
#cleaning
rm(a, cat3, cat3_coded, cat3_label_encoded, cat3_mean, cat3_mean_df, cat3_uniq, i, many_var3, num, var3, t1)

##########################################################################################
################ Fourth feature: brand name preprocessing ################################

# label encoding
t[,5][is.na(t[,5]) == TRUE] = "unknown"
sum(is.na(t[,5]))
var4_mean_df = read.table(file="var4_mean_df.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
bn_coded = as.numeric()
rm(a)
for (i in c(1:nrow(t))) {
  a = match(t[i,5], var4_mean_df$var4_uniq)
  bn_coded = c(bn_coded, a)
}
bn_coded[is.na(bn_coded) == TRUE] = 2901 # which is the label encoding for unknown brand names (which were not there in training dataset)
bn_label_encoded = bn_coded
# append to regression file
t_reg = cbind(t_reg, bn_label_encoded)
#cleaning
rm(a, bn_coded, bn_label_encoded, i, num, var4, var4_uniq, var4_mean, var4_mean_df)

####################################################################################
################ Fifth feature: shipping preprocessing #############################
var5 = t[,6]
# use mode of this variable in the training dataset for imputation
mode_5 = 0
# impute missing values with mode
var5[is.na(var5) == TRUE] = mode_5
length(var5)
var5_shipping = var5
# append to regression file
t_reg = cbind(t_reg, var5_shipping)
#cleaning
rm(mode_5, modefun, var5, var5_shipping)

########################################################################################
################ Sixth feature: item description preprocessing ##########################
var6 = t[,7]
# remove all special characters
var6 = gsub("\\[rm\\]", "", var6) # to remove [rm] values from data
var6 = gsub("[^a-zA-Z0-9 ]", "", var6)
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
# create corpus so that preprocessing methods can be applied
postCorpus = Corpus(VectorSource(var6))
#case folding
postCorpus = tm_map(postCorpus, tolower)
writeLines(as.character(postCorpus[[2]]))
#remove stop words
postCorpus = tm_map(postCorpus, removeWords, stopwords('english'))
writeLines(as.character(postCorpus[[2]]))
#remove punctuation marks
postCorpus = tm_map(postCorpus, removePunctuation)
writeLines(as.character(postCorpus[[2]]))
#remove unnecesary spaces
postCorpus = tm_map(postCorpus, stripWhitespace)
writeLines(as.character(postCorpus[[2]]))
# stemming
postCorpus  = tm_map(postCorpus, stemDocument, language="english")
writeLines(as.character(postCorpus[[2]]))
# convert to string vector again
col1 = data.frame(text=sapply(postCorpus, identity))
#lemmatization
require(textstem)
col1 = lemmatize_words(col1)
s = c(1:10000)
ss = data.frame(cbind(s, as.character(col1$text)))
sel_terms_var6 = read.table(file="sel_terms_var6.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
fea_var6 = data.frame(train_id = t[,1])
for (i in 1:50) {
  vec = as.integer()
  for (j in c(1:length(ss[,2]))) {
    c = as.integer(str_count(as.character(ss[j,2]), pattern=as.character(sel_terms_var6$V1[i])))
    vec = c(vec, c)
  } 
  fea_var6 = cbind(fea_var6, vec)
  print(paste0("iteration:", i))
} 
dim(fea_var6)
fea_var6 = fea_var6[,2:51]
mod_var6_terms = paste("var6_disc", sel_terms_var6$V1, sep="_")
colnames(fea_var6) = mod_var6_terms[1:50]
dim(fea_var6)
# append to regression file
t_reg = cbind(t_reg, fea_var6)
test = t_reg
test_id = test[,1]
colname = read.table(file="colname_114.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
colnames(test) = colname$V1
#cleaning
rm(col1,i, mode_var6_terms, s, sel_terms, ss, v, v1, var6, var6_coded, var6_terms)


##################################################################################################
################ filtering: nzv, multicollinearity, outlier, variable importance ##################

col_nzv_mul = read.table(file="colnames_train_nzv_mul.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
test_nzv_mul <- test[names(test) %in% col_nzv_mul$V1]
names(test_nzv_mul)
# separate the predictor and target variable
test_nzv_mul_predictor = test_nzv_mul[,1:49]
# function to for outlier treatment
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

# extract important variable selected at the time of training
var_imp = read.table(file="var_imp.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
sel_var = as.character(var_imp$variable[c(1:18, 20, 29, 39, 47)])
test_nzv_mul_predictor_varimp = test_nzv_mul_predictor[,sel_var]
test_final = cbind(test_id, test_nzv_mul_predictor_varimp)
dim(test_final)
names(test_final)
write.table(test_final, file="test_data_for_prediction.csv", row.names=F, col.names=T, sep="\t", quote=F)

###############################################################################################################
################################ predictions using final model ############################################
# now the kaggle test data is preprocessed and ready to be made predictions using our final model

# load the model (this will only work for R version 3.3.2 so please check before loading)
xgbTree = get(load("Final_model.RData"))


pred = predict(xgbTree, as.matrix(test_final[2:23]))
test_results = cbind(test_id, pred)
colnames(test_results) = c("test_id", "Predicted_Price")

write.table(test_results, file="test_results.csv", row.names=F, col.names=T, sep="\t", quote=F)

# So this final test_results.csv file is in the format which can be uploaded at kaggle (with predictions on stage 2 data) 
# after that I can get my Leaderboard ranking but now since the competition is over
# I am attaching this final file with this submission for your kind reference
