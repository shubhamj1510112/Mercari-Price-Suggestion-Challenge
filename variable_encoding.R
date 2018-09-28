setwd("C:/Users/user/Desktop/pr_ed/final_upload/Variable_Encoding/supporting_files_VE")
getwd()
rm(list=ls())

# to view few lines of training file  
cat(readLines("train.tsv", n=100), sep = "\n")
# As the raw tsv file from kaggle had a lot of special charcaters  
# which were interfering with the loading of complete file 
# some preprocessing and conversion to csv was performed
t = read.table(file="train.tsv", header=T, sep="\t", quote="", na.strings = "", fill = T, comment.char = "")
# remove special characters
t1 = as.data.frame(lapply(t, function(x) gsub("[^a-zA-Z0-9/ ]", "", x))) 
write.table(t1, file="train.csv", row.names=F, col.names=T, sep="\t", quote=F) # save as csv file
rm(t)
rm(t1)

# Now reload the preprocessed csv file 
# define column class beforehand
co = c('integer', 'character', 'integer', 'character', 'character', 'numeric', 'integer', 'character')
t = read.table(file="train.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "", colClasses = co)
# parallely create new encoded data which will be used for regression analysis
t_reg = data.frame(train_id = t$train_id)
rm(co)

########################################################################
########################### Data Exploration #############################
# check dimension
dim(t)
# count of each data type in total features
table(sapply(1:ncol(t), function(x) class(t[,x])))
# check column names
colnames(t)


###########################################################################
########################## missing value analysis ##########################
# count missing values in each column
missing_val = data.frame(apply(t, 2, function(x) {sum(is.na(x))}))
missing_val$col = row.names(missing_val)
names(missing_val)[1] = "Missing_Total_Number"
missing_val$percent_missing = (missing_val$Missing_Total_Number/nrow(t))*100
# sort by descending order
missing_val = missing_val[order(-missing_val$percent_missing),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1,3)]
# create file containing information about missing data points in each column
write.table(missing_val, file="missing_perc.txt", row.names = F, col.names=T, sep="\t", quote=F)
# plot missing value bar chart
require(ggplot2)
jpeg(file="missing_bar.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))
ggplot(missing_val, aes_string(x = reorder(as.factor(missing_val[,1]), -missing_val[,3]), y=missing_val[,3])) +
  geom_bar(stat="identity", fill = "Darkslateblue") +
  theme_bw() + 
  xlab("Variable Name") + ylab("Percent Missing Values") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=40), axis.text.x = element_text(angle=45, hjust = 1))
dev.off()
rm(missing_val)
# detach packages which are not needed so that RAM is free for other tasks
detach("package:ggplot2", unload=TRUE)

######################################################################
################ first feature: name preprocessing ###################

# load libraries ; try with R version 3.4.4 or above
# name has a lot of text so we will identify most useful terms for 
# predicting price and use them as dummy variables for prediction
# for example all computers will have different prices from all pens
# so this information will be encoded in these terms
install.packages("stringi")
install.packages("stringr")
install.packages("tm")
install.packages("wordcloud")
install.packages("slam")
install.packages("RSentiment")
require("stringi")
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
# to remove [rm] values from data
t$name= gsub("\\[rm\\]", "", t$name)
# remove all the other special characters 
t$name= gsub("[^a-zA-Z0-9 ]", "", t$name)
# create corpus so that text preprocessing methods can be applied
postCorpus = Corpus(VectorSource(t$name))
# convert all to lowercase
postCorpus = tm_map(postCorpus, tolower)
writeLines(as.character(postCorpus[[2]]))
# remove stop words
postCorpus = tm_map(postCorpus, removeWords, stopwords('english'))
writeLines(as.character(postCorpus[[1]]))
# remove punctuation marks
postCorpus = tm_map(postCorpus, removePunctuation)
writeLines(as.character(postCorpus[[1]]))
# remove unnecesary whitespaces
postCorpus = tm_map(postCorpus, stripWhitespace)
writeLines(as.character(postCorpus[[1]]))
# text stemming: convert to base word and remove suffix and prefix
postCorpus  = tm_map(postCorpus, stemDocument, language="english")
writeLines(as.character(postCorpus[[5]]))
# convert to string vector again
col1 = data.frame(text=sapply(postCorpus, identity))
# text lemmatization
require(textstem)
col1 = lemmatize_words(col1)
# create term document matrix to select useful terms
# here term frequency will be used as metric to find most useful word
tdm = TermDocumentMatrix(postCorpus, control=list(bounds=list(global=c(100, 10000)), minWordLength=2, weighting=weightTf))
dim(tdm)
terms = as.character(rownames(tdm))
# select terms which are present in most number of observations
# essentially selecting item names which are most sold on that shopping application
######### run on a cluster #####################################
# this code can be tested on laptop with fewer iterations
#v = as.numeric()
#for (i in 1:length(terms)) {
#  v1 = rowSums(as.matrix(tdm[i, 1:1482535])) # here 1482535 is total number of observations in the data
#  v = c(v, v1)
#  print(paste0("iteration:", i))
#} 
#v = sort(v, decreasing = T) # rank the item names based on frequency
#sel_terms = as.character(names(v[1:4260])) # 4260 are number of terms from tdm
#write.table(sel_terms, file="sel_terms_var1.csv", row.names=F, col.names=T, sep="\t", quote=F)
#####################################################################
# load the file from cluster
sel_terms = read.table(file="sel_terms_var1.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# use below code for creating dichotomous dummy variaables for selected terms
# But running this code will take more than one day so it was run on a 
# multicore cluster and results are stored in a file whiich is directly loaded here
# but still this code is functional and can be checked
# selected only top 50 terms due to compuatational resources limitations
################## run on a cluster #####################################
# this code can be tested on laptop with fewer iterations
#fea = data.frame(train_id = t[,1])
#for (i in 1:50) {
#  vec = as.integer()
#  for (j in c(1:length(col1$text))) {
#    c = as.integer(str_count(as.character(col1$text[j]), pattern=as.character(sel_terms[i])))
#    vec = c(vec, c)
#  } 
#  fea = cbind(fea, vec)
#  print(paste0("iteration:", i))
#} 
#dim(fea)
#rm(c,vec)
#write.table(fea, file="var1_name_coded.csv", row.names=F, col.names=F, sep="\t", quote=F)
#########################################################################
# load the file made in cluster
fea = read.table(file="var1_name_coded.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
fea = fea[,2:51]
# rename the column names for easy identification
mod_var1_terms = paste("var1", sel_terms$x[1:50], sep="_")
colnames(fea) = mod_var1_terms
# append to name variable encoded to regression file
t_reg = cbind(t_reg, fea)
rm(fea)

######################################################################################
################ second feature: item_condition_id preprocessing ######################

# find unique values for this variable
unique(t$item_condition_id)
# find number of missing of NA values
sum(is.na(t$item_condition_id)) # had 4539 missing data values
# assign to new variable for easy handling
var2 = as.numeric(t[,3])
# calculate mean, median, and mode of this variable
mean_1 = mean(var2, na.rm=T)
median_1 = median(var2, na.rm=T)
modefun = function(x) {
  uniq = unique(x)
  uniq[which.max(tabulate(match(x, uniq)))]
}
mode_1 = modefun(var2)
# imputation of missing values
# knn was not used because it gave results similar to mean median, or mode
# moreover knn compuation was taking a lot of time so not worth doing it here
# function for optimizing Imputation method among the mean, median, mode and
# then performing imputation with selected method
impopcon = function(data, i) {
  test = data[i]
  # imputation with mean
  data[i] = NA
  data[is.na(data[i])] = mean_1
  meant = data[i]
  # imputation with meadian
  data[i] = NA
  data[is.na(data[i])] = median_1
  mediant = data[i]
  #imputation with mode method
  data[i] = NA
  data[is.na(data[i])] = mode_1
  modet = data[i]
  # testing
  menatt = abs(test - meant) 
  mediantt = abs(test - mediant) 
  modett = abs(test - modet) 
  tt = c(menatt, mediantt, modett)
  tt1 = min(tt)
  if (tt1 == modett) {
    print ("Mode selected for imputation")
    data[is.na(data[i])] = mean_1
  } else if (tt1 == mediantt) {
    print ("Median selected for imputation")
    data[is.na(data[i])] = median_1
  } else {
    print ("Mean selected for imputation")
    data[is.na(data[i])] = mode_1
  }
  return(data)
}
#run function for performing impuation with most accurate method
var2_item_condition = impopcon(var2, 2) # median was most accurate so used for imputation
# check again for any missing values
sum(is.na(var2_item_condition)) #zero this time

# append to regression file
t_reg = cbind(t_reg, var2_item_condition)

# cleaning so that RAM is free
rm (impopcon, mean_1, median_1, mode_1, modefun, new_var3, var2, var3, var2_item_condition)

################################################################################################
################ third feature: category_name preprocessing ####################################

# define new variable for easy processing
var3 = t[,4]
# count number of unique values
length(unique(var3))
# count missing values
sum(is.na(var3))
# impute missing category name as unknown
var3[is.na(var3)] = "unknown/unknown/unknown"
sum(is.na(var3))
#remove special character such as & except / character this is 
# because we want to create three variables (cat1, cat2, cat3) from this one variable
var3 = gsub("\\[rm\\]", "", var3) # to remove [rm] values from data
var3 = gsub("[^a-zA-Z0-9 /]", "", var3)
# split this one column into three columns
require("reshape2")
many_var3 = as.data.frame(colsplit(var3, "/", c('cat1', 'cat2', 'cat3')))
length(unique(many_var3$cat1))
length(unique(many_var3$cat2))
length(unique(many_var3$cat3))

################################# dummy encoding for cat1 ######################################

# take unique values into new variable
cat1_terms = as.character(unique(many_var3$cat1))
# define new variable for cat1 for easy processing
cat1 = as.character(many_var3$cat1)
# write.table(cat1_terms, file="cat1_terms.csv", row.names=F, col.names=T, sep="\t", quote=F)
# write.table(cat1, file="cat1.csv", row.names=F, col.names=T, sep="\t", quote=F)
# use below code for creating dichotomous variaables for selected category one values
# But running this code will take more than one day so it was run on a 
# multicore cluster and results are stored in a file whiich is directly loaded here
# but still this code is functional and can be checked
# all the categories were selected as they are very crucial for making accurate prediction
################## run on cluster #####################################
# this code can be tested on laptop with fewer iterations
#fea = data.frame(train_id = t[,1])
#for (i in 1:c(1:length(cat1_terms))) {
#  vec = as.integer()
#  for (j in c(1:length(cat1))) {
#    c = as.integer(str_count(as.character(cat1[j]), pattern=as.character(cat1_terms[i])))
#    vec = c(vec, c)
#  } 
#  fea = cbind(fea, vec)
#  print(paste0("iteration:", i))
#}
#rm(c,vec)
#dim(fea)
#View(fea[1:20,1:5])
#write.table(fea, file="cat1_coded.csv", row.names=F, col.names=F, sep="\t", quote=F)
#rm(fea)
##################################################################### 
# load the file made in cluster
cat1_coded = read.table(file="cat1_coded.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
cat1_coded = cat1_coded[,2:12]
cat1_terms = read.table(file="cat1_terms.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# rename the column names for easy identification
mod_cat1_terms = paste("cat1", cat1_terms$x, sep="_")
colnames(cat1_coded) = mod_cat1_terms
dim(cat1_coded)
# perform PCA analysis to select the most useful linear combinations 
# which explains the highest amount of variance
mypc <- prcomp(cat1_coded, scale. = T, center = T)
# save pca model so that it can be used to calculate PCs for the new test data
save(mypc, file="mypc.RData")
require(dplyr)
require(tibble)
require(ggfortify)
std_dev <- mypc$sdev
pr_var <- std_dev^2
# create scree plot for proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
png(filename = "cat1_cumulative_scree.png", width = 15, height = 15, units='in', res=300, bg = "white")
par(oma=c(0,0,0,0), mar=c(7,8,2,2), mgp=c(4.7, 1.5, 0), las=0)
plot(cumsum(prop_varex), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b", col="brown4",cex = 3, pch=16, lwd=2.5, cex.lab = 3, cex.axis = 3)
dev.off()
# selected final 8 PCs which explain more than 80% of variance 
# now these PCs will be used for training the model
cat1_pc_sel = tbl_df(mypc$x) %>% select(PC1:PC8)
dim(cat1_pc_sel)
# rename the PC names for easy identification
cat1_pc_colname = paste("cat1", colnames(cat1_pc_sel), sep="_")
colnames(cat1_pc_sel) = cat1_pc_colname
# append to regression file
t_reg = cbind(t_reg, cat1_pc_sel)
# cleaning
rm(cat1, cat1_coded, cat1_pc_colname, cat1_pc_sel, cat1_terms, co_name, mod_cat1_terms, mypc, pr_var, prop_varex, std_dev)


# define new data frame for label encoding cat1 and cat2 
t1 = as.data.frame(cbind(as.numeric(t[,1]), as.character(many_var3$cat1), as.character(many_var3$cat2), as.character(many_var3$cat3), as.numeric(t[,6])))
#write.table(t1, file="t1_for_cat2_3.csv", row.names=F, col.names=F, sep="\t", quote=F)
#t1 = read.table(file="t1_for_cat2_3.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
t1$V1 = as.numeric(as.character(t1$V1))
t1$V5 = as.numeric(as.character(t1$V5))


################################# label encoding for cat2 ################################################
# define new variable for easy processing
cat2 = t1[,3]
length(unique(cat2))
sum(is.na(cat2))
# alternative strategy of label encoder was used
# due to high number of levels for this variable
# here ranking was done based on the mean price value
# this is because the price value is our target variable
cat2_uniq = as.character(unique(cat2))
cat2_mean = as.numeric()
rm(a)
for (i in c(1:length(cat2_uniq))) {
  a = mean(t1[which(t1[,3] == cat2_uniq[i]),5], na.rm=T)
  cat2_mean = c(cat2_mean, a)
  print(paste0("interation:", i))
}
length(cat2_uniq)
length(cat2_mean)
cat2_mean_df = as.data.frame(cbind(cat2_uniq, cat2_mean))
dim(cat2_mean_df)
View(cat2_mean_df)
cat2_mean_df = cat2_mean_df[order(cat2_mean),]
num = c(1:114)
View(num)
cat2_mean_df = cbind(cat2_mean_df, Rank = num)
# write.table(cat2_mean_df, file="cat2_mean_df.csv", row.names=F, col.names=T, sep="\t", quote=F)
# cat2_mean_df = read.table(file="cat2_mean_df.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# label encoding
# cat2_coded = as.numeric()
##################### run on cluster #############
# this code can be tested on laptop with fewer iterations
#for (i in c(1:nrow(t1))) {
#  a = match(as.character(t1[i,3]), as.character(cat2_mean_df$cat2_uniq))
#  cat2_coded = c(cat2_coded, a)
#}
#rm(a)
# write.table(cat2_coded, file="cat2_coded.csv", row.names=F, col.names=F, sep="\t", quote=F)
##################################################
# load the output file from cluster
cat2_coded = read.table(file="cat2_coded.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
cat2_label_encoded = cat2_coded$V1
# append to regression file
t_reg = cbind(t_reg, cat2_label_encoded)
#cleaning
rm(a, cat2, cat2_coded, cat2_label_encoded, cat2_mean, cat2_mean_df, cat2_uniq, i, num,)


################################# label encoding for cat3 ###################################################
# define new variable for easy handling
cat3 = t1[,4]
length(unique(cat3))
sum(is.na(cat3))
# alternative strategy of label encoder 
# due to high number of levels for this variable
# Ranking based on the mean price value
# as the price value is our target variable
cat3_uniq = as.character(unique(cat3))
cat3_mean = as.numeric()
rm(a)
for (i in c(1:length(cat3_uniq))) {
  a = mean(t1[which(t1[,4] == cat3_uniq[i]),5], na.rm=T)
  cat3_mean = c(cat3_mean, a)
  print(paste0("interation:", i))
}
length(cat3_uniq)
length(cat3_mean)
cat3_mean_df = as.data.frame(cbind(cat3_uniq, cat3_mean))
dim(cat3_mean_df)
View(cat3_mean_df)
cat3_mean_df = cat3_mean_df[order(cat3_mean),]
num = c(1:869)
View(num)
cat3_mean_df = cbind(cat3_mean_df, Rank = num)
# write.table(cat3_mean_df, file="cat3_mean_df.csv", row.names=F, col.names=T, sep="\t", quote=F)
# cat3_mean_df = read.table(file="cat3_mean_df.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# label encoding
# cat3_coded = as.numeric()
##################### run on cluster ######################
# this code can be tested on laptop with fewer iterations
#for (i in c(1:nrow(t1))) {
#  a = match(as.character(t1[i,4]), as.character(cat3_mean_df$cat3_uniq))
#  cat3_coded = c(cat3_coded, a)
#}
#rm(a)
#write.table(cat3_coded, file="cat3_coded.csv", row.names=F, col.names=F, sep="\t", quote=F)
#############################################################
# load the output file from cluster
cat3_coded = read.table(file="cat3_coded.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
cat3_label_encoded = cat3_coded$V1
# append to regression file
t_reg = cbind(t_reg, cat3_label_encoded)
#cleaning
rm(a, cat3, cat3_coded, cat3_label_encoded, cat3_mean, cat3_mean_df, cat3_uniq, i, many_var3, num, var3, t1)

##################################################################################################################
################ Fourth feature: brand name preprocessing ########################################################

# define new variable for easy handling
var4 = t[,5]
length(unique(var4))
sum(is.na(var4))
# impute missing category name as unknown
var4[is.na(var4) == TRUE] = "unknown"
sum(is.na(var4))
# here also the label encoder strategy as used
# Ranking was done based on the mean price value
# as the price value is our target variable
var4_uniq = as.character(unique(var4))
var4_mean = as.numeric()
t[,5][is.na(t[,5]) == TRUE] = "unknown"
rm(a)
for (i in c(1:length(var4_uniq))) {
  a = mean(t[which(t[,5] == var4_uniq[i]),6], na.rm=T)
  var4_mean = c(var4_mean, a)
  print(paste0("interation:", i))
}
length(var4_uniq)
length(var4_mean)
var4_mean_df = as.data.frame(cbind(var4_uniq, var4_mean))
dim(var4_mean_df)
var4_mean_df = var4_mean_df[order(var4_mean),]
num = c(1:4804)
View(num)
var4_mean_df = cbind(var4_mean_df, Rank = num)
# write.table(var4_mean_df, file="var4_mean_df.csv", row.names=F, col.names=T, sep="\t", quote=F)
# var4_mean_df = read.table(file="var4_mean_df.csv", header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
# label encoding
# bn_coded = as.numeric()
##################### run on cluster ######################
# this code can be tested on laptop with fewer iterations
#for (i in c(1:nrow(t))) {
#  a = match(as.character(t[i,5]), as.character(var4_mean_df$var4_uniq))
#  bn_coded = c(bn_coded, a)
#}
#rm(a)
#write.table(bn_coded, file="bn_coded.csv", row.names=F, col.names=F, sep="\t", quote=F)
##############################################################
# load result file from cluster
bn_coded = read.table(file="bn_coded.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
bn_label_encoded = bn_coded$V1
# append to regression file
t_reg = cbind(t_reg, bn_label_encoded)
#cleaning
rm(a, bn_coded, bn_label_encoded, i, num, var4, var4_uniq, var4_mean, var4_mean_df)

####################################################################################
################ Fifth feature: shipping preprocessing #############################

# define new variable for easy processing
var5 = t[,7]
unique(var5)
# identify number of missing values
sum(is.na(var5))
# define function to calculate the mode
modefun = function(x) {
  uniq = unique(x)
  uniq[which.max(tabulate(match(x, uniq)))]
}
# calculate mode for this variable
mode_5 = modefun(var5)
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

var6 = t[,8]
length(var6)
# we look for specific words in the description 
# which will help us predicting the price
# for example something in great condition will have high price 
# in comparison to somthing not functional
# remove all special characters
var6 = gsub("\\[rm\\]", "", var6) # to remove [rm] values from data
var6 = gsub("[^a-zA-Z0-9 ]", "", var6)
# load packages for text analysis
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
# create term document matrix to select useful terms
tdm = TermDocumentMatrix(postCorpus, control=list(bounds=list(global=c(14000, 1482535)), minWordLength=2, weighting=weightTf))
dim(tdm)
terms = as.character(rownames(tdm))
# select most frequent words or terms which describe item quality
v = as.numeric()
for (i in 1:length(terms)) {
  v1 = rowSums(as.matrix(tdm[i, 1:1482535])) # 1482535 are number of observations from data
  v = c(v, v1)
  print(paste0("iteration:", i))
} 
v = sort(v, decreasing = T)
sel_terms = as.character(names(v[1:262])) # 262 are number of terms from tdm
# write.table(sel_terms, file="sel_terms_var6.csv", row.names=F, col.names=F, sep="\t", quote=F)
# create new variable for easy searching
s = c(1:1482535)
ss = data.frame(cbind(s, col1 = as.character(col1$text)))
# write.table(ss, file="var6_to_search_into.csv", row.names=F, col.names=F, sep="\t", quote=F)
# use below code for creating dichotomous variables for selected terms
# But running this code will take more than one day so it was run on a 
# multicore cluster and results are stored in a file whiich is directly loaded here
# but still this code is functional and can be checked
# selected top 50 terms due to computational resources limitations
################## run on cluster #####################################
# this code can be tested on laptop with fewer iterations
#fea = data.frame(train_id = t[,1])
#for (i in 1:50) {
#  vec = as.integer()
#  for (j in c(1:length(ss[,2]))) {
#    c = as.integer(str_count(as.character(ss[j,2]), pattern=as.character(sel_terms[i])))
#    vec = c(vec, c)
#  } 
#  fea = cbind(fea, vec)
#  print(paste0("iteration:", i))
#} 
#rm(vec,c)
#dim(fea)
#write.table(fea, file="item_description.csv", row.names=F, col.names=F, sep="\t", quote=F)
#########################################################################
# load the output from multicore cluster
var6_coded = read.table(file="var6_coded.csv", header=F, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
var6_coded = var6_coded[,2:51]
mod_var6_terms = paste("var6_disc", sel_terms, sep="_")
colnames(var6_coded) = mod_var6_terms[1:50]
dim(var6_coded)
# append to regression file
t_reg = cbind(t_reg, var6_coded)
#cleaning
rm(col1,i, mode_var6_terms, s, sel_terms, ss, v, v1, var6, var6_coded, var6_terms)

#####################################################################################
################ Seventh target variable: Price ####################################

# define new variable for easy preocessing
var7_Price = t[,6]
length(var7_Price)
length(unique(var7_Price))
summary(var7_Price)
sum(is.na(var7_Price))
# add target variable at end
# append to regression file
t_reg = cbind(t_reg, var7_Price)
# remove observations with missing price value as they can not be used for training
t_reg = t_reg1[complete.cases(t_reg), ]


#####################################################################################
################ Final file for modeling ############################################

# With this all the variable encoding is complete 
# Now In total we have 114 variables
# of which last is our target variable which is price
# and we have 113 predictor variables
# all these predictor variable are numeric as all of them are coded by either dummy strategy or label encoding strategy

# Now create final file which will be used for modeling
write.table(t_reg, file="Final_for_modeling.csv", row.names=F, col.names=T, sep="\t", quote=F)

