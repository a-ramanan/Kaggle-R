#Kaggle Analytics Edge Competition
#Predicting if iPads on eBay will be sold

library(tm)
library(caret)
library(caTools)
library(glmnet)
library(randomForest)

eBayTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors = FALSE)
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors = FALSE)

#changing into factor variables

eBayTrain$condition = as.factor(eBayTrain$condition)
eBayTrain$storage = as.factor(eBayTrain$storage)
eBayTrain$productline = as.factor(eBayTrain$productline)
eBayTrain$carrier = as.factor(eBayTrain$carrier)
eBayTrain$color = as.factor(eBayTrain$color)
eBayTrain$cellular = as.factor(eBayTrain$cellular)
eBayTest$condition = as.factor(eBayTest$condition)
eBayTest$storage = as.factor(eBayTest$storage)
eBayTest$productline = as.factor(eBayTest$productline)
eBayTest$carrier = as.factor(eBayTest$carrier)
eBayTest$color = as.factor(eBayTest$color)
eBayTest$cellular = as.factor(eBayTest$cellular)

#Preprocessing the Description variable in train and test

CorpusDescription = Corpus(VectorSource(c(eBayTrain$description, eBayTest$description)))

CorpusDescription = tm_map(CorpusDescription, content_transformer(tolower), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, PlainTextDocument, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removePunctuation, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removeWords, stopwords("english"), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, stemDocument, lazy=TRUE)

#Creating the document term matrix
dtm = DocumentTermMatrix(CorpusDescription)
wordCount = rowSums(as.matrix(dtm))

sparse = removeSparseTerms(dtm, 0.99)

PreprocWords = as.data.frame(as.matrix(sparse))
colnames(PreprocWords) = paste0("frTr",colnames(PreprocWords))

PreprocTrain = head(PreprocWords, nrow(eBayTrain))
PreprocTest =tail(PreprocWords, nrow(eBayTest))
PreprocTrain$wordCount = head(wordCount, nrow(eBayTrain))
PreprocTest$wordCount = tail(wordCount, nrow(eBayTest))

#Normalising the data
varAdd = c("biddable", "startprice")
varAddChar = c("condition", "storage", "productline")
normTrain = cbind(eBayTrain[,varAdd], PreprocTrain)
normTest = cbind(eBayTest[, varAdd], PreprocTest)
preproc = preProcess(normTrain)
normTrain = predict(preproc, normTrain)
normTest = predict(preproc, normTest)
normTrain = cbind(eBayTrain[,varAddChar], normTrain)
normTest = cbind(eBayTest[,varAddChar], normTest)
normTrain$sold = eBayTrain$sold


#Splitting training data further in to train and test to compute accuracy of models
set.seed(144)
split = sample.split(normTrain$sold, SplitRatio = 0.75)
train1 = subset(normTrain, split ==  TRUE)
test1 = subset(normTrain, split == FALSE)

#LR3 model that has the highest AUC on leaderboard
LR3 = glm(sold ~ frTr100+frTrhous+frTrtest+startprice+biddable+condition+storage+productline, data=normTrain, family="binomial")
PredLR3 = predict(LR3, newdata=normTest, type="response")
RMSE1 = sqrt((sum((test1$sold - PredLR3)^2))/nrow(test1))

# I - Lasso Regression with the same variables as in LR3
cols = c(1:6,33,67)
logCV = cv.glmnet(x=data.matrix(train1[,cols]), y=train1[,79], family="binomial", type.measure = "auc", alpha=0.3)
Pred2 = predict(logCV, type="response", newx = data.matrix(test1[,cols]), s=c('lambda.1se','lambda.min'))
RMSE2 = sqrt((sum((test1$sold - Pred2)^2))/nrow(test1))

# II - RandomForest
train1$sold = as.factor(train1$sold)
RF = randomForest(sold ~ frTr100+frTrhous+frTrtest+startprice+biddable+condition+storage+productline, data=train1, ntree=1000)
Pred2 = predict(RF, newdata=test1, type="prob")
RMSE2 = sqrt((sum((test1$sold - Pred2)^2))/nrow(test1))

# III - LR 
LR = glm(sold ~ frTr100+frTrhous+frTrtest+startprice*biddable+condition+storage+productline, data=normTrain, family="binomial")
Pred2 = predict(LR, newdata=normTest, type="response")
RMSE2 = sqrt((sum((test1$sold - Pred2)^2))/nrow(test1))

#Constructing an ensemble
my_predictions = (PredLR3*3 + Pred2*2)/5
RMSE = sqrt((sum((test1$sold - my_predictions)^2))/nrow(test1))

print(paste(RMSE1, RMSE2, RMSE, sep=" "))
## ensemble that worked best is LR3 and LR (III) CUE FOR LATER!!! : TRY OTHER INTERACTION TERMS
#Output for reference
##Constructing an ensemble
#my_predictions = (PredLR3 + Pred2)/2
#RMSE = sqrt((sum((test1$sold - my_predictions)^2))/nrow(test1))
 
#print(paste(RMSE1, RMSE2, RMSE, sep=" "))
#"0.378229814646293 0.373424365059881 0.375086644311529"
##Constructing an ensemble
#my_predictions = (PredLR3 + Pred2*2)/3
#RMSE = sqrt((sum((test1$sold - my_predictions)^2))/nrow(test1))
 
#print(paste(RMSE1, RMSE2, RMSE, sep=" "))
#"0.378229814646293 0.373424365059881 0.374366672020961"
my_solution = data.frame(UniqueID = eBayTest$UniqueID, Probability1 = my_predictions)
write.csv(my_solution, file="ensembleLR3andLRwith32weights.csv", row.names = FALSE)
