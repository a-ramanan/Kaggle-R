#Kaggle Analytics Edge Competition
#Predicting if iPads on eBay will be sold using lasso regression with the glmnet package

library(tm)
library(caret)
library(caTools)
library(glmnet)

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




# Running lasso logistic regression with all variables in the training set

logCV = cv.glmnet(x=data.matrix(normTrain[,1:78]), y=normTrain[,79], family="binomial", type.measure = "auc")
prediction = predict(logCV, type="response", newx = data.matrix(normTest[1:78]), s='lambda.min')

#Running lasso for LR3 variables
cols = c(1:6,33,67)
logCV = cv.glmnet(x=data.matrix(normTrain[,cols]), y=normTrain[,79], family="binomial", type.measure = "auc")
prediction = predict(logCV, type="response", newx = data.matrix(normTest[,cols]), s='lambda.min')


my_solution = data.frame(UniqueID = eBayTest$UniqueID, Probability1 = prediction)
write.csv(my_solution, file="glmnetlasso.csv", row.names=FALSE)
