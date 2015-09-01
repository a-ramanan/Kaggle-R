#Kaggle Analytics Edge Competition
#Predicting if iPads on eBay will be sold

library(tm)
library(caret)
library(caTools)
library(randomForest)
library(e1071)
library(mice)

eBayTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors = FALSE)
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors = FALSE)

#changing into factor variables

#Dependent Variable as factor for RF
eBayTrain$sold = factor(eBayTrain$sold, labels = c("No", "Yes"))

eBayTrain$condition = as.factor(eBayTrain$condition)
eBayTrain$storage = as.factor(eBayTrain$storage)
eBayTrain$productline = as.factor(eBayTrain$productline)
eBayTrain$carrier = as.factor(eBayTrain$carrier)
eBayTrain$color = as.factor(eBayTrain$color)
#eBayTrain$cellular = as.factor(eBayTrain$cellular)
eBayTest$condition = as.factor(eBayTest$condition)
eBayTest$storage = as.factor(eBayTest$storage)
eBayTest$productline = as.factor(eBayTest$productline)
eBayTest$carrier = as.factor(eBayTest$carrier)
eBayTest$color = as.factor(eBayTest$color)
#eBayTest$cellular = as.factor(eBayTest$cellular)

##NAs introduced because of type conversion to integers
eBayTrain$cellular = as.integer(eBayTrain$cellular)
eBayTest$cellular= as.integer(eBayTest$cellular)

#Impute NA values
eBayTrain$carrier[eBayTrain$carrier == "Unknown"] = NA
eBayTrain$color[eBayTrain$color == "Unknown"] = NA
eBayTrain$storage[eBayTrain$storage == "Unknown"] = NA

eBayTest$carrier[eBayTest$carrier == "Unknown"] = NA
eBayTest$color[eBayTest$color == "Unknown"] = NA
eBayTest$storage[eBayTest$storage == "Unknown"] = NA

var.for.imputation = c("cellular", "carrier", "color", "storage")
imputed = complete(mice(eBayTrain[var.for.imputation]))
eBayTrain[var.for.imputation] = imputed


imputed = complete(mice(eBayTest[var.for.imputation]))
eBayTest[var.for.imputation] = imputed

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
varAdd = c("biddable", "startprice", "cellular")
varAddChar = c("condition", "storage", "productline", "color")
normTrain = cbind(eBayTrain[,varAdd], PreprocTrain)
normTest = cbind(eBayTest[, varAdd], PreprocTest)
preproc = preProcess(normTrain)
normTrain = predict(preproc, normTrain)
normTest = predict(preproc, normTest)
normTrain = cbind(eBayTrain[,varAddChar], normTrain)
normTest = cbind(eBayTest[,varAddChar], normTest)
normTrain$sold = eBayTrain$sold


#varAdd = c( "sold", "biddable", "startprice", "condition", "storage", "productline")
#varAddTest = c( "biddable", "startprice", "condition", "storage", "productline")
#PreprocTrain = cbind(eBayTrain[,varAdd], PreprocTrain)
#PreprocTest = cbind(eBayTest[,varAddTest], PreprocTest)


#Splitting training data further in to train and test to compute accuracy of models
set.seed(144)
split = sample.split(normTrain$sold, SplitRatio = 0.75)
train1 = subset(normTrain, split ==  TRUE)
test1 = subset(normTrain, split == FALSE)


#Random Forest Model
set.seed(88)
ctrl = trainControl(method="repeatedcv", number=20, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
RF = train(sold ~ ., data=normTrain, method="rf", trControl = ctrl, metric="ROC", type="Classification")
RF.pred = predict(RF, newdata = normTest, type="prob")

my_solution = data.frame(UniqueID = eBayTest$UniqueID, Probability1 = RF.pred[,2])
write.csv(my_solution, file="RF.csv", row.names=FALSE)
