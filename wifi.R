library(dplyr)
library(caret)
library(corrplot) 
library("e1071")
library("caret")

data <- read.table(
  '~/Desktop/UJIndoorLoc/trainingData.csv',
  sep=",",
  header=TRUE,
  as.is=TRUE
  #nrows=100
)
testdata<- read.table(
  '~/Desktop/UJIndoorLoc/ValidationData.csv',
  sep=",",
  header=TRUE,
  as.is=TRUE
  #nrows=100
)
# select sum of row counts, groupby column, where 
# value = NULL (na) and count > 0. Then drop any na's
naCount <- data.frame(
  sapply(
    data,
    function(y) sum(
      length(
        which(
          is.na(y)
        )
      )
    )
  )
)

subset(naCount, naCount[1] > 0)
data <- na.omit(data)
naCount <- NULL

#generate target Y label from distinct factor interactions of
# FLOOR, BUILDING (FB)
# BUILDING (B)

data <- cbind(data,interaction(data[,523:524], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FB"

data <- cbind(data,interaction(data[,523], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "B"

# drop source FLOOR, BUILDINGID, SPACEID, and RELATIVEPOS cols
data <- data[,-c(523:526)]

# reduce feature space - find near zero variance columns in 
# the WAP feature space (1:521) and drop them. Tuned 
# freq/uniqueCut args to compensate for the sparse nature 
# of sampling (somewhat)
data <- data[,-c(
  caret::nearZeroVar(
    data[,1:520],
    # saveMetrics=TRUE, # valuable info for tuning
    freqCut = 50,
    uniqueCut = 1,
    names = FALSE
  )
)]

# create X and Y (factors and target) matricies
X <- dplyr::select(data, (starts_with("WAP")))

Y <- dplyr::select(data, (starts_with("FB")))
Y <- Y[,1] # caret accepts only vector, not data.frame

# visualize covariance matrix
# note that you can't draw cor to Y because Y is a factor
corrplot(cor(X)) # this could be cleaner

data<-data[,-c(10:14)]

# note that there are only 76 labels for 100 instances (rows)
# this is 1.3 samples per class (not that great)

ptr<-0.7
inTrain <- createDataPartition(y = data$FB, p=ptr, list=FALSE)
training <- data[ inTrain,]
testing <- data[-inTrain,]
nrow(training)
nrow(testing)





# create trainControl object
set.seed(1337)
ctrl <- trainControl(
  method="cv",
  repeats=5
)

testdata <- cbind(testdata,interaction(testdata[,523:524], sep="-", drop=TRUE))
colnames(testdata)[ncol(testdata)] <- "FB"

testdata <- cbind(testdata,interaction(testdata[,523], sep="-", drop=TRUE))
colnames(testdata)[ncol(testdata)] <- "B"

testdata<-testdata[,c(69,70,77,78,84,87,332,496,517,530,531)]


## KNN Model
# train dat model 
KnnFit <- train(X,Y , data=training, method = "knn",trControl = ctrl,
                preProcess = c("center","scale"), tuneLength = 6)

KnnFit

##predict
knnPredict <- predict(KnnFit, newdata = testdata)
table(knnPredict)

##Confusion Matrix
confusionMatrix(knnPredict, testdata$FB)

write.csv(knnPredict, "~/Desktop/predictionsKNN.csv", row.names = F)

##Random Forest Model
library(randomForest)
##Random Forest Model 10 fold cross validation
rfFit <- train(X,Y, data=data, method = "rf",
                preProcess = c("center","scale"), tuneLength = 8, trControl = ctrl)

rfFit
plot(rfFit, print.thres = 0.5, type="S")

##predict
rfPredict <- predict(rfFit, newdata = testdata)
table(rfPredict)

confusionMatrix(rfPredict, testdata$FB)

#Create a predictions.csv file on desktop
write.csv(rfPredict, "~/Desktop/predictionsRF.csv", row.names = F)

## SVM Model
svmFit <- svm(X,Y, data=data, 
              ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)), trControl = ctrl)
svmFit

##not working
##predict
svmPredict<-predict(svmFit,testdata)
table(svmPredict)

confusionMatrix(svmPredict, testdata$FB)



##C5.0
install.packages("C50")
library(C50)

##J48
##Linear
lmFit <- lm(X,Y, data=data)
lmFit

##predict
lmPredict <- predict(lmFit,testdata, level=0.95)
table(lmPredict)

#confusionMatrix(rfPredict, testdata$FB)
