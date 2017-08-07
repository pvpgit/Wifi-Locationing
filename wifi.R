# imports
library(dplyr)
library(caret)

data <- read.table(
  '~/Desktop/UJIndoorLoc/trainingData.csv',
  sep=",",
  nrows=1000,
  header=TRUE,
  as.is=TRUE
)

testdata<- read.table(
  '~/Desktop/UJIndoorLoc/ValidationData.csv',
  sep=",",
  #nrows=100,
  header=TRUE,
  as.is=TRUE
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

# generate target Y label from distinct factor interactions of
# FLOOR, BUILDINGID, SPACEID, and RELATIVEPOSITION (FBSR)
# FLOOR, BUILDING, SPACEID (FBS)
# FLOOR, BUILDING (FB)
# BUILDING (B)
data <- cbind(data,interaction(data[,523:526], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FBSR"

data <- cbind(data,interaction(data[,523:525], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FBS"

data <- cbind(data,interaction(data[,523:524], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FB"

data <- cbind(data,interaction(data[,523], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "B"

# drop source FLOOR, BUILDINGID, SPACEID, and RELATIVEPOS cols
data <- data[,-c(523:526)]

testdata <- cbind(testdata,interaction(testdata[,523:526], sep="-", drop=TRUE))
colnames(testdata)[ncol(testdata)] <- "FBSR"

testdata <- cbind(testdata,interaction(testdata[,523:525], sep="-", drop=TRUE))
colnames(testdata)[ncol(testdata)] <- "FBS"

testdata <- cbind(testdata,interaction(testdata[,523:524], sep="-", drop=TRUE))
colnames(testdata)[ncol(testdata)] <- "FB"

testdata <- cbind(testdata,interaction(testdata[,523], sep="-", drop=TRUE))
colnames(testdata)[ncol(testdata)] <- "B"

testdata <- testdata[,-c(523:526)]
#testdata<-testdata[,c(69,70,77,78,84,87,332,496,517,530)]

# stack all WAP factors in one column, this allows us to
# transform values without looping over column indexes,
# something not really possible in R
wapStack <- stack(
  dplyr::select(
    data, dplyr::starts_with("WAP")
  )
)

# move the row.names df index into the dataframe itself
# this is needed because the dplyr transformations will 
# lose the original df index which is needed to reconstruct
# the wapStack when we rejoin the dplyr data 'tibbles'
wapStack <- cbind(wapStack, row.names(wapStack))
colnames(wapStack)[ncol(wapStack)] <- "id"
wapStack <- wapStack[,c(ncol(wapStack), 1:(ncol(wapStack)-1))]

# change values of 100 RSSI to zero (no signal)
wapNulls <- dplyr::filter(wapStack, values==100) %>%
  dplyr::transmute(id, values = 0, ind)

# convert remaining RSSI values to linear scale between 0-1
wapVals <- dplyr::filter(wapStack, values!=100) %>%
  dplyr::transmute(id, values = 10^(values/10)/1000*10^12, ind)

# join the 'no signal' and 'signal' frames by rows
wapJoined <- rbind(wapNulls, wapVals)

# overwrite the df idx with the 'id' field
row.names(wapJoined) <- wapJoined[,1]

# sort the joined df asc by the id field
# note that id must be converted from factor to num
wapJoined[,1] <- as.numeric(wapJoined[,1])
wapJoined <- dplyr::arrange(wapJoined, id) %>%
  dplyr::select(-id) # drop the id column

# finally, unstack the joined set to get our factor 
# matrix back with no '100' values and linear RSSIs
wapClean <- unstack(wapJoined)

# merge the wapClean df back into the original data df
data <- cbind(
  wapClean, 
  dplyr::select(data, -(starts_with("WAP")))
)

# clean up intermediary tables
wapStack <- NULL
wapNulls <- NULL
wapVals <- NULL
wapJoined <- NULL
wapClean <- NULL

# reduce feature space - find near zero variance columns in 
# the WAP feature space (1:521) and drop them. Tuned 
# freq/uniqueCut args to compensate for the sparse nature 
# of sampling (somewhat)
data <- data[,-c(
  caret::nearZeroVar(
    data[,1:529],
    # saveMetrics=TRUE, # valuable info for tuning
    freqCut = 50,
    uniqueCut = 1,
    names = FALSE
  )
)]

# Clear the unused columns in test data set.
data.columns <- colnames(data)
testdata <- testdata[, ! names(testdata) %in% setdiff(names(testdata), data.columns), drop = F]
testdata.columns <- colnames(testdata)
#testdata.columns 

# create X and Y (factors and target) matricies
X <- dplyr::select(data, starts_with("WAP"))

# create class label Y matrix, FLOOR.BLDG.SPACEID
#Y_FBS <- dplyr::select(data, ends_with("FBS"))
#Y_FBS <- Y_FBS[,1] # caret accepts only vector, not data.frame

# create class label Y matrix, FLOOR.BLDG (no SPACEID)
Y_FB <- dplyr::select(data, ends_with("FB"))
Y_FB <- Y_FB[,1] # caret accepts only vector, not data.frame

# create trainControl object
set.seed(1337)
ctrl <- trainControl(
  method="cv",
  repeats=1
  )

# train knn model
modKnn <- train( 
  X,
  Y_FB, 
  #data=data,
  method = "knn",
  preProcess = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl
)

modKnn

##predict
knnPredict <- predict(modKnn,data=testdata)
table(knnPredict)

##Confusion Matrix
#confusionMatrix(knnPredict, testdata$FB)
confusionMatrix(modKnn, testdata$Y_FB)
write.csv(knnPredict, "~/Desktop/predictionsKNN.csv", row.names = F)

## train svm model
library(LiblineaR)
modSvm <- train(
  X,
  Y_FB, #note that FBS or FBSR will probably not work here
  method = 'svmLinear3',
  preProcess = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl
)

modSvm
plot(modSvm, scales = list(x = list(log =2)))
svmPredict <- predict(modSvm,data=testdata)
table(svmPredict)

##Confusion Matrix
confusionMatrix(modSvm, testdata$Y_FB)
confusionMatrix(svmPredict, testdata$FB)
write.csv(svmPredict, "~/Desktop/predictionssvm.csv", row.names = F)

##Random Forest Model
library(randomForest)
##Random Forest Model 10 fold cross validation
modRF <- train(X,Y_FB, data=data, method = "rf",
               preProcess = c("center","scale"), tuneLength = 8, trControl = ctrl)

modRF
plot(modRF, print.thres = 0.5, type="S")

##predict
rfPredict <- predict(modRF, newdata = testdata)
table(rfPredict)

#confusionMatrix(modRF, testdata$Y_FB)
confusionMatrix(rfPredict, testdata$FB)

#C50
library(C50)
##Random Forest Model 10 fold cross validation
modC50 <- train(X,Y_FB, method="C5.0", trControl=ctrl)
              
modC50
summary(modC50)
plot(modC50)
C5pred<-predict(modC50, testdata)
confusionMatrix(C5pred, testdata$FB)

#CART
library(rpart)
library(rpart.plot)
modCART <- train(X,Y_FB, method = "rpart", 
           tuneLength = 8, trControl = ctrl)
modCART
CARTPredict <- predict(modCART, newdata = testdata)
confusionMatrix(CARTPredict, testdata$FB)
#rpartProbs<-predict(modCART, testdata, type="prob")
#library(pROC)
#rpartROC<-roc(testdata$FB, rpartProbs[,"3-2"], levels=rev(testdata$FB))
#histogram(rpartProbs)

cvValues<-resamples(list(KNN=modKnn, SVM=modSvm, RF=modRF, C50=modC50, CART=modCART))
cv2Values<-resamples(list(RF=rfPredict, C50=C5pred, RPART=CARTPredict)) 
summary(cvValues)

splom(cvValues)
dotplot(cvValues, main="Dot Plot")
parallelplot(cvValues, main="Main Label")
xyplot(cvValues)

Diffs<-diff(cvValues) 
summary(Diffs)
dotplot(Diffs)

