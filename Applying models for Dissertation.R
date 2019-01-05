############################################################################################
######################### Human Activity Recognition-Classification ########################
############################################################################################
####################################Coded by Eren Aktas#####################################
############################################################################################
### Importing necessary libraries to use the related functions
.libPaths()
install.packages('e1071', dependencies = TRUE)
install.packages('party')
install.packages('randomForest')
library(class) 
library(e1071)
library(party)
library(randomForest)
library(MASS)
library(caret)

set.seed(42)

#############################################################################################
##################################### Data Understanding ####################################
#############################################################################################
# Dataset Folder downloaded from:
# https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

### Data Integration Part
dataSources <- "UCI_HAR_Dataset"

setwd("C:/Users/AKTAS/Desktop/Dissertation/UCI_HAR_Dataset")

if(!file.exists(dataSources)){
  ttable <- read.table("activity_labels.txt", sep = "")
  activity_labels <- as.character(ttable$V2)
  ttable <- read.table("features.txt", sep = "")
  attribute_names <- ttable$V2
  
  Xtrain <- read.table("train/X_train.txt", sep = "")
  names(Xtrain) <- attribute_names
  Ytrain <- read.table("train/y_train.txt", sep = "")
  names(Ytrain) <- "Activity "
  Ytrain$Activity <- as.factor(Ytrain$Activity)
  levels(Ytrain$Activity) <- activity_labels
  train_subjects <- read.table("train/subject_train.txt", sep = "")
  names(train_subjects) <- "subject "
  train_subjects$subject <- as.factor(train_subjects$subject )
  train_data <- cbind (Xtrain, train_subjects, Ytrain )
  
  Xtest <- read.table("test/X_test.txt", sep = "")
  names(Xtest) <- attribute_names
  Ytest <- read.table("test/y_test.txt", sep = "")
  names(Ytest) <- "Activity "
  Ytest$Activity <- as.factor( Ytest$Activity )
  levels(Ytest$Activity) <- activity_labels
  test_subjects <- read.table("test/subject_test.txt", sep = "")
  names(test_subjects) <- "subject"
  test_subjects$subject <- as.factor(test_subjects$subject )
  test_data <- cbind (Xtest, test_subjects, Ytest )
  
  save(train_data, test_data, file = dataSources)
  remove(ttable, activity_labels, attribute_names, Xtrain, Ytrain,
         train_subjects, Xtest, Ytest, test_subjects)
}

### Data Combining Part
dataSources <- "UCI_HAR_Dataset"
load(dataSources)

test_data <- test_data[,-563]
train_data <- train_data[,-c(562,564)]
HAR <- rbind(train_data, test_data)

remove(train_data, test_data, dataSources)

############################################################################################
######################## Data Exploration & Preparation ####################################
############################################################################################
dim(HAR) #Dimensions

colnames(HAR) #Feature Names

summary(HAR) #Structure of Dataset

attributes(HAR) #Attributes

View(HAR[,c(562,563)]) #Subject and Activity Features
View(HAR[c(1:3),]) # The first 3 records


## Ordering levels of subject feature owing to its disorder
HAR$subject <- factor(HAR$subject,levels = c("1", "2", "3", "4","5","6","7","8","9","10",
                                             "11","12","13","14","15","16","17","18","19",
                                             "20","21","22","23","24","25","26","27","28",
                                             "29","30"))

## Making illegal Feature Names Syntactically Valid
names(HAR) <- make.names(names(HAR))

# The control of Missing Values
sum(is.na(HAR))

# Frequency of Target Variable
table(HAR$Activity)
barplot(table(HAR$Activity),col= c("green","red","black","blue", "yellow","brown"),
        ylim=c(0,2000), cex.names=0.55, space=02)
grid()

# Frequency of 'Subject' Variable
# install.packages('ggplot2')
# install.packages('ggpubr')
library(ggplot2)
library(ggpubr)
theme_set(theme_pubr())

table(HAR$subject)
ggplot(HAR, aes(subject)) +
  geom_bar(fill = "#0073C2FF") +
  ylim(0,500) +
  theme_pubclean()

## To use the dataset for tools like Tableau, we need to export it by following code:
## 1st option:
# write.table(HAR, file = "HARtable.txt")

## 2nd option:
# install.packages("openxlsx")
# library(openxlsx)
# write.xlsx(HAR,"HAR.xlsx",sheetName = "Sheet1",col.names = TRUE,showNA=FALSE)

# To see normalization throughout the Range of scaling dataset
max(HAR[,1:561], na.rm = FALSE)
min(HAR[,1:561], na.rm = FALSE)

############################################################################################
##################################### Data Modelling #######################################
############################################################################################

############################################################################################
######################################Sampling 90:10########################################
############################################################################################
vect1=rep(0,5)
for(i in 1:5) {
  n=nrow(HAR) 
  indexes = sample(n,n*(90/100))
  trainset = HAR[indexes,]
  testset = HAR[-indexes,]
  
  svm_lin <- svm(Activity~ ., data=trainset, method='C-classification', kernel='linear')
  bnc <- naiveBayes(Activity~ ., data=trainset)
  ctree <- ctree(Activity ~ ., data=trainset)
  knn9 <- knn(train = trainset[,1:561], test = testset[,1:561], cl = trainset$Activity, k=9)
  rf <- randomForest(Activity ~ ., data = trainset)

  #test set predictions for linear
  pred_test_svm_lin <-predict(svm_lin,testset)
  p_lin= mean(pred_test_svm_lin==testset$Activity)
  
  #test set predictions for bnc
  pred_test_bnc <-predict(bnc,testset)
  p_bnc= mean(pred_test_bnc==testset$Activity)
  
  #test set predictions for decision trees
  pred_test_ctree <-predict(ctree, testset)
  p_ctree=mean(pred_test_ctree==testset$Activity)
  
  #test set predictions for kNN
  p_knn9= mean(knn9==testset$Activity)
  
  #test set predictions for Random Forest
  pred_test_rf <-predict(rf,testset)
  p_rf= mean(pred_test_rf==testset$Activity)
  
  p_vect=c(p_lin, p_bnc, p_ctree, p_knn9, p_rf)
  
  vect1 <- vect1 + (1/5)*p_vect
  
}
vect1

# The last dataset_ montecarlo 5
# [1] 0.9904854 0.7275728 0.9491262 0.9683495 0.9815534

############################################################################################
######################################Sampling 80:20########################################
############################################################################################
vect2=rep(0,5)
for(i in 1:5) {
  
n=nrow(HAR) 
indexes = sample(n,n*(80/100))
trainset = HAR[indexes,]
testset = HAR[-indexes,]

svm_lin <- svm(Activity~ ., data=trainset, method='C-classification', kernel='linear')
bnc <- naiveBayes(Activity~ ., data=trainset)
ctree <- ctree(Activity ~ ., data=trainset)
knn9 <- knn(train = trainset[,1:561], test = testset[,1:561], cl = trainset$Activity, k=9)
rf <- randomForest(Activity ~ ., data = trainset)

#test set predictions for linear
pred_test_svm_lin <-predict(svm_lin,testset)
#confusion_matrixlin= table(pred = pred_test, true = testset$activity)
p_lin= mean(pred_test_svm_lin==testset$Activity)

#test set predictions for bnc
pred_test_bnc <-predict(bnc,testset)
#confusion_matrixlin= table(pred = pred_test, true = testset$activity)
p_bnc= mean(pred_test_bnc==testset$Activity)

#test set predictions for decision trees
pred_test_ctree <-predict(ctree, testset)
#confusion_matrix_ctree= table(pred = pred_test_ctree, true = testset$activity)
p_ctree=mean(pred_test_ctree==testset$Activity)

#test set predictions for kNN
p_knn9= mean(knn9==testset$Activity)

#test set predictions for Random Forest
pred_test_rf <-predict(rf,testset)
#confusion_matrixlin= table(pred = pred_test, true = testset$activity)
p_rf= mean(pred_test_rf==testset$Activity)

p_vect=c(p_lin, p_bnc, p_ctree, p_knn9, p_rf)

vect2 <- vect2 + (1/5)*p_vect

}
vect2

# The last dataset_ montecarlo 5
# vect2
# [1] 0.9869903 0.7377670 0.9456311 0.9595146 0.9802913
# [1] 0.9870874 0.7341748 0.9422330 0.9634951 0.9830097

############################################################################################
######################################Sampling 70:30########################################
############################################################################################
vect3=rep(0,5)
for(i in 1:5) {
  
  n=nrow(HAR) 
  indexes = sample(n,n*(70/100))
  trainset = HAR[indexes,]
  testset = HAR[-indexes,]
  
  svm_lin <- svm(Activity~ ., data=trainset, method='C-classification', kernel='linear')
  bnc <- naiveBayes(Activity~ ., data=trainset)
  ctree <- ctree(Activity ~ ., data=trainset)
  knn9 <- knn(train = trainset[,1:561], test = testset[,1:561], cl = trainset$Activity, k=9)
  rf <- randomForest(Activity ~ ., data = trainset)
  
  #test set predictions for linear
  pred_test_svm_lin <-predict(svm_lin,testset)
  #confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_lin= mean(pred_test_svm_lin==testset$Activity)
  
  #test set predictions for bnc
  pred_test_bnc <-predict(bnc,testset)
  #confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_bnc= mean(pred_test_bnc==testset$Activity)
  
  #test set predictions for decision trees
  pred_test_ctree <-predict(ctree, testset)
  #confusion_matrix_ctree= table(pred = pred_test_ctree, true = testset$activity)
  p_ctree=mean(pred_test_ctree==testset$Activity)
  
  #test set predictions for kNN
  p_knn9= mean(knn9==testset$Activity)
  
  #test set predictions for Random Forest
  pred_test_rf <-predict(rf,testset)
  #confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_rf= mean(pred_test_rf==testset$Activity)
  
  p_vect=c(p_lin, p_bnc, p_ctree, p_knn9, p_rf)
  
  vect3 <- vect3 + (1/5)*p_vect
  
}
vect3

# The last dataset_ montecarlo 5
# vect3
# [1] 0.9886084 0.7489320 0.9417476 0.9623301 0.9791586

############################################################################################
######################################Sampling 60:40########################################
############################################################################################
vect4=rep(0,5)
for(i in 1:5) {
  
  n=nrow(HAR) 
  indexes = sample(n,n*(60/100))
  trainset = HAR[indexes,]
  testset = HAR[-indexes,]
  
  svm_lin <- svm(Activity~ ., data=trainset, method='C-classification', kernel='linear')
  bnc <- naiveBayes(Activity~ ., data=trainset)
  ctree <- ctree(Activity ~ ., data=trainset)
  knn9 <- knn(train = trainset[,1:561], test = testset[,1:561], cl = trainset$Activity, k=9)
  rf <- randomForest(Activity ~ ., data = trainset)
  
  #test set predictions for linear
  pred_test_svm_lin <-predict(svm_lin,testset)
  #confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_lin= mean(pred_test_svm_lin==testset$Activity)
  
  #test set predictions for bnc
  pred_test_bnc <-predict(bnc,testset)
  #confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_bnc= mean(pred_test_bnc==testset$Activity)
  
  #test set predictions for decision trees
  pred_test_ctree <-predict(ctree, testset)
  #confusion_matrix_ctree= table(pred = pred_test_ctree, true = testset$activity)
  p_ctree=mean(pred_test_ctree==testset$Activity)
  
  #test set predictions for kNN
  p_knn9= mean(knn9==testset$Activity)
  
  #test set predictions for Random Forest
  pred_test_rf <-predict(rf,testset)
  #confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_rf= mean(pred_test_rf==testset$Activity)
  
  p_vect=c(p_lin, p_bnc, p_ctree, p_knn9, p_rf)
  
  vect4 <- vect4 + (1/5)*p_vect
  
}
vect4

# The last dataset_ montecarlo 5
# vect4

# [1] 0.9868932 0.7367476 0.9350485 0.9562136 0.9778155
############################################################################################
############################################################################################
############################################################################################
Mat=cbind(vect1,vect2,vect3,vect4)

xdata <- c(60,70,80,90)
y1 <- Mat[1,]
y2 <- Mat[2,] 
y3 <- Mat[3,]
y4 <- Mat[4,] 
y5 <- Mat[5,]

# plot the first curve by calling plot() function
# First curve is plotted
plot(xdata, y1, xlab="Split Ratio", ylab="Probability of Correctness", type="o", col="blue",
     pch="o", lty=1,  ylim=c(0.5,1) )

points(xdata, y2, col="red", pch="*")
lines(xdata, y2, col="red",lty=2)

points(xdata, y3, col="green",pch="+")
lines(xdata, y3, col="green", lty=3)

points(xdata, y4, col="black",pch="*")
lines(xdata, y4, col="black", lty=3)


points(xdata, y5, col="purple",pch="o")
lines(xdata, y5, col="purple", lty=3)

legend(
  "bottomright", 
  lty=c(1,2,1,2), cex=0.55,
  col=c("blue", "red","green", "black", "purple"), 
  legend = c("Linear_SVM", "BNC", "DT", "KNN","RF")
)

############################################################################################
############################################################################################
#The Best Accuracies of algorithms belong to the 90:10 split ratio
#90:10 Split Ratio Confusion Matrix Analysis

library(caret)
caret::confusionMatrix(pred_test_svm_lin, testset$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_bnc, testset$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_ctree, testset$Activity, positive="1", mode="everything")
caret::confusionMatrix(knn9, testset$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_rf, testset$Activity, positive="1", mode="everything")



############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################


vect2 <- c(0.9869903,0.7377670, 0.9456311, 0.9595146, 0.9802913)

vect1 <- c(0.9904854, 0.7275728, 0.9491262, 0.9683495, 0.9815534)

remove(actual)




############################################################################################
#################################### Multinomial Logit Regression ##########################
############################################################################################

# I have to apply Multinomial rather than Logictic because my output variable activity
# is multinomial not binomial. However, I have still weights problem(too much predictors)
install.packages('e1071', dependencies = TRUE)
library(class) 
library(e1071) 
library("nnet")

HAR2=HAR
HAR2$activity <- relevel(HAR2$activity, ref=1)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

HAR_2 <- as.data.frame(lapply(HAR[2:562], normalize))

HAR2[,'train'] <- ifelse(runif(nrow(HAR2))<0.8,1,0)
#separate training and test sets
trainset <- HAR2[HAR2$train==1,]
testset <- HAR2[HAR2$train==0,]
dim(trainset)
trainset=trainset[,-564] 
testset=testset[,-564]

### trainset.multinom <- multinom(activity~., data=trainset)
## Getting Error:
# Error in nnet.default(X, Y, w, mask = mask, size = 0, skip = TRUE, softmax = TRUE,  : 
#                        too many (3384) weights

trainset.multinom <- multinom(activity~., data=trainset[,1:5])

summary(trainset.multinom)
zvalues <- summary(trainset.multinom)$coefficients / summary(trainset.multinom)$standard.errors
pnorm(abs(zvalues), lower.tail=FALSE)*2
???????????????????


############################### SVM for classification #################################### 
library(e1071) # load required library
set.seed(42) #set seed to ensure reproducible results

####Monte Carlo runs
#vect=rep(0,5)
#for(i in 1:100) {  
  
  #split into training and test sets
  n=nrow(HAR) 
  indexes = sample(n,n*(80/100))
  trainset = HAR[indexes,]
  testset = HAR[-indexes,]
  head(testset)
  
  #build model   linear kernel and C-classification (soft margin) with default cost (C=1)
  svm_lin <- svm(activity~ ., data=trainset, method='C-classification', kernel='linear')
  svm_pol <- svm(activity~ ., data=trainset, method='C-classification', kernel='polynomial')
  svm_sig <- svm(activity~ ., data=trainset, method='C-classification', kernel='sigmoid')
  svm_rad <- svm(activity~ ., data=trainset, method='C-classification', kernel='radial')
  bnc <- naiveBayes(activity~ ., data=trainset)
  
  #test set predictions for linear
  pred_test <-predict(svm_lin,testset)
  confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_lin= mean(pred_test==testset$activity)
  
  #test set predictions for polynomial
  pred_test <-predict(svm_pol,testset)
  confusion_matrixpoly= table(pred = pred_test, true = testset$activity)
  p_pol= mean(pred_test==testset$activity)
  
  #test set predictions for sigmoid
  pred_test <-predict(svm_sig,testset)
  confusion_matrixsig= table(pred = pred_test, true = testset$activity)
  p_sig=mean(pred_test==testset$activity)
  
  #test set predictions for radial
  pred_test <-predict(svm_rad,testset)
  confusion_matrixrad= table(pred = pred_test, true = testset$activity)
  p_rad=mean(pred_test==testset$activity)
  
  #test set predictions for bnc
  pred_test <-predict(bnc,testset)
  confusion_matrixlin= table(pred = pred_test, true = testset$activity)
  p_bnc= mean(pred_test==testset$activity)
  
  p_vect=c(p_lin, p_pol, p_sig, p_rad, p_bnc)
  #### Radial SVM in first runnýng was the best: 
  #### [1] 0.9626039 0.9542936 0.8337950 0.9639889 0.8227147
#  vect=vect+(1/100)*p_vect
#}

#vect

########################### Decision Trees ################################################
install.packages('party')
library("party")

n=nrow(HAR) 
indexes = sample(n,n*(80/100))
trainset = HAR[indexes,]
testset = HAR[-indexes,] 
# using ctree
HAR_ctree <- ctree(activity ~ ., data=trainset)
    
#test set predictions
pred_test_ctree <-predict(HAR_ctree, testset)
confusion_matrix_ctree= table(pred = pred_test_ctree, true = testset$activity)
p_ctree=mean(pred_test_ctree==testset$activity)
p_ctree

################################## Artificial Neural Network ##############################
## Fit neural network ############################################################
install.packages('neuralnet') # install library
library(neuralnet) # load library

dataANN <- na.omit(HAR)

# Random sampling
samplesize = 0.80 * nrow(dataANN)
set.seed(80)
index = sample( seq_len ( nrow ( dataANN ) ), size = samplesize )

# Create training and test set
datatrain = dataANN[ index, ]
datatest = dataANN[ -index, ]

#######
## Scale data for neural network ###### getting error here. 
#max = apply(dataANN , 2 , max)
#min = apply(dataANN, 2 , min)
#scaled = scale(dataANN, center = min, scale = max - min)

# creating training and test set
#trainNN = scaled[index , ]
#testNN = scaled[-index , ]
#head(trainNN)

# fit neural network
set.seed(2)
NN = neuralnet(activity ~ tBodyAcc.mean.X , data = datatrain, hidden = 3 , linear.output = T )

NN<-nnet(activity~., data=datatrain[,1:50], size=10) 
# plot neural network
plot(NN)
####################################################################################
## Prediction cyl using neural network in the mtcars 
predict_testNN = compute(NN, testNN[,c(4:6)])
predict_testNN = (predict_testNN$net.result * (max(data$cyl) - min(data$cyl))) + min(data$cyl)

plot(datatest$cyl, predict_testNN, col='blue', pch=16, ylab = "predicted cyl  NN", xlab = "real cyl")
abline(0,1)

# Calculate Root Mean Square Error (RMSE)
RMSE.NN = sqrt(sum((datatest$cyl - predict_testNN)^2) / nrow(datatest) )


###########################################################################################
################################## KNN ####################################################
###########################################################################################

str(HAR)
HAR = HAR[-1]
round(prop.table(table(HAR$activity)) * 100, digits = 1)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

HAR_n <- as.data.frame(lapply(HAR[2:562], normalize))

HAR_train <- HAR_n[1:2887,]
HAR_test <- HAR_n[2888:3609,]

HAR_train_labels <- HAR[1:2887, 1]
HAR_test_labels <- HAR[2888:3609, 1] 

# Fitting model
library(class)

HAR_test_pred8 <- knn(train = HAR_train, test = HAR_test,cl = HAR_train_labels, k=8)
HAR_test_pred9 <- knn(train = HAR_train, test = HAR_test,cl = HAR_train_labels, k=9)
HAR_test_pred10 <- knn(train = HAR_train, test = HAR_test,cl = HAR_train_labels, k=10)
HAR_test_pred11 <- knn(train = HAR_train, test = HAR_test,cl = HAR_train_labels, k=11)
HAR_test_pred12 <- knn(train = HAR_train, test = HAR_test,cl = HAR_train_labels, k=12)
HAR_test_pred60 <- knn(train = HAR_train, test = HAR_test,cl = HAR_train_labels, k=60)
# Evaluating the model performance
install.packages("gmodels")
library(gmodels)

CrossTable( x = HAR_test_labels, y = HAR_test_pred, prop.chisq = FALSE)

confusion_matrix8= table(pred = HAR_test_pred8, true = HAR_test_labels)
p_knn8= mean(HAR_test_pred8==HAR_test_labels)

confusion_matrix9= table(pred = HAR_test_pred9, true = HAR_test_labels)
p_knn9= mean(HAR_test_pred9==HAR_test_labels)

confusion_matrix10= table(pred = HAR_test_pred10, true = HAR_test_labels)
p_knn10= mean(HAR_test_pred10==HAR_test_labels)

confusion_matrix11= table(pred = HAR_test_pred11, true = HAR_test_labels)
p_knn11= mean(HAR_test_pred11==HAR_test_labels)

confusion_matrix12= table(pred = HAR_test_pred12, true = HAR_test_labels)
p_knn12= mean(HAR_test_pred12==HAR_test_labels)

confusion_matrix60= table(pred = HAR_test_pred60, true = HAR_test_labels)
p_knn60= mean(HAR_test_pred60==HAR_test_labels)

v <-c(p_knn8 ,p_knn9, p_knn10, p_knn11, p_knn12, p_knn60)
v ## the best p_knn9 : 0.9016620499

###########################################################################################
################################## Random Forest ##########################################
###########################################################################################

library(party)
library(randomForest)


n=nrow(HAR) 
indexes = sample(n,n*(80/100))
trainset = HAR[indexes,]
testset = HAR[-indexes,]
head(testset)

# Create the forest.
output.forest <- randomForest(activity ~ ., data = trainset)

# View the forest results.
print(output.forest) 

# Importance of each predictor.
print(importance(output.forest,type = 2)) 


pred_test <-predict(output.forest,testset)
#confusion_matrixlin= table(pred = pred_test, true = testset$activity)
p_random= mean(pred_test==testset$activity)


###########################################################################################

library(parallel)
detectCores()


PCA_comps <- prcomp ( HAR [,-c (562 ,563) ], scale.= TRUE )
summary ( PCA_comps )
dim ( PCA_comps$x)
PCA_comps$rotation [1:5 ,1:5]
PCA_var <- PCA_comps$sdev^2
head ( PCA_var, 20)
PCA_Prop_Var <- PCA_var/ ncol ( HAR [,-c (562 ,563) ])
head ( PCA_Prop_Var, 10) * 100
sum ( head ( PCA_Prop_Var, 100) * 100)

plot ( PCA_Prop_Var [1:100] , xlab = "PC",ylab = " Proportion ")
plot ( cumsum ( PCA_Prop_Var ), xlab = " Principal Components ",ylab = " Proportion of Variance captured ")
abline (h =0.95)
abline (v =100)

PCA_HARdata <- data.frame ( HAR$Activity, PCA_comps$x[, 1:100])
colnames ( PCA_HARdata ) [1] <- 'Activity'

##################################

# ######## Dimensionality Reduction ##############
# Method 1
# Correlated features
library ( caret )
FullCorrelationMatrix <- cor ( sapply ( HAR [,-c(562 ,563) ], as.numeric ))
# write . csv( FullCorrelationMatrix , file = " CorrMat .csv ")
plot ( FullCorrelationMatrix )
dim ( FullCorrelationMatrix )
table ( round ( FullCorrelationMatrix , digits = 1))


HighCorrelatedList <- findCorrelation (na.omit ( FullCorrelationMatrix ), cutoff =0.8 , names = T, exact = T)
length ( HighCorrelatedList )
Uncorrelated <- HAR [, ! colnames ( HAR ) %in% HighCorrelatedList ]
Uncorrelated <- subset ( Uncorrelated , select = -c( subject ))
dim ( Uncorrelated )
rm( FullCorrelationMatrix, HighCorrelatedList, HighCorrelatedList1 ,
    HighCorrelatedList2 , HighCorrelatedList3 )
