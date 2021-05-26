# load relevant libraries
library(caret)
library(tidyverse)
library(ggpubr)
library(e1071)
library(ggplot2)
library(pROC)

# simulate non-linearly separable data: three-class dataset with 50 obs in each class + 2 features
set.seed(42)
class1 = matrix(rnorm(n = 50 *2, mean = 0, sd = 0.2), ncol = 2)
class2 = matrix(rnorm(n = 50 *2, mean = 0, sd = 0.3), ncol = 2) * 4
class3 = matrix(rnorm(n = 50 *2, mean = 0, sd = 0.45), ncol = 2) * 9

# generate labels
lab = factor(1:3)
lab = rep(lab, each = 50) # 50 observations per label/class

# combine data into one data frame
df = as.data.frame(rbind(class1, class2, class3))
lab = as.data.frame(lab)
df = cbind(df, lab)
df

# rename columns
colnames(df)
names(df)[names(df) == "V1"] = "x1"
names(df)[names(df) == "V2"] = "x2"
names(df)[names(df) == "lab"] = "Class"
df

# plot simulated data
require(ggplot2)
require(reshape2)
ggplot(df, aes(x = x1, y = x2, colour = Class), ) +
  geom_point(size = 3, alpha = 0.5) +
  ggtitle("Simulated Data Scatter Plot")

# get training and test sets
set.seed(42)
train.index = createDataPartition(df$Class, p = 0.5, list = FALSE, times = 1)
train = df[train.index,]
test = df[-train.index, -3]

#############################    RBF Kernel    #################################

# set train control
fitControl = trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3)

# set normal radial function
set.seed(42)
svm.Radial = train(Class ~., data = train, method = "svmRadial",
                   trControl = fitControl,
                   preProcess = c("center", "scale"),
                   tuneLength = 5)
svm.Radial

# plot radial function
plot(svm.Radial)

# tune both parameters C and sigma
grid.Radial = expand.grid(sigma = c(0.01, 0.1, 1, 3, 10),
                         C = c(0.01, 0.1, 0.25,1, 10))

# set new radial function
set.seed(42)
svm.Radialg = train(Class ~., data = train, method = "svmRadial",
                    trControl = fitControl,
                    preProcess = c("center", "scale"),
                    tuneGrid = grid.Radial)
svm.Radialg # sigma = 3 and C = 1 

# plot new radial function with tuned parameters
plot(svm.Radialg)

# get test error / mean accuracy
pred.Radialg = predict(svm.Radialg, train[,-3])
radial.macc = mean(pred.Radialg == train[,3])
radial.macc # mean accuracy = 0.8533333
radial.terr = 1 - radial.macc
radial.terr # test error = 0.1466667

#########################    Polynomial Kernel    ##############################

# set polynomial function
svm.Polynomial = train(Class ~., data = train, method = "svmPoly",
                       trControl = fitControl,
                       preProcess = c("center", "scale"),
                       tunelength = 5)
svm.Polynomial # polynomial degree = 3, scale = 0.1 and C = 1

# plot polynomial function
plot(svm.Polynomial)

# get test error / mean accuracy
pred.Polynomial = predict(svm.Polynomial, train[,-3])
poly.macc = mean(pred.Polynomial == train[,3])
poly.macc # mean accuracy = 0.6933333
poly.terr = 1 - poly.macc
poly.terr # test error = 0.3066667

##################    Support Vector Classifier (SVC)    #######################

# tune parameter C
grid.SVC = expand.grid(C = c(0.01, 0.1, 1, 10, 100))

# set SVC
svm.SVC = train(Class ~., train, method = "svmLinear",
                trControl = fitControl,
                preProcess = c("center", "scale"),
                tuneGrid = grid.SVC)
svm.SVC # C = 10

# plot SVC
plot(svm.SVC)

# get test error / mean accuracy
pred.SVC = predict(svm.SVC, train[,-3])
svc.macc =  mean(pred.SVC == train[,3])
svc.macc # mean accuracy = 0.5333333
svc.terr = 1 - svc.macc
svc.terr # test error = 0.4666667

