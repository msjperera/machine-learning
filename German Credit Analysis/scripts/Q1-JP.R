# load relevant libraries
library(caret)
library(rattle)
library(tree)
library(randomForest)
library(pROC)
library(ROCR)
library(party)
library(rpart)

###############################    Part 1    ###################################

# load german data
data(GermanCredit)

# delete 2 variables where values repeat for both classes
GermanCredit[, c("Purpose.Vacation", "Personal.Female.Single")] = list(NULL)

# check for imbalances in responses/classes
summary(GermanCredit$Class)

# get training and test sets
set.seed(5)
train.index = createDataPartition(GermanCredit$Class, p = 0.7, list = FALSE, times = 1)
train = GermanCredit[train.index,]
test = GermanCredit[-train.index,]

# create decision tree
german.tree = tree(Class ~ ., train)
summary(german.tree)
german.tree

# plot decision tree
plot(german.tree)
text(german.tree, cex = 0.8)

# decide optimal number of trees
set.seed(5)
german.cv = cv.tree(german.tree, FUN = prune.misclass, K = 10)
german.cv

par(mfrow = c(1, 2))
plot(german.cv$size, german.cv$dev, type = "b",
     xlab = "number of leaves of the tree", ylab = "CV error rate%",
     cex.lab = 1, cex.axis = 1, pch = 20) # no. of leaves with smallest CV error rate = 4
plot(german.cv$k, german.cv$dev, type = "b",
     xlab = expression(alpha), ylab = "CV error rate%",
     cex.lab = 1, cex.axis = 1, pch = 20)

# prune the tree
set.seed(5)
german.prune = prune.misclass(german.tree, best = 4) # added parameter with optimal leaves = 4

# plot pruned tree
par(mfrow = c(1, 1))
plot(german.prune)
text(german.prune, pretty = 1, cex = 0.8)

# summary
summary(german.prune)
german.prune

# get test error rate / mean accuracy
pred = predict(german.prune, test[,-10], type = "class")
table(pred, test[,10])
dt.macc = mean(pred == test[,10])
dt.macc  # mean accuracy = 0.6966667
dt.terr = 1 - dt.macc 
dt.terr # test error rate = 0.3033333

###############################    Part 2    ###################################

# set up train control
fitControl = trainControl(
  method = "repeatedcv",
  number = 10, # 10-fold cross-validation
  repeats = 3)

# produce random forest model
set.seed(5)
rf.fit = train(Class ~., data = train, method = "rf", metric = "Accuracy",
               trControl = fitControl, tuneLength = 5)
rf.fit # obtain optimal mtry = 16
rf.fit$finalModel

# plot random forest model
par(mfrow = c(1, 1))
plot(rf.fit)

# produce random forest model using optimal mtry
set.seed(5)
german.rf = randomForest(Class ~., data = train, mtry = 16, importance = TRUE)
german.rf

# get test error rate / mean accuracy
pred = predict(german.rf, test[,-10], type = "class")
table(pred, test[,10])
rf.macc = mean(pred == test[,10]) 
rf.macc # mean accuracy = 0.75
rf.terr = 1 - rf.macc 
rf.terr # test error rate = 0.25

# get variable importance plot
importance(german.rf)
varImpPlot(german.rf, cex = 0.8, main = "Variable Importance Plots")

#############################    Part 3    #################################

# extract decision tree model in Part 1
dt.tree = rpart(german.prune)
dt.pred = predict(dt.tree, type = "prob", newdata = test)[,2]
dt.roc = roc(test$Class, dt.pred)
dt.roc # AUC = 0.6781

# extract random forest model in Part 2
rf.pred = predict(german.rf, type = "prob", newdata = test)[,2]
rf.roc = roc(test$Class, rf.pred)
rf.roc # AUC = 0.7623

# plot ROC curves
par(mfrow = c(1, 1))
plot(rf.roc, main = "ROC Curves", col = "blue4")
lines(dt.roc, col = "firebrick")
legend("bottomright", col = c("blue4", "firebrick"), pch = 20, pt.cex = 1,
       cex = 0.9, legend = c("Random Forest Model", "Decision Tree Model"))
