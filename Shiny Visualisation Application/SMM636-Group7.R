# load important libraries
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(plyr)
library(ggplot2)
library(ggpubr)
library(rpart)
library(rpart.plot)
library(rattle)
library(dplyr)

############ set up functions ############
# load the data function and clean the data
load_data = function(file) {
  if (file == 'heart') {
    # load the data
    df = read.csv('Heart.csv', header = T)
    # remove NAs rows
    df = df[complete.cases(df),]
    # make the factor features factors, if they are not already, and named them appropriately
    df$Sex = as.factor(df$Sex)
    df$Sex = revalue(df$Sex, c("0"="female", "1"="male"))
    df$Slope = as.factor(df$Slope)
    df$Slope = revalue(df$Slope, c("1"="Upsloping", "2"="Flat", "3" = "Downsloping"))
    df$Ca = as.factor(df$Ca)
    df$Fbs = as.factor(df$Fbs)
    df$Fbs = revalue(df$Fbs, c("0"="False", "1"="True"))
    df$ExAng = as.factor(df$ExAng)
    df$ExAng = revalue(df$ExAng, c("0"="No", "1"="Yes"))
    df$RestECG = as.factor(df$RestECG)
    df$RestECG = revalue(df$RestECG, c("0"="Normal", "1"="Medium", "2" = "Severe")) 
    df$Thal = as.factor(df$Thal) 
    df$ChestPain = as.factor(df$ChestPain)
    df$AHD = as.factor(df$AHD)
   }
  else if (file == 'titanic') {
    # load the data
    df = read.csv('titanic.csv', header = T)
    # make the factor features factors, if they are not already, and named them appropriately
    drop <- c("Name")
    df = df[ , !(names(df) %in% drop)]
    df$Survived = as.factor(df$Survived)
    df$Survived = revalue(df$Survived, c("0"="No", "1"="Yes"))
    df$Pclass = as.factor(df$Pclass)
    df$Pclass = revalue(df$Pclass, c("1"="Upper Class", "2"="Middle Class", "3" = "Lower Class"))
    }
  else {
    # if there is no such data print:
    print(paste("There is no such data as", name))
    break
  }
  return(df)
}

# load the respective label function
load_label = function(file) {
  if (file == 'heart') {
    label = 'AHD'  
  }
  else if(file == 'titanic') {
    label = 'Survived'
  }
  return(label)
}

# split the data function
split_ratio = function(data,prop) {
  set.seed(42)
  train.index=createDataPartition(data[ , ncol(data)],p=prop,list=FALSE)
  train=data[train.index,]
  test=data[-train.index,]
  return(list(train=train,test=test))
  }

# get tree equation function
tree_func = function(train,minsplit, cp, label){
  set.seed(42)
  rpart.control = rpart.control(minsplit = minsplit, cp = cp)
  model = as.formula(paste(label,'.',sep=' ~ '))
  tree.rpart = rpart(model, train, method = "class", control = rpart.control)
    
  return(tree.rpart)
}

# function to plot importance of variables for decision trees
plot_importance = function(tree, threshold = 14){
  # change to dataframe format
  to_plot = data.frame(tree$variable.importance)
  # clean the data, as to make it plotable
  to_plot <- cbind(Variable = rownames(to_plot), to_plot)
  rownames(to_plot) <- 1:nrow(to_plot)
  names(to_plot)[names(to_plot) == "tree.variable.importance"] <- "Importance_Score"
  to_plot <- to_plot %>%  
    mutate(mycolor = ifelse(to_plot$Importance_Score > threshold , "type2", "type1"))
  return(ggdotchart(to_plot, y = "Importance_Score", x= "Variable", sorting = "descending", rotate = T,
                    add = "segments", color = "mycolor", dot.size = 6, 
                    title = "Importance of Variables") +
           theme_light() +
           theme(
             legend.position = "none",
             panel.border = element_blank()) + xlab('Variables') + ylab('Mean Decrease Gini'))
}

# get tree accuracy for the data for decision trees
tree_acc = function(tree, data, label){
  set.seed(42)
  pred = predict(tree, data[ , !(names(data) %in% label)], type = 'class')
  acc = mean(pred==data[,label]) 
  return(acc)
}

# get random forest function
rf_func = function(train, label, ncol, ntree = 500){
  set.seed(42)
  model = as.formula(paste(label,'.',sep=' ~ '))
  rf = randomForest(model, data = train, mtry = round(sqrt(ncol)),
                           importance = TRUE, ntree = ntree) 
  return(rf)
}

# get random forest accuracies
rf_acc = function(rf, data, label){
  set.seed(42)
  pred.rf=predict(rf,data[ , !(names(data) %in% label)])
  acc = mean(pred.rf==data[,label])
  
  return(acc)
}

# get sensitivity and specificity function from random forest
get_class_error = function(rf, data, label){
  set.seed(42)
  pred.rf=predict(rf,data[ , !(names(data) %in% label)])
  to_use = confusionMatrix(pred.rf, data[, label])
  to_use = data.frame(to_use$byClass[1:2])
  sensitivity = to_use[1,1]
  specificity = to_use[2,1]
  return(list(sensitivity = sensitivity, specificity = specificity))
  }

# create mean decrease accuracy plot
rf_acc_plot = function(rf){
  to_plot = data.frame(rf$importance)
  to_plot = to_plot[, 3:4]
  mean_acc = mean(to_plot[, 1])
  mean_gini = mean(to_plot[,2])
  to_plot <- cbind(Variable = rownames(to_plot), to_plot)
  rownames(to_plot) <- 1:nrow(to_plot)
  to_plot <- to_plot %>%  
    mutate(mycolor = ifelse(to_plot$MeanDecreaseAccuracy > mean_acc & to_plot$MeanDecreaseGini > mean_gini, "type2", "type1"))
  return(ggdotchart(to_plot, y = "MeanDecreaseAccuracy", x= "Variable", sorting = "descending", rotate = T,
                    add = "segments", color = "mycolor", dot.size = 6, 
                    title = "Importance of Variables") +
           theme_light() +
           theme(
             legend.position = "none",
             panel.border = element_blank()) + xlab('Variables') + ylab('Mean Decrease Accuracy'))
}

# create mean decrease gini plot
rf_gini_plot = function(rf){
  to_plot = data.frame(rf$importance)
  to_plot = to_plot[, 3:4]
  mean_acc = mean(to_plot[, 1])
  mean_gini = mean(to_plot[,2])
  to_plot <- cbind(Variable = rownames(to_plot), to_plot)
  rownames(to_plot) <- 1:nrow(to_plot)
  to_plot <- to_plot %>%  
    mutate(mycolor = ifelse(to_plot$MeanDecreaseAccuracy > mean_acc & to_plot$MeanDecreaseGini > mean_gini, "type2", "type1"))
  return(ggdotchart(to_plot, y = "MeanDecreaseGini", x= "Variable", sorting = "descending", rotate = T,
                    add = "segments", color = "mycolor", dot.size = 6) +
           theme_light() +
           theme(
             legend.position = "none",
             panel.border = element_blank()) + xlab('Variables') + ylab('Mean Decrease Gini'))
}

############## shiny #############
library(shiny)

#######################################################################################
######################################## UI ##########################################
######################################################################################
ui = fluidPage(
  theme = shinythemes::shinytheme('journal'),
  navbarPage(
    "Machine Learning Models",
    tabPanel("Data",
             sidebarPanel(
               radioButtons("df_1", "Datasets:", 
                            choices = c("heart", "titanic")),
               radioButtons("dataset_1", "Display:", 
                            choices = list("all","train", "test"), inline = T),
               sliderInput("ratio_1", "Training/Test Ratio:", min = 0, max = 1, value = 0.7),
               actionButton("action2", "Reset", class = "btn-primary")
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Context",
                          br(),
                          h2("Heart Disease Dataset"),
                          "This dataset is provided by the University of California Irvine (UCI), named the 'Cleveland
                          dataset'. The common purpose and task of this dataset is to predict, based on the given attributes
                          of a patient, whether they have a heart disease or not using classification tools, such as
                          decision trees. After, predictions can be made based on the machine learning algorithm.
                          More information with regards to the attributes of the dataset, summary statistics and decision
                          tree visualisations can be found by navigating the tabs.",
                          br(),
                          br(),
                          h2("Titanic Dataset"),
                          "The RMS Titanic was a British-made passenger ship which sank in the North Atlantic
                          Ocean in 1912. With an estimated amount of 2,224 passengers and crew aboard the ship,
                          a record 1,500 people died from the tragedy, making it one of the world's deadliest disasters.
                          The specific task the dataset is used for is to conduct classification, using decision trees,
                          to observe the characteristics of people who did and did not survive the disaster and, based
                          on this, make predictions. More information with regards to the attributes of the dataset,
                          summary statistics and decision tree visualisations can be found by navigating the tabs."),
                 tabPanel("Dataset",
                          DT::dataTableOutput("dataset")),
                 tabPanel("Variables",
                          h3("Type"),
                          verbatimTextOutput("str"),
                          h3("Description"),
                          htmlOutput("desc")),
                 tabPanel("Summary Statistics",
                          verbatimTextOutput("summary"))
               )
             )
    ),
    tabPanel("Decision Tree",   
             sidebarPanel(
               radioButtons("df", "Datasets:", 
                            choices = c("heart", "titanic")),
               sliderInput("ratio", "Training/Test Ratio:", min = 0, max = 1, value = 0.7),
               sliderInput("minsplit", "Minimum Split:", min = 0, max = 100, value = 20),
               sliderInput("cp", "Complexity:", min = -0.5, max = 0.5, value = 0, 0.01),
               h4("Dependent Variable:"),
               verbatimTextOutput("pred_1"),
               h4("Training Accuracy:"),
               verbatimTextOutput("train_desc"),
               h4("Test Accuracy:"),
               verbatimTextOutput("test_desc"),
               actionButton("action3", "Reset", class = "btn-primary"),
               actionButton("additional_acc1", "Additional Information")
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Tree Visualisation", 
               plotOutput('plot')),
               tabPanel("Variable Importance",
               plotOutput('plot_2'),
               sliderInput("color_filter","Threshold", min = 0, max = 25, value = 14))
              ) 
             )
             ),
    tabPanel("Random Forest",
             sidebarPanel(
               radioButtons("df_3", "Datasets:", 
                            choices = c("heart", "titanic")),
               sliderInput("ratio_3", "Training/Test Ratio:", min = 0, max = 1, value = 0.7),
               sliderInput("mtry", "No. of Predictors for Each Split:" , min = 1, max = 6, value = 4),
               sliderInput("ntree", "No. of Trees Grown:" , min = 10, max = 1000, value = 200),
               h4("Dependent Variable:"),
               verbatimTextOutput("pred_2"),
               h4("Training Accuracy:"),
               verbatimTextOutput("train_desc_3"),
               h4("Test Accuracy:"),
               verbatimTextOutput("test_desc_3"),
               actionButton("action4", "Reset", class = "btn-primary"),
               actionButton("additional_acc2", "Additional Information")
             ),
             mainPanel(
               plotOutput('rf_plot'),
               plotOutput("rf_plot_2")
                       )
             )
  )
)

#######################################################################################
######################################## Server #######################################
######################################################################################
server = function(input, output, session) {
  ### Decision trees
  # get the data
  get_df = reactive({
    df = load_data(input$df)
  })
 # get the label
 get_label = reactive({
   label = load_label(input$df)
 })
 # get the training dataset
 train_df = reactive({
   train = split_ratio(get_df(), input$ratio)$train
 })
 # get the test dataset
 test_df = reactive({
   test = split_ratio(get_df(), input$ratio)$test
 })
 # get the decision tree analysis
 get_tree = reactive({
   tree = tree_func(train_df(),input$minsplit, input$cp, get_label())
 })
 ### Random forest
 # get the data
 get_df_rf = reactive({
   df = load_data(input$df_3)
 })
 # get the label
 get_label_rf = reactive({
   label = load_label(input$df_3)
 })
 # get the training dataset
 train_df_rf = reactive({
   train = split_ratio(get_df_rf(), input$ratio_3)$train
 })
 # get the test dataset
 test_df_rf = reactive({
   test = split_ratio(get_df_rf(), input$ratio_3)$test
 })
 # get the random forest analysis
 get_rf = reactive({
   rf = rf_func(train_df_rf(), get_label_rf(), input$mtry , input$ntree)  
 })
 # get the training sensitivity value
 get_train_sensitivity = reactive({
   sensitivity = get_class_error(get_rf(),train_df_rf(),get_label_rf() )$sensitivity 
 })
 # get the training specificity value
 get_train_specificity = reactive({
   sensitivity = get_class_error(get_rf(),train_df_rf(),get_label_rf() )$specificity 
 })
 # get the test sensitivity value
 get_test_sensitivity = reactive({
   sensitivity = get_class_error(get_rf(),test_df_rf(),get_label_rf ())$sensitivity 
 })
 # get the test specificity value
 get_test_specificity = reactive({
   sensitivity = get_class_error(get_rf(),test_df_rf(),get_label_rf())$specificity 
 })
 ## plots, text, tables and additional features
 # descriptions
 output$train_desc_3 = renderText({
  paste(round(rf_acc(get_rf(), train_df_rf(),get_label_rf()), 2))
 })
 output$test_desc_3 = renderText({
  paste(round(rf_acc(get_rf(), test_df_rf(),get_label_rf()), 2))
 })
 # plots
 output$plot = renderPlot({
   fancyRpartPlot(get_tree(), palettes=c("Oranges", "Greens")) 
 })
 output$plot_2 = renderPlot({
   plot_importance(get_tree(), input$color_filter)
 })
 # set predictor
 output$pred_1 = renderText({
   if(input$df == "heart"){
     paste("AHD")
   }
   else{
     paste("Survived")
    }})
 output$pred_2 = renderText({
  if(input$df_3 == "heart"){
    paste("AHD")
  }
  else{
    paste("Survived")
  }
})
 # additional information pop-ups
 observeEvent(input$additional_acc1, {
  showModal(modalDialog(
    title = "Additional Information",
    h3("Explanation:"),
    p("The tree visualisation tab shows the branches and leaves produced from predicting the target variable by
       learning decision rules based on the training data. A simple example would be in the case of using the Heart
       Disease dataset with the default values: suppose if the patient had a normal blood flow with the injection of thallium
       ('thal' = 1), and the number of major vessels colored by flourosopy did not equate to 1 (Ca = 2, 3), this would mean
       that the patient did not have a heart disease. Based on the inputs of 'thal' and 'Ca', it is possible to follow the tree
       to the leaves, which gives the final yes or no value."),
    p("Users can adjust the Mean Decrease in Gini threshold to be able to specifically see which variables are most important 
       according to the user’s subjectivity and make the visualisation more interpretable."),
    h5("Note: please see the 'Additional Information' in the 'Random Forest' tab for details regarding the variable importance."),
    easyClose = T,
    footer = NULL
   ))
 })
 observeEvent(input$additional_acc2, {
  showModal(modalDialog(
    title = "Additional Information",
    h3("Explanation:"),
    p("The first plot portrays the Mean Decrease Accuracy which examines the accuracy lost by excluding each variable.
       For example, given the default parameters for the Titanic dataset, 'Sex' has the highest Mean Decrease
       Accuracy which suggests that this variable is important for estimating the value of the target variable and successful
       classification. Similarly, the Mean Decrease in Gini measures the mean of a variable's decrease in node purity which, again,
       based on the example above, shows that the variable 'Sex' is important for estimating the value of the target variable."),
    h5("Note: The blue colour code is based on the importance of a variable in BOTH plots."),
    p("For example, if the No. of Predictors for Each Split is reduced to a value of 2 for the Heart Disease dataset, it can be
       observed that the variable 'Slope' is the 4th highest variable of importance in the Mean Decrease Accuracy plot, but it is
       below the general threshold for the Mean Decrease in Gini plot."),
    br(),
    h3("Sensitivity/Specificity Results:"),
    h4("Training Set"),
    p("Sensitivity: ", round(get_train_sensitivity(), 2)),
    p("Specificity: ", round(get_train_specificity(), 2)),
    h4("Test Set"),
    p("Sensitivity: ", round(get_test_sensitivity(), 2)),
    p("Specificity: ", round(get_test_specificity(), 2)),
    easyClose = T,
    footer = NULL 
   ))
 })
 # train and test data descriptions
 output$train_desc = renderText({
   paste(round(tree_acc(get_tree(), train_df(),get_label()), 2))
 })
 output$test_desc = renderText({
   paste(round(tree_acc(get_tree(), test_df(),get_label() ), 2))
 })
 # display train and test sets
 display_train = reactive({
   split_ratio(load_data(input$df_1), input$ratio_1)$train
 })
 display_test = reactive({
   split_ratio(load_data(input$df_1), input$ratio_1)$test
 })
 
 ## dataset - 1st tab
 output$dataset = DT::renderDataTable({
   if (input$df_1 == "heart"){
     if (input$dataset_1 == "train"){
       display_train()
       }
     else if (input$dataset_1 == "test"){
       display_test()
     }
     else if (input$dataset_1 == "all"){
       load_data(input$df_1)
     }
    }
   else if (input$df_1 == "titanic"){
     if (input$dataset_1 == "train"){
       display_train()
       }
     else if (input$dataset_1 == "test"){
       display_test()
     }
     else if (input$dataset_1 == "all"){
       load_data(input$df_1)
     }
   }
 })
 ## summary statistics - 2nd tab
 output$summary = renderPrint({
   if (input$df_1 == "heart"){
     if (input$dataset_1 == "train"){
       summary(display_train())
     }
     else if (input$dataset_1 == "test"){
       summary(display_test())
     }
     else if (input$dataset_1 == "all"){
       summary(load_data(input$df_1))
     }
   }
   else if (input$df_1 == "titanic"){
     if (input$dataset_1 == "train"){
       summary(display_train())
     }
     else if (input$dataset_1 == "test"){
       summary(display_test())
     }
     else if (input$dataset_1 == "all"){
       summary(load_data(input$df_1))
     }
   }
 })
 ## variables - 3rd tab
 output$str = renderPrint({
   if (input$df_1 == "heart"){
     if (input$dataset_1 == "train"){
       str(display_train())
     }
     else if (input$dataset_1 == "test"){
       str(display_test())
     }
     else if (input$dataset_1 == "all"){
       str(load_data(input$df_1))
     }
   }
   else if (input$df_1 == "titanic"){
     if (input$dataset_1 == "train"){
       str(display_train())
     }
     else if (input$dataset_1 == "test"){
       str(display_test())
     }
     else if (input$dataset_1 == "all"){
       str(load_data(input$df_1))
     }
   }
 })
 
 # reset button 1
 observeEvent(input$action2, {
   updateRadioButtons(session, "dataset_1", "Display:", 
                      choices = list("all","train", "test"), inline = T)
   updateSliderInput(session,"ratio_1", "Training/Test Ratio:", min = 0, max = 1, value = 0.7)
 })
 # reset button 2
 observeEvent(input$action3, {
   updateSliderInput(session, "ratio", "Training/Test Ratio:", min = 0, max = 1, value = 0.7)
   updateSliderInput(session, "minsplit", "Minimum Split:", min = 0, max = 100, value = 20)
   updateSliderInput(session, "cp", "Complexity:", min = -0.5, max = 0.5, value = 0)
   updateSliderInput(session, "color_filter","Threshold", min = 0, max = 25, value = 14)
 })
 # reset button 3
 observeEvent(input$action4, {
   updateSliderInput(session,"ratio_3", "Training/Test Ratio:", min = 0, max = 1, value = 0.7)
   updateSliderInput(session,"mtry", "No. of Predictors for Each Split:", min = 1, max = 6, value = 4)
   updateSliderInput(session,"ntree", "No. of Trees Grown:", min = 10, max = 1000, value = 200)
 })
 
 ## random forest plots
 output$rf_plot = renderPlot({
    rf_acc_plot(get_rf())
 })
 output$rf_plot_2 = renderPlot({
   rf_gini_plot(get_rf())
 })
 ## variable descriptios
 output$desc = renderUI({
   if (input$df_1 == "heart"){
     HTML(paste("- Age: age in years ",
                "- Sex: gender (1 = female, 2 = male)",
                "- ChestPain: chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)",
                "- RestBP: resting blood pressure",
                "- Chol: serum cholestoral in mg/dl",
                "- Fbs: fasting blood sugar > 120 mg/dl (1 = false, 2 = true)",
                "- RestECG: resting electrocardiographic results",
                "- MaxHR: maximum heart rate achieved",
                "- ExAng: exercise induced angina (1 = no; 2 = yes)",
                "- Oldpeak: ST depression induced by exercise relative to rest",
                "- Slope: the slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)",
                "- Ca: number of major vessels (0-3) colored by flourosopy",
                "- Thal: any effects after injection of thallium (1 = normal, 2 = fixed defect, 3 = reversable defect)",
                "- AHD: acquired heart disease (1 = no, 2 = yes)",
                sep = "<br/>"))
   }
   else if (input$df_1 == "titanic"){
     HTML(paste("- Survived: survival (1 = no, 2 = yes)",
                "- Pclass: ticket class (1 = upper class, 2 = middle class, 3 = lower class)",
                "- Sex: gender",
                "- Age: age in years",
                "- Siblings.Spouses.Aboard: number of siblings/spouses aboard the Titanic",
                "- Parents.Children.Aboard: number of parents/children aboard the Titanic",
                "- Fare: passenger fare",
                sep = "<br/>"))
   }
 })
 
}

shinyApp(ui, server)


