# The purpose of this document is to create a fine tuned SVR model

# file paths
scripts <- 'C:/Users/tommy/Google Drive/Coursework/1machine_learning/scripts/'
data <- "C:/Users/tommy/Google Drive/Coursework/1machine_learning/data/"
raw <- "C:/Users/tommy/Google Drive/Coursework/1machine_learning/data/raw/"

# custom fuctions
source(paste0(scripts,'helper_functions.R'))

# load or install packages
LoadPackages(c("dplyr", "ggplot2", "caret", "tibble", 'Boruta', 'stringr',
               'kernlab'))

# load data
load(paste0(raw,"MLProjectData.RData"))
train_data <-  MLProjectData %>% mutate(set = 'training')
submit_data <- test.data %>% mutate(set = "submission")
all_data <- bind_rows(train_data, submit_data)

# create new binary variable
all_data <- all_data %>% 
  mutate(imp_cat = ifelse( 
    cat16 == T |
      cat6 == T |
      cat19 == T |
      cat11 == T |
      cat21 == T |
      cat9 == T, 1, 0 ))

# select variables for model
select_data <- all_data %>% 
  dplyr::select('set', "target", "imp_cat", 'num6', 'num5', 
                'num58', 'num4', 'num23', 'num32', "num18",
                "num1", "num3")

# separate the submission data from the rest
submit_data <- select_data %>% filter(set == "submission")

# separate the non submission data
select_data2 <- select_data %>% 
  dplyr::filter(set != "submission") %>% 
  dplyr::mutate(id = row_number()) %>%
  dplyr::select(id, everything())

# split to train and test sets
set.seed(888)
train <- select_data2  %>% sample_frac(.8) # random 80% for training
test <- anti_join(select_data2, train, by = "id") # the remaining for test
train <- train %>% select(-id, -set) # remove the id and set cols

#  create list with indexes for each cv fold
#  makes cross validation more comparable across models
set.seed(888)
cv_folds <- createFolds(train$target, k = 5, list = TRUE, returnTrain = TRUE)

# object to control model tuning 
ctrl <- trainControl(method = "cv", index = cv_folds, savePredictions = 'final',
                     verboseIter = T)

# tuning grid for svr
# hyperparameters chosen using "grad student" descent
set.seed(888)
svm_grid2 <- expand.grid(C = 2^seq(-2.5, -1.5, length = 15), 
                         sigma = seq(.026, .028, length = 10))

# train svr model
svm2 <- train(target ~ ., data = train, tuneGrid = svm_grid2,
              metric = "MAE", method = "svmRadial", 
              preProcess = c("center", "scale"), trControl = ctrl)

# cv results for MAE
svm2$results %>% arrange(MAE) %>% head()

# make predictions on test set
pred <- predict(svm2, test)

# MAE for test set
MAE(pred, test$target)

################################################################################
# Train final svm model with train and test combined

# combine the train and test set
train_test <- bind_rows(train, test) %>% select(-id, -set)

# create svm model using parameters found from cross validation
set.seed(888)
svm_final <- train(target ~ ., data = train_test, 
                   tuneGrid = expand.grid(sigma = 0.026, C= 0.2050838),
                   metric = "MAE", method = "svmRadial", 
                   preProcess = c("center", "scale"), trControl = ctrl)



# create submission df with predictions on submission data
Blue4 <- data.frame(Prediction = predict(svm_final, submit_data)) %>%
  dplyr::mutate(Row = row_number()) %>% select(Row, Prediction)

# save submission df as csv
write.csv(Blue4, paste0(data,"Blue4.csv"), row.names = F)



