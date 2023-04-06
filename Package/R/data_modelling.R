# Data Modelling - Personal Key Indicators of Heart Disease

# Setting Work space (Root of the project directory)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Loading data and splitting it

load("../Dataset/Pre-Processed/one_hot_data.RData")
load("../Dataset/Pre-Processed/normalized_data.RData")
load("../Dataset/Pre-Processed/pca_4pc_data.RData")

# Let's first work with the one hot encoded data 80% on training set and 20% in test set

dataset_indx = data.frame(pre_pr_X, y)
dataset_indx$y = as.factor(dataset_indx$y)
set.seed(42)
idx = sample(c(TRUE,FALSE), nrow(dataset_indx), replace=TRUE, prob=c(0.8,0.2))
train_set = dataset_indx[idx, ]
test_set = dataset_indx[!idx, ]

# --- Modelling Default Architectures ---

# Random Forest ---------------------->
library(randomForest) 

rf = randomForest(formula = y~.,
                  data = train_set,
                  ntree = 100)

# Print parameters of the model
print(rf)

# Fine tune for mtry (number of variable selected at each split)

mtry = tuneRF(train_set[,-18],train_set$y, ntreeTry = 500,
              stepFactor = 1.5, improve = 0.01, trace = TRUE, plot=TRUE)

best.m = mtry[mtry[,2] == min(mtry[,2]),1]

print(mtry)
print(best.m) # Which is 3 for this random forest

# Train model with best parameter

# Taking time ---

system.time(randomForest(y~.,data=train_set,mtry=best.m,importance=TRUE,ntree=500))

rf_tuned = randomForest(y~.,data=train_set,mtry=best.m,importance=TRUE,ntree=500)
print(rf_tuned) 

# Let's see variable importance
importance(rf_tuned)
varImpPlot(rf_tuned)

# Model evaluation

pred_rf=predict(rf_tuned, type="prob")

install.packages("ROCR")
library(ROCR)

perfo = prediction(pred_rf[,2], train_set$y)
# 1. Area under curve
auc = performance(perfo, "auc")
# 2. True Positive and Negative Rate
pred3 = performance(perfo, "tpr","fpr")
# 3. Plot the ROC curve
plot(pred3,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

library(caret)

predictions_test = predict(rf_tuned, test_set)
confusionMatrix(predictions_test, test_set$y, mode="everything", positive="1")

# K-NN Model ---------------------->

library(class)
knn_model = knn(train_set[,-18], test_set[,-18], cl=train_set$y, k=5)
confusionMatrix(table(knn_model, test_set$y), mode = "everything", positive="1")

knn_model_3 = knn(train_set[,-18], test_set[,-18], cl=train_set$y, k=3)
confusionMatrix(table(knn_model_3, test_set$y), mode = "everything", positive="1")

knn_model_7 = knn(train_set[,-18], test_set[,-18], cl=train_set$y, k=7)
confusionMatrix(table(knn_model_7, test_set$y), mode = "everything", positive="1")

knn_model_12 = knn(train_set[,-18], test_set[,-18], cl=train_set$y, k=12)
confusionMatrix(table(knn_model_12, test_set$y), mode = "everything", positive="1")

# Taking time of the best model
system.time(knn(train_set[,-18], test_set[,-18], cl=train_set$y, k=5))

# Let's plot best model (k=5)
knn_model_5 = knn(train_set[,-18], test_set[,-18], cl=train_set$y, k=5, prob=TRUE)
prob_knn = attr(knn_model_5, "prob")
prob_knn <- 2*ifelse(knn_model == "-1", 1-prob_knn, prob_knn) - 1
pred_knn <- prediction(prob_knn, test_set$y)
pred_knn <- performance(pred_knn, "tpr", "fpr")
plot(pred_knn, avg= "threshold", colorize=T, lwd=3, main="Voilà, a ROC curve!")

# Plotting K-NN clustering among variables
library(plyr)
library(ggplot2)

plot.df = data.frame(test_set, prob_knn)

plot.df1 = data.frame(x = plot.df$BMI, 
                      y = plot.df$AgeCategory, 
                      predicted = plot.df$prob_knn)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

ggplot(plot.df, aes(BMI, AgeCategory)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)


# Logistic Regression ---------------------->

install.packages("tidymodels")
install.packages("glmnet")
library(tidymodels)

# Train a logistic regression model
lr_model <- logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(train_set$y ~ ., data = train_set[,-18])

# Model summary
tidy(lr_model)

# Doing predictions

# Class Predictions
pred_class <- predict(lr_model,
                      new_data = test_set[,-18],
                      type = "class")

# Class Probabilities
pred_proba <- predict(lr_model,
                      new_data = test_set[,-18],
                      type = "prob")

results <- test_set[,-18] %>%
  select(y) %>%
  bind_cols(pred_class, pred_proba)

accuracy(results, truth = test_set$y, estimate = .pred_class)
confusionMatrix(results$.pred_class, test_set$y, mode="everything", positive="1")

# Tuning Logistic Regression ---

# Define the logistic regression model with penalty and mixture hyperparameters
log_reg <- logistic_reg(mixture = tune(), penalty = tune(), engine = "glmnet")

# Define the grid search for the hyperparameters
grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 4, penalty = 3))

# Define the workflow for the model
log_reg_wf <- workflow() %>%
  add_model(log_reg) %>%
  add_formula(y ~ .)

# Define the resampling method for the grid search
folds <- vfold_cv(train_set, v = 5)

# Tune the hyperparameters using the grid search
log_reg_tuned <- tune_grid(
  log_reg_wf,
  resamples = folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

select_best(log_reg_tuned, metric = "roc_auc")

# Create the optimal model of Logistic Regression ---

# Measuring time of best model

system.time(logistic_reg(penalty=0.0000000001, mixture = 0) %>%
              set_engine("glmnet") %>%
              set_mode("classification") %>%
              fit(train_set$y ~ ., data = train_set[,-18]))

lr_tuned_model = logistic_reg(penalty=0.0000000001, mixture = 0) %>%
                 set_engine("glm") %>%
                 set_mode("classification") %>%
                 fit(train_set$y ~ ., data = train_set[,-18])

tidy(lr_tuned_model, exponentiate = TRUE) %>%
  filter(p.value < 0.05)

# Class Predictions
pred_class_tuned <- predict(lr_tuned_model,
                      new_data = test_set[,-18],
                      type = "class")

# Class Probabilities
pred_proba_tuned <- predict(lr_tuned_model,
                      new_data = test_set[,-18],
                      type = "prob")

results_tuned <- test_set[,-18] %>%
  select(y) %>%
  bind_cols(pred_class_tuned, pred_proba_tuned)

confusionMatrix(results_tuned$.pred_class, test_set$y, mode="everything", positive="1")

# XGBoost model -------------------------------->

install.packages("xgboost")
library(xgboost)
train.y <- as.numeric(as.factor(train_set$y)) - 1
dtrain = xgb.DMatrix(data=as.matrix(train_set[,-18]), label=train.y)
bst_model = xgboost(data= dtrain, max_depth=2, eta=1, nrounds=2, objective="binary:logistic")
predictions_xgb_prob = predict(bst_model, as.matrix(test_set[,-18]))
predictions_xgb_binary = as.numeric(predictions_xgb_prob > 0.5)
predictions_xgb_factor = factor(predictions_xgb_binary)
confusionMatrix(predictions_xgb_factor, test_set$y, mode="everything", positive="1")

# Now using gblinear
test.y <- as.numeric(as.factor(test_set$y)) - 1
dtest = xgb.DMatrix(data=as.matrix(test_set[,-18]), label=test.y)
watchlist <- list(train=dtrain, test=dtest)
bst_GBLINEAR <- xgb.train(data=dtrain, booster = "gblinear", nthread = 2, nrounds=2, watchlist=watchlist, eval.metric = "error", eval.metric = "logloss", objective = "binary:logistic")
pred_xgb_gbl = predict(bst_GBLINEAR, as.matrix(test_set[,-18]))
pred_xgb_gbl_bin = as.numeric(pred_xgb_gbl > 0.5)
pred_xgb_gbl_fact = factor(pred_xgb_gbl_bin)
confusionMatrix(pred_xgb_gbl_fact, test_set$y, mode="everything", positive="1")

# Trying tunning
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

param <- list(max_depth = 2, eta = 1)
bst_tn <- xgb.train(param, dtrain, nrounds = 2, watchlist, logregobj, maximize = FALSE)
pred_xgb_tn = predict(bst_tn, as.matrix(test_set[,-18]))
pred_xgb_tn_bin = as.numeric(pred_xgb_tn > 0.5)
pred_xgb_tn_fac = factor(pred_xgb_tn_bin)
confusionMatrix(pred_xgb_tn_fac, test_set$y, mode="everything", positive="1")

# Taking time of the best model XGBoost
system.time(xgb.train(param, dtrain, nrounds = 2, watchlist, logregobj, maximize = FALSE))


# --- Model Refinement - Using Logistic Regression ---

# Let's first apply over sampling ---
install.packages("ROSE")
library(ROSE)
data_balanced_oversampled = ovun.sample(y~., data=train_set, method="over", N=467470)$data
table(data_balanced_oversampled$y)

# Training model

# Time measure
system.time(logistic_reg(penalty=0.0000000001, mixture = 0) %>%
              set_engine("glm") %>%
              set_mode("classification") %>%
              fit(data_balanced_oversampled$y ~ ., data = data_balanced_oversampled[,-18]))

lr_tuned_ovs = logistic_reg(penalty=0.0000000001, mixture = 0) %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(data_balanced_oversampled$y ~ ., data = data_balanced_oversampled[,-18])

# Class Predictions
pred_class_ovs <- predict(lr_tuned_ovs,
                            new_data = test_set[,-18],
                            type = "class")
# Class Probabilities
pred_proba_ovs <- predict(lr_tuned_ovs,
                            new_data = test_set[,-18],
                            type = "prob")

results_ovs <- test_set[,-18] %>%
  select(y) %>%
  bind_cols(pred_class_ovs, pred_proba_ovs)

confusionMatrix(results_ovs$.pred_class, test_set$y, mode="everything", positive="1")

# Let's second apply under sampling ---

data_balanced_undersampled <- ovun.sample(y~., data=train_set, method="under", N=43992)$data
table(data_balanced_undersampled$y)

system.time(logistic_reg(penalty=0.0000000001, mixture = 0) %>%
              set_engine("glm") %>%
              set_mode("classification") %>%
              fit(data_balanced_undersampled$y ~ ., data = data_balanced_undersampled[,-18]))

lr_tuned_unds = logistic_reg(penalty=0.0000000001, mixture = 0) %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(data_balanced_undersampled$y ~ ., data = data_balanced_undersampled[,-18])

# Class Predictions
pred_class_unds <- predict(lr_tuned_unds,
                          new_data = test_set[,-18],
                          type = "class")
# Class Probabilities
pred_proba_unds <- predict(lr_tuned_unds,
                          new_data = test_set[,-18],
                          type = "prob")

results_unds <- test_set[,-18] %>%
  select(y) %>%
  bind_cols(pred_class_unds, pred_proba_unds)

confusionMatrix(results_unds$.pred_class, test_set$y, mode="everything", positive="1")

# Let's third, do both ---

data_balanced_both <- ovun.sample(y~., data=train_set, method="both", p=0.5, N=55000, seed=1)$data
table(data_balanced_both$y)
table(train_set$y)

system.time(logistic_reg(penalty=0.0000000001, mixture = 0) %>%
              set_engine("glm") %>%
              set_mode("classification") %>%
              fit(data_balanced_both$y ~ ., data = data_balanced_both[,-18]))

lr_tuned_bth = logistic_reg(penalty=0.0000000001, mixture = 0) %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(data_balanced_both$y ~ ., data = data_balanced_both[,-18])

# Class Predictions
pred_class_bth <- predict(lr_tuned_bth,
                           new_data = test_set[,-18],
                           type = "class")
# Class Probabilities
pred_proba_bth <- predict(lr_tuned_bth,
                           new_data = test_set[,-18],
                           type = "prob")

results_bth <- test_set[,-18] %>%
  select(y) %>%
  bind_cols(pred_class_bth, pred_proba_bth)

confusionMatrix(results_bth$.pred_class, test_set$y, mode="everything", positive="1")

# Finally, let's use ROSE technique ---

data_balanced_rose = ROSE(y~., data = train_set, seed = 1)$data
table(data_balanced_rose$y)

system.time(logistic_reg(penalty=0.0000000001, mixture = 0) %>%
              set_engine("glm") %>%
              set_mode("classification") %>%
              fit(data_balanced_rose$y ~ ., data = data_balanced_rose[,-18]))

lr_tuned_rose = logistic_reg(penalty=0.0000000001, mixture = 0) %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(data_balanced_rose$y ~ ., data = data_balanced_rose[,-18])

# Class Predictions
pred_class_rose <- predict(lr_tuned_rose,
                          new_data = test_set[,-18],
                          type = "class")
# Class Probabilities
pred_proba_rose <- predict(lr_tuned_rose,
                          new_data = test_set[,-18],
                          type = "prob")

results_rose <- test_set[,-18] %>%
  select(y) %>%
  bind_cols(pred_class_rose, pred_proba_rose)

confusionMatrix(results_rose$.pred_class, test_set$y, mode="everything", positive="1")

# Second possible refinement, using dataset after PCA ---

dataset_pca_indx = data.frame(data_4pc, y)
dataset_pca_indx$y = as.factor(dataset_pca_indx$y)
set.seed(42)
idx_pca = sample(c(TRUE,FALSE), nrow(dataset_pca_indx), replace=TRUE, prob=c(0.8,0.2))
train_set_pca = dataset_pca_indx[idx_pca, ]
test_set_pca = dataset_pca_indx[!idx_pca, ]

# Balancing data 

data_pca_bboth <- ovun.sample(y~., data=train_set_pca, method="both", p=0.5, N=55000, seed=1)$data
table(data_pca_bboth$y)

# Training model

system.time(logistic_reg(penalty=0.0000000001, mixture = 0) %>%
              set_engine("glm") %>%
              set_mode("classification") %>%
              fit(data_pca_bboth$y ~ ., data = data_pca_bboth[,-18]))

lr_pca_both = logistic_reg(penalty=0.0000000001, mixture = 0) %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(data_pca_bboth$y ~ ., data = data_pca_bboth[,-18])

# Class Predictions
pred_class_pca <- predict(lr_pca_both,
                           new_data = test_set_pca[,-18],
                           type = "class")
# Class Probabilities
pred_proba_pca <- predict(lr_pca_both,
                           new_data = test_set_pca[,-18],
                           type = "prob")

results_pca <- test_set_pca[,-18] %>%
  select(y) %>%
  bind_cols(pred_class_pca, pred_proba_pca)

confusionMatrix(results_pca$.pred_class, test_set_pca$y, mode="everything", positive="1")
