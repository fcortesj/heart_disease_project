# Data pre-processing - Personal Key Indicators of Heart Disease

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# --- Importing data ---
raw_data=read.csv("../Dataset/heart_2020_cleaned.csv")
head(raw_data)

# --- Pre-processing steps ---
pre_pr_data=data.frame(raw_data)
y=factor(pre_pr_data[,1], levels=c('Yes', 'No'), labels=c(1.0, 0.0))
X=pre_pr_data[,2:ncol(pre_pr_data)]
pre_pr_X=data.frame(X)

# Transforming categorical data in numerical - Yes, No (Bi-classes)
install.packages("tictoc")
library(tictoc)

tic("Standard Encoding Timing")
pre_pr_X$Smoking = factor(pre_pr_X$Smoking,
                                  levels = c('Yes', 'No'),
                                  labels = c(1.0, 0.0))

pre_pr_X$AlcoholDrinking = factor(pre_pr_X$AlcoholDrinking,
                             levels = c('Yes', 'No'),
                             labels = c(1.0, 0.0))

pre_pr_X$Stroke = factor(pre_pr_X$Stroke,
                                     levels = c('Yes', 'No'),
                                     labels = c(1.0, 0.0))

pre_pr_X$DiffWalking = factor(pre_pr_X$DiffWalking,
                            levels = c('Yes', 'No'),
                            labels = c(1.0, 0.0))

pre_pr_X$Sex = factor(pre_pr_X$Sex,
                                 levels = c('Female', 'Male'),
                                 labels = c(1.0, 0.0))

pre_pr_X$PhysicalActivity = factor(pre_pr_X$PhysicalActivity,
                         levels = c('Yes', 'No'),
                         labels = c(1.0, 0.0))

pre_pr_X$Asthma = factor(pre_pr_X$Asthma,
                                      levels = c('Yes', 'No'),
                                      labels = c(1.0, 0.0))

pre_pr_X$KidneyDisease = factor(pre_pr_X$KidneyDisease,
                            levels = c('Yes', 'No'),
                            labels = c(1.0, 0.0))

pre_pr_X$SkinCancer = factor(pre_pr_X$SkinCancer,
                            levels = c('Yes', 'No'),
                            labels = c(1.0, 0.0))

# Transforming categorical data in numerical - Multiple classes
pre_pr_X$Race = factor(pre_pr_X$Race,
                                levels = c('White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'),
                                labels = c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

pre_pr_X$Diabetic = factor(pre_pr_X$Diabetic,
                                levels = c('Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'),
                                labels = c(1.0, 2.0, 3.0, 4.0))

pre_pr_X$GenHealth = factor(pre_pr_X$GenHealth,
                          levels = c('Poor', 'Fair', 'Good', 'Very good', 'Excellent'),
                          labels = c(1.0, 2.0, 3.0, 4.0, 5.0))

pre_pr_X$AgeCategory = factor(pre_pr_X$AgeCategory,
                               levels = c('18-24','25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'),
                               labels = c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0))

# Handling factor to numeric
indx <- sapply(pre_pr_X, is.factor)
pre_pr_X[indx] <- lapply(pre_pr_X[indx], function(x) as.numeric(as.character(x)))
pre_pr_data <- data.frame(pre_pr_X)

y <- as.numeric(as.character(y))

# Ending time measure
toc("Standard Encoding Timing")


# 2nd way to encode the data - One Hot data Encoder
install.packages("caret")
library(caret)

system.time(dmy <- dummyVars(" ~ .", data = pre_pr_data, fullRank = T))
system.time(onehot_data <- data.frame(predict(dmy, newdata = pre_pr_data)))

# --- Calculate Correlation among encoded features ---

install.packages("Hmisc")
library("Hmisc")

pre_pr_correlation <- rcorr(as.matrix(pre_pr_data))
one_hot_correlation <- rcorr(as.matrix(onehot_data))
df_pre_pr_correlation <- pre_pr_correlation$r
df_pre_pr_correlation <- replace(df_pre_pr_correlation, is.na(df_pre_pr_correlation), 0)

# Drawing correlation plots

install.packages("corrplot")
install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")
library("corrplot")

corrplot(pre_pr_correlation$r, type = "upper", 
         tl.col = "black", tl.srt = 45)

# Too many elements (hard to visualize)
corrplot(one_hot_correlation$r, type = "upper", 
         tl.col = "black", tl.srt = 45, number.cex= 7/ncol(one_hot_correlation$r))

# Getting highest correlations among all the data set
install.packages("lares")
library("lares")

corr_cross(pre_pr_data, 
           max_pvalue = 0.07, # display only significant correlations (at 5% level)
           top = 10 # display top 10 couples of variables (by correlation coefficient)
)

corr_cross(onehot_data, 
           max_pvalue = 0.07, # display only significant correlations (at 5% level)
           top = 10 # display top 10 couples of variables (by correlation coefficient)
)

# Using other modules and scaled data

install.packages("corrr")
install.packages("ggcorrplot")
install.packages("FactoMineR")
install.packages("factoextra")
library("corrr")
library("ggcorrplot")
library("FactoMineR")
library("factoextra")

# Scaling the data
system.time(data_normalized <- scale(pre_pr_data))

# New correlation matrix
corr_matrix <- cor(pre_pr_data)
ggcorrplot(corr_matrix)

# ---- Feature Selection procedures ----

# --- PCA Analysis of the data ---

tic("PCA Time")
data.pca = princomp(corr_matrix)
summary(data.pca)
toc("PCA Time")

# We can see that at least we will need 9 components to represent accurately the data

data.pca$loadings[2:17, 1:8]

# Visualizations of PCA ---

# Percentage of explained variance by each principal component

fviz_eig(data.pca, addlabels = TRUE)

# Biplot of the attributes

fviz_pca_var(data.pca, col.var = "black")

# Contribution of each variable (Squared Cosine Calculation)

fviz_cos2(data.pca, choice = "var", axes = 1:2, col= "red")

# Biplot with variable contribution

fviz_pca_var(data.pca, col.var = "cos2",
             gradient.cols = c("black", "orange", "green"),
             repel = TRUE)

# Plotting the data distribution in the space
install.packages("ggfortify")
library(ggfortify)

autoplot(data.pca, label=TRUE)

# Now, let's do the same but with the complete data and not just the correlation matrix among them

data.pca.normalized = princomp(data_normalized)
summary(data.pca.normalized)

# Percentage of explained variance by each principal component. Looking that we will need at least 7 components to achieve 40% explained variance

fviz_eig(data.pca.normalized, addlabels = TRUE)

# Biplot of the attributes

fviz_pca_var(data.pca, col.var = "black")

# Contribution of each variable (Squared Cosine Calculation)

fviz_cos2(data.pca, choice = "var", axes = 1:2)

# Biplot with variable contribution

fviz_pca_var(data.pca, col.var = "cos2",
             gradient.cols = c("black", "orange", "green"),
             repel = TRUE)

# --- Association rules - Apriori Algorithm ---

# First, we are going to joint the resulting data sets in one

# Simple One-Hot Encoded

final_pre_processed_data = cbind(pre_pr_X, y)
colnames(final_pre_processed_data)[colnames(final_pre_processed_data) == "y"] = "HeartDisease"

# Scaled Normalized One-Hot Encoded

final_normalized_data = cbind(data_normalized, y)
colnames(final_normalized_data)[colnames(final_normalized_data) == "y"] = "HeartDisease"

# PCA with 7 principal components
data_4pc = data.pca.normalized$scores[,1:7]
final_pca_data = cbind(data_4pc, y)
colnames(final_pca_data)[colnames(final_pca_data) == "y"] = "HeartDisease"

# Installing libraries needed
install.packages("arules")
install.packages("arulesViz")
library(arules)
library(arulesViz)

# First, we are going to convert our data as a transaction set
trans = as(final_pre_processed_data, "transactions")
summary(trans)
itemLabels(trans)

# We are going to find association rules with default settings, we have of 80.000 rules
default_rules = apriori(trans)
#inspect(default_rules)

# We will then explore the rules a little more constrained using supp, confidence interval and min length
filtered_rules = apriori(trans,
                         parameter = list(supp=0.25, conf=0.9),
                         control = list(verbose=F))
filtered_rules_sorted = sort(filtered_rules, by="lift")
#inspect(filtered_rules_sorted)

# Now, let's filter even more using the target columns as our right hand side part of the rule
tic("Apriori_Time")
filtered_rules = apriori(trans,
                         parameter = list(supp=0.25, conf=0.9),
                         appearance = list(rhs="HeartDisease=[0,1]"))
filtered_rules_sorted = sort(filtered_rules, by="lift")
toc("Apriori_Time")
#inspect(filtered_rules_sorted)


# Plot the results
plot(filtered_rules_sorted, method="graph", control=list(type="items"), limit=20, colors=c("black","blue", "black"))

plot(filtered_rules_sorted, method="paracoord", control=list(reorder=TRUE), limit=20)

# Removing redundant rules (Not meaning ful one)
final_rules <- filtered_rules_sorted[!is.redundant(filtered_rules_sorted)]

# --- Saving Final Results Data ---

save(pre_pr_X, y, file="../Dataset/Pre-Processed/one_hot_data.RData")
save(data_normalized, y, file="../Dataset/Pre-Processed/normalized_data.RData")
save(data_4pc, y, file="../Dataset/Pre-Processed/pca_4pc_data.RData")