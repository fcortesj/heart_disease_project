# Exploratory analysis of the data - Personal Key Indicators of Heart Disease

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# --- Installing and importing libraries ---
install.packages("dplyr")
install.packages("ggplot2")

library(dplyr)
library(ggplot2)

# --- Importing data ---
raw_data=read.csv("../Dataset/heart_2020_cleaned.csv")
head(raw_data)

# -- Let's visualize the summary (Categorical and Numeric columns)

summary(raw_data)

# --- Pre-processing steps ---

# Checking null values (No null values) ---
sum(is.na(raw_data)) 

# Separate data set according to heart disease or not
yhd_data=raw_data[raw_data$HeartDisease == "Yes",]
nhd_data=raw_data[raw_data$HeartDisease == "No",]

# Get numerical data
numerical_columns <- unlist(lapply(raw_data, is.numeric))
numeric_data <- raw_data[,numerical_columns]

# --- Reviewing important elements ---

# Unique values per column
sapply(raw_data, function(x) n_distinct(x))

# --- Basic Interesting Graphs (General) ---

# How many people suffered from heart diseased?
mytable <- table(raw_data$HeartDisease)
lbls <- paste(names(mytable), "\n", mytable, sep="")
pie(mytable,
    labels = lbls,
    main="Pie Chart of Heart Diseased column data\n (with sample sizes)")

# From the diseased, what was the gender most affected?
barplot(table(yhd_data$Sex),
     xlab="Gender",
     ylab="Frequency")

# From the diseased, what was their age?
barplot(table(yhd_data$AgeCategory),
     xlab="Age Range",
    ylab="Frequency")

# From the diseased, what was their race?
barplot(table(yhd_data$Race),
        xlab="Race",
        ylab="Frequency")

# From the diseased, were they diabetic?
barplot(table(yhd_data$Diabetic),
        xlab="Diabetic",
        ylab="Frequency")

# From the diseased, did they do physical activity?
barplot(table(yhd_data$PhysicalActivity),
        xlab="Physical Activity",
        ylab="Frequency")

# From the diseased, did they have asthma?
barplot(table(yhd_data$Asthma),
        xlab="Asthma",
        ylab="Frequency")

# From the diseased, did they have Kidney Cancer?
barplot(table(yhd_data$KidneyDisease),
        xlab="Kidney Cancer",
        ylab="Frequency")

# From the diseased, did they have Skin Cancer?
barplot(table(yhd_data$SkinCancer),
        xlab="Skin Cancer",
        ylab="Frequency")

# From the diseased, did they smoke?
barplot(table(yhd_data$Smoking),
        xlab="Smoking",
        ylab="Frequency")

# From the diseased, did they drink alcohol?
barplot(table(yhd_data$AlcoholDrinking),
        xlab="Alcohol Drinking",
        ylab="Frequency")

# From the diseased, did they have an stroke before?
barplot(table(yhd_data$Stroke),
        xlab="Stroke",
        ylab="Frequency")

# From the diseased, did they have difficulties walking or using the stairs?
barplot(table(yhd_data$DiffWalking),
        xlab="Difficulties in Walking or Stair Climbing",
        ylab="Frequency")

# --- Interesting graphs from numerical data ---

#BMI (Body Mass Index) variable
hist(numeric_data$BMI, main="Distribution of BMI")
boxplot(yhd_data$BMI, horizontal=TRUE, main="BMI from diseased patients")
# From the diseased, what was their BMI (Body Mass Index)?
temp<-density(table(yhd_data$BMI))
plot(temp, type="n", main="BMI from diseased patients")
polygon(temp, col="lightgray",border="gray")

#Physical Health variable
hist(numeric_data$PhysicalHealth, main="Distribution of PhysicalHealth")
boxplot(yhd_data$PhysicalHealth, horizontal=TRUE, main="PhysicalHealth from diseased patients")
# From the diseased, did they suffered from physical injuries in the past (30 days)?
temp<-density(table(yhd_data$PhysicalHealth))
plot(temp, type="n", main="Physical Injuries in the past 30 days for diseased patients")
polygon(temp, col="lightgray",border="gray")

# Mental Health variable
hist(numeric_data$MentalHealth, main="Distribution of Mental Health")
boxplot(yhd_data$MentalHealth, horizontal=TRUE, main="Mental Health from diseased patients")
# From the diseased, did they suffered from bad mental health in the past (30 days)?
temp<-density(table(yhd_data$MentalHealth))
plot(temp, type="n", main="Bad Mental episodes in the past 30 days for diseased patients")
polygon(temp, col="lightgray",border="gray")

# Sleep Time variable
hist(numeric_data$SleepTime, main="Distribution of Sleep Time")
boxplot(yhd_data$SleepTime, horizontal=TRUE, main="Sleep Time from diseased patients")
temp<-density(table(yhd_data$SleepTime))
plot(temp, type="n", main="Sleep Time for diseased patients")
polygon(temp, col="lightgray",border="gray")

# --- Comparisons between variables ---

# Influence of alcohol drinking and physical health
boxplot(yhd_data$PhysicalHealth, horizontal=TRUE, main="PhysicalHealth from diseased patients")
