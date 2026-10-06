# Week 1: Data Cleaning and Preliminary Analysis with R
# Dataset: Titanic passenger dataset (891 rows, 12 columns)

# 1. Packages
install.packages(c("tidyverse", "janitor", "scales"))
library(tidyverse)
library(janitor)
library(scales)

# 2. Import
url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic <- read.csv(url, stringsAsFactors = FALSE)

# 3. Initial inspection
dim(titanic)
str(titanic)
summary(titanic)

# Missing-value count
colSums(is.na(titanic))

# 4. Standardise column names
titanic <- clean_names(titanic)

# 5. Basic cleaning and feature engineering
titanic_clean <- titanic %>%
  mutate(
    survived = factor(survived, levels = c(0,1),
                      labels = c("No","Yes")),
    pclass = factor(pclass, levels = c(1,2,3),
                    labels = c("1st","2nd","3rd")),
    sex = factor(sex),
    embarked = na_if(embarked, ""),
    embarked = replace_na(embarked, "S"),
    embarked = factor(embarked),
    age = ifelse(is.na(age),
                 ave(age, pclass, sex, FUN = function(x)
                   median(x, na.rm = TRUE)),
                 age),
    family_size = sib_sp + parch + 1,
    is_alone = ifelse(family_size == 1, "Yes", "No"),
    deck = ifelse(is.na(cabin), "Unknown",
                  substr(cabin, 1, 1))
  )

# Check whether duplicates exist
sum(duplicated(titanic_clean))

# Cabin has very high missingness; retain useful deck information
# and remove the original high-missingness cabin field.
titanic_clean <- titanic_clean %>%
  select(-cabin)

# Remove identifier/text columns not required for preliminary numeric analysis
titanic_model <- titanic_clean %>%
  select(-passenger_id, -name, -ticket)

# 6. Outlier detection using IQR
outlier_report <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr_value <- q3 - q1
  lower <- q1 - 1.5 * iqr_value
  upper <- q3 + 1.5 * iqr_value
  n <- sum(x < lower | x > upper, na.rm = TRUE)
  data.frame(Q1=q1, Q3=q3, IQR=iqr_value,
             Lower=lower, Upper=upper, Outliers=n)
}

outlier_report(titanic_clean$fare)
outlier_report(titanic_clean$age)

# Cap extreme fares at the IQR upper fence (winsorisation)
fare_upper <- as.numeric(quantile(titanic_clean$fare, .75, na.rm=TRUE)) +
              1.5 * IQR(titanic_clean$fare, na.rm=TRUE)
titanic_clean <- titanic_clean %>%
  mutate(fare_clean = pmin(fare, fare_upper))

# 7. Min-max normalization for selected numerical variables
minmax <- function(x) (x - min(x, na.rm=TRUE)) /
                      (max(x, na.rm=TRUE) - min(x, na.rm=TRUE))

titanic_clean <- titanic_clean %>%
  mutate(
    age_norm = minmax(age),
    fare_norm = minmax(fare_clean),
    family_size_norm = minmax(family_size)
  )

# 8. Categorical encoding
# Binary encoding for sex and one-hot/dummy encoding for embarked.
titanic_clean <- titanic_clean %>%
  mutate(
    sex_binary = ifelse(sex == "male", 1, 0),
    embarked_C = ifelse(embarked == "C", 1, 0),
    embarked_Q = ifelse(embarked == "Q", 1, 0),
    embarked_S = ifelse(embarked == "S", 1, 0)
  )

# 9. Descriptive statistics
summary(titanic_clean[, c("age","fare_clean","family_size","sib_sp","parch")])

# Survival rate
prop.table(table(titanic_clean$survived))

# Survival by sex
prop.table(table(titanic_clean$sex, titanic_clean$survived), margin=1)

# Survival by class
prop.table(table(titanic_clean$pclass, titanic_clean$survived), margin=1)

# 10. Correlation matrix for numeric variables
num_df <- titanic_clean %>%
  select(survived_num = as.numeric(survived)-1,
         pclass_num = as.numeric(pclass),
         age, sib_sp, parch, fare_clean, family_size)

cor(num_df, use = "pairwise.complete.obs")

# 11. Visualizations
ggplot(titanic_clean, aes(x=survived)) +
  geom_bar() +
  labs(title="Survival Distribution", x="Survived", y="Count")

ggplot(titanic_clean, aes(x=sex, fill=survived)) +
  geom_bar(position="dodge") +
  labs(title="Survival by Sex", x="Sex", y="Count")

ggplot(titanic_clean, aes(x=pclass, fill=survived)) +
  geom_bar(position="dodge") +
  labs(title="Survival by Passenger Class", x="Passenger Class", y="Count")

ggplot(titanic_clean, aes(x=age)) +
  geom_histogram(bins=30, na.rm=TRUE) +
  labs(title="Age Distribution", x="Age", y="Count")

ggplot(titanic_clean, aes(y=fare_clean)) +
  geom_boxplot() +
  labs(title="Fare After Outlier Treatment", y="Fare")

# 12. Save cleaned dataset
write.csv(titanic_clean, "titanic_cleaned_week1.csv", row.names=FALSE)
