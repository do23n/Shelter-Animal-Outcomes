---
title: "Project"
author: "Sabirah, Linh, Nhu"
date: "`r format(Sys.time(), '%d %B, %Y')`"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(purrr)
library(glmnet)
library(caret)
library(lubridate) # dates
library(tidyr)
library("e1071")
library(xgboost)
```

### Project Description/Motivation: 

Every year, approximately 7.6 million companion animals end up in US shelters. Many animals are given up as unwanted by their owners, while others are picked up after getting lost or taken out of cruelty situations. Many of these animals find forever families to take them home, but just as many are not so lucky. 2.7 million dogs and cats are euthanized in the US every year.

Using a dataset of intake information including breed, color, sex, and age from the Austin Animal Center, we’re asking Kagglers to predict the outcome for each animal.

We also believe this dataset can help us understand trends in animal outcomes. These insights could help shelters focus their energy on specific animals who need a little extra help finding a new home.

```{r}
animals <- read.csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTVu8lqdQwLvIctdsj8MlnE7d53d58b8gAoGFd-sgLqsakuuV156WV8Ab2i2FzXL9k1k-fCaQqfxC9R/pub?output=csv")
head(animals)
```


```{r}
len_full = nrow(animals)
unique(animals$AnimalType)
```

### Cleanup data


#### `AgeuponOutcome` variable

We referred to (1) for the idea used in this section.

Here I’ll use strsplit to break it up into two parts: the numeric value and its unit of time (i.e., days, weeks, months, and years). I’m also going to rip off the final “s” from everything (i.e., day versus days) so that we’re left with just four levels of a new factor variable
```{r}
# Use strsplit to break AgeuponOutcome up into two parts: the numeric time value and its unit of time
# First, we need to convert AgeuponOutcome to character
animals$AgeuponOutcome <- as.character(animals$AgeuponOutcome)
# Get the numeric time value:
animals$TimeValue <- sapply(animals$AgeuponOutcome,  
                      function(x) strsplit(x, split = " ")[[1]][1])
# Get the unit of time:
animals$UnitofTime <- sapply(animals$AgeuponOutcome,  
                      function(x) strsplit(x, split = " ")[[1]][2])
# Fortunately any "s" marks the plural, so we can just pull them all out
animals$UnitofTime <- gsub('s', '', animals$UnitofTime)
animals$TimeValue  <- as.numeric(animals$TimeValue)
animals$UnitofTime <- as.factor(animals$UnitofTime)
```

Now we need to get the animals’ ages as measured by a common unit, and let's convert everything into days.

```{r}
# Make a multiplier vector
multiplier <- ifelse(animals$UnitofTime == 'day', 1,
              ifelse(animals$UnitofTime == 'week', 7,
              ifelse(animals$UnitofTime == 'month', 30, # Close enough
              ifelse(animals$UnitofTime == 'year', 365, NA))))
# Apply our multiplier
animals$AgeinDays <- animals$TimeValue * multiplier
```


#### `DateTime` variable

We referred to (1) for the idea used in this section.

```{r}
# Extract time variables from date (uses the "lubridate" package)
animals$Hour    <- hour(animals$DateTime)
animals$Weekday <- wday(animals$DateTime)
animals$Month   <- month(animals$DateTime)
animals$Year    <- year(animals$DateTime)
# Time of day may also be useful
animals$TimeofDay <- ifelse(animals$Hour > 5 & animals$Hour < 11, 'morning',
                  ifelse(animals$Hour > 10 & animals$Hour < 16, 'midday',
                  ifelse(animals$Hour > 15 & animals$Hour < 20, 'lateday', 'night')))
```


#### Cleaning up the levels of Breed
Since many animals are of mixed breeds and there are too many breeds, we will split the variable Breed into 2 variables: IsMix(Yes,No), and SimpleBreed(Only contains the first breed listed of each animal).

```{r}
# Take a look at some of the levels
levels(factor(animals$Breed))[1:10]
# Use "grepl" to look for "Mix"
animals$IsMix <- ifelse(grepl('Mix', animals$Breed), 1, 0)
# Remove " Mix", Split on "/" and only take the characters come before "/" to simplify Breed
animals$SimpleBreed <- sapply(animals$Breed, 
                      function(x) gsub(' Mix', '', 
                        strsplit(as.character(x), split = '/')[[1]][1]))
# Now only around 230 levels of breed so have simplified it
```


#### Cleaning up the colors
The same process for Breed is applied to Colors to simplified the color of each animal. The variable SimpleColor contains only the first color listed of each animal.

```{r}
# Use strsplit to grab the first color
animals$SimpleColor <- sapply(animals$Color, 
                      function(x) strsplit(as.character(x), split = '/| ')[[1]][1])
# Testing the new colors 
levels(factor(animals$SimpleColor))
```

#### Splitting Sex Intactness into 2 Separate Variables
The Variable Sex is splitted into 2 variables: Intact (Yes/No/Unknown), and Sex(Male/Female/Unknown)

```{r}
# Use "grepl" to look for "Intact"
animals$Intact <- ifelse(grepl('Intact', animals$SexuponOutcome), 1,
               ifelse(grepl('Unknown', animals$SexuponOutcome), 'Unknown', 0))
# Use "grepl" to look for sex
animals$Sex <- ifelse(grepl('Male', animals$SexuponOutcome), 'Male',
            ifelse(grepl('Unknown', animals$Sex), 'Unknown', 'Female'))
```

#### Assessing Missing Data

```{r}
count_missing <- function(x) {
  sum(is.na(x))
}
#Missing data in the original data set:
map_dbl(animals, count_missing)
```
#### Imputation of Missing Data 
We impute the missing data points in AgeinDays (which is TimeValue * UnitofTime) using the median age of the animals:
```{r}
impute_missing_median <- function(x) {
x[is.na(x)] <- median(x, na.rm = TRUE)
return(x)
}
animals <- animals %>% mutate_at("AgeinDays", impute_missing_median)
#check if all the missing values has been filled in
sum(is.na(animals$AgeinDays))
```

#### Adding `Lifestage` variable
The variable Lifestage has 2 levels: baby (less than 1 year old), and adult (older than 1 year old)

```{r}
# Use the age variable to make a puppy/kitten variable
animals$Lifestage[animals$AgeinDays < 365] <- 'baby'
animals$Lifestage[animals$AgeinDays >= 365] <- 'adult'
# Add new variable Lifestage 
animals$Lifestage <- factor(animals$Lifestage)
```

### Plots

### Make plot to see if 'LifeStage' makes a difference in outcome

```{r}
ggplot(animals[1:26729, ], aes(x = Lifestage, fill = OutcomeType)) + 
  geom_bar(position = 'fill', colour = 'black') +
  ggtitle("Animal Outcome: Babies versus Adults")
```


#### Make plot to see if `TimeofDay` makes a difference in outcome.

```{r}
# Reshape full dataset
daytimes <- animals[1:len_full, ] %>%
  group_by(AnimalType, TimeofDay, OutcomeType) %>%
  summarise(num_animals = n())
# Plot
ggplot(daytimes, aes(x = TimeofDay, y = num_animals, fill = OutcomeType)) +
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  facet_wrap(~AnimalType) +
  coord_flip() +
  labs(y = 'Proportion of Animals', 
       x = 'Time of Day',
       title = 'Outcomes by Time of Day: Cats & Dogs')
```

#### Make plot to see if animal type makes any difference in the outcomes
```{r}
outcomes <- animals[1:len_full, ] %>%
  group_by(AnimalType, OutcomeType) %>%
  summarise(num_animals = n())
ggplot(outcomes, aes(x = AnimalType, y = num_animals, fill = OutcomeType)) +
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  coord_flip() +
  labs(y = 'Proportion of Animals', 
       x = 'Animal',
       title = 'Outcomes by Animal Type: Cats & Dogs') 
```

#### The effect of `SexuponOutcome` toward outcomes
```{r}
unique(animals$SexuponOutcome)
outcomes_sex <- animals[1:len_full, ] %>%
  group_by(AnimalType, SexuponOutcome, OutcomeType) %>%
  summarise(num_animals = n())
ggplot(outcomes_sex, aes(x = SexuponOutcome, y = num_animals, fill = OutcomeType)) +
  facet_wrap(~AnimalType) + 
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  coord_flip() +
  labs(y = 'Proportion of Animals', 
       x = 'Sex',
       title = 'Outcomes by Animal\'s Sex: Cats & Dogs') 
```

```{r}
outcomes_Lifestage <- animals[1:len_full, ] %>%
  group_by(AnimalType, Lifestage, OutcomeType) %>%
  summarise(num_animals = n())
ggplot(outcomes_Lifestage, aes(x = Lifestage, y = num_animals, fill = OutcomeType)) +
  facet_wrap(~AnimalType) + 
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  coord_flip() +
  labs(y = 'Proportion of Animals', 
       x = 'Lifestage',
       title = 'Outcomes by Animal\'s Life stage: Cats & Dogs') 
```

#### The effect of `AgeinDays` toward outcomes

```{r}
outcomes_age <- animals[1:len_full, ] %>%
  group_by(AnimalType, AgeinDays, OutcomeType) %>%
  summarise(num_animals = n())
ggplot(outcomes_age, aes(x = AgeinDays, y = num_animals, col = OutcomeType)) +
  facet_wrap(~AnimalType) + 
  geom_line(size = 1, alpha = 0.8) +
  labs(y = 'Proportion of Animals', 
       x = 'Age In Days',
       title = 'Outcomes by Age: Cats & Dogs') 
```

### Models

#### Train/Test Split
```{r}
#convert SimpleColor, SimpleBreed to factors to ensure that in both the training and test set, the same levels of the variables will be used.
animals <- animals %>%
  mutate(
    SimpleColor = factor(SimpleColor),
    SimpleBreed = factor(SimpleBreed)
  )
```

```{r}
# Initial train/test split of the animal shelter data
set.seed(53278)
tt_inds <- caret::createDataPartition(animals$OutcomeType, p = 0.8)
animals_train <- animals %>% dplyr::slice(tt_inds[[1]])
animals_test <- animals %>% dplyr::slice(-tt_inds[[1]])
```

### Set up cross validation folds
```{r}
set.seed(1213)
crossval_val_fold_inds <- caret::createFolds(
  y = animals_train$OutcomeType,
  k = 10
)
#get train set fold inds
get_complimentary_inds <- function(k){
  return (seq_len(nrow(animals_train))[-k])
}
crossval_train_fold_inds <- purrr::map(crossval_val_fold_inds, get_complimentary_inds)
```

#### Comparing observations in test set v.s. training set
```{r}
# checking to see if train set and test set observations have the same levels of SimpleColor
anti_join(animals_test, animals_train, by = "SimpleColor")
# it turns out that there are test set observations where `SimpleColor`is `Agouti`, but this simple color name does not exist in the training set.
```

Test set performance: We can only predict test set observations where `SimpleColor`is not `Agouti`, since this name does not exist in the training set. However, there are only 2 animals whose `SimpleColor` is `Agouti`, and let's just ignore them (sorry Sierra and Chip...).

```{r}
# checking to see if train set and test set observations have the same levels of SimpleBreed
anti_join(animals_test, animals_train, by = "SimpleBreed")
# it turns out that there are test set observations where `SimpleColor`is `Agouti`, but this simple color name does not exist in the training set.
```
We can only predict test set observations where `SimpleBreed`is not 'Afghan Hound, 'English Setter', 'Treeing Tennesse Brindle', 'Otterhound', 'Entlebucher', or 'Spinone Italiano', since these levels do not exist in the training set. However, there are only 7 such animals, and let's just ignore them (sorry to all of you...).

### Gradient Tree Boosting: 

### 1st model

```{r}
###############################
#
# Do NOT run this block of code
#
###############################
xgb_fit <- train(
  OutcomeType ~ AnimalType + Lifestage + IsMix + Sex + SimpleColor,
  data = animals_train,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 10,
    returnResamp = "all",
    index = crossval_train_fold_inds,
    indexOut = crossval_val_fold_inds,
    savePredictions = TRUE
  ),
  tuneGrid = expand.grid(
    nrounds = c(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    eta = c(0.5, 0.6, 0.7), # learning rate; 0.3 is the default
    gamma = 0, # minimum loss reduction to make a split; 0 is the default
    max_depth = 1:5, # how deep are our trees?
    subsample = c(0.5, 0.9, 1), # proportion of observations to use in growing each tree
    colsample_bytree = 1, # proportion of explanatory variables used in each tree
    min_child_weight = 1 # think of this as how many observations must be in each leaf node
  )
)
xgb_fit$results %>% filter(Accuracy == max(Accuracy))
saveRDS(xgb_fit, file = "xgb_fit.rds")
```


```{r}
##############################
#
# Only need to run this block
#
##############################
xgb_fit <- readRDS("xgb_fit.rds")
mean(animals_test[animals_test$SimpleColor != 'Agouti', ]$OutcomeType != predict(xgb_fit, animals_test[animals_test$SimpleColor != 'Agouti', ]))
```

Test set error rate is 0.5224635, and this is a pretty bad prediction... 

We'll try to make another fit using more explanatory variables. 

### Variable Importance

```{r}
plot(varImp(xgb_fit, scale = FALSE), top = 15)
```

```{r}
# Extract variable importance from varImp
ImpMeasure_xgb_fit<-data.frame(varImp(xgb_fit, scale = FALSE)$importance)
# Extract corresponding explanatory variable names of greatest to lowest variable importance
ImpMeasure_xgb_fit$Vars <- row.names(ImpMeasure_xgb_fit)
# sum of variable importance over all levels of variable `Sex`
sum_Sex <- sum(ImpMeasure_xgb_fit[grepl( "Sex" , ImpMeasure_xgb_fit$Vars ),  ]$Overall)
# sum of variable importance over all levels of variable `SimpleColor`
sum_SimpleColor <- sum(ImpMeasure_xgb_fit[grepl( "SimpleColor" , ImpMeasure_xgb_fit$Vars ),  ]$Overall)
# Adding two corresponding "observations" to ImpMeasure_xgb_fit for (all) `Sex` and (all) `SimpleColor`
ImpMeasure_xgb_fit <- rbind(ImpMeasure_xgb_fit, Sex = c(sum_Sex, 'Sex'))
ImpMeasure_xgb_fit <- rbind(ImpMeasure_xgb_fit, SimpleColor = c(sum_SimpleColor, 'SimpleColor'))
# Extract only the explanatory variables used in xgb_fit
ImpMeasure_xgb_fit <- ImpMeasure_xgb_fit[which(ImpMeasure_xgb_fit$Vars %in% c('Lifestagebaby', 'AnimalTypeDog', 'Sex', 'SimpleColor', 'IsMix')), ]
# plot variable importance
ggplot(ImpMeasure_xgb_fit, aes(x=`Vars`, y=Overall)) + 
  geom_point(stat='identity', fill="red", size=2)  +
  geom_segment(aes(y = 0, 
                   x = `Vars`, 
                   yend = Overall, 
                   xend = `Vars`), 
               color = "cornflowerblue") +
  labs(x = "Explanatory variables used in model fit", y = "Variable Importance", title = "Variable Importance in 1st gradient tree boosting fit")
```
Most important variables are Lifestage (baby >> adult), AnimalType (dog >> cat), and Sex (unknown >> all others). 

### 2nd model

```{r}
###############################
#
# Do NOT run this block of code
#
###############################
xgb_fit2 <- train(
  OutcomeType ~ AnimalType + Lifestage + IsMix + SexuponOutcome + SimpleBreed + SimpleColor,
  data = animals_train,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 10,
    returnResamp = "all",
    index = crossval_train_fold_inds,
    indexOut = crossval_val_fold_inds,
    savePredictions = TRUE
  ),
  tuneGrid = expand.grid(
    nrounds = c(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    eta = c(0.5, 0.6, 0.7), # learning rate; 0.3 is the default
    gamma = 0, # minimum loss reduction to make a split; 0 is the default
    max_depth = 1:5, # how deep are our trees?
    subsample = c(0.5, 0.9, 1), # proportion of observations to use in growing each tree
    colsample_bytree = 1, # proportion of explanatory variables used in each tree
    min_child_weight = 1 # think of this as how many observations must be in each leaf node
  )
)
xgb_fit2$results %>% filter(Accuracy == max(Accuracy))
saveRDS(xgb_fit2, file = "xgb_fit2.rds")
```

```{r}
##############################
#
# Only need to run this block
#
##############################
xgb_fit2 <- readRDS("xgb_fit2.rds")
mean(animals_test[animals_test$SimpleColor != 'Agouti', ]$OutcomeType != predict(xgb_fit2, animals_test[animals_test$SimpleColor != 'Agouti', ]))
```
Test set error rate: 0.3783227 

#### Variable Importance

We look at variable importance for the 2nd gradient tree boosting model as it has a lower test set error rate, therefore a better model and perhaps give more insights than the first one.

Initial plot of variable importance:

```{r}
plot(varImp(xgb_fit2, scale = FALSE), top = 15)
```

Many explanatory variables have a lot of levels, so we thought of summing the variable importance over all levels of each explanatory variable to get the overall importance for the corresponding explanatory variable.

```{r}
# Extract variable importance from varImp
ImpMeasure_xgb_fit2<-data.frame(varImp(xgb_fit2, scale = FALSE)$importance)
# Extract corresponding explanatory variable names of greatest to lowest variable importance
ImpMeasure_xgb_fit2$Vars <- row.names(ImpMeasure_xgb_fit2)
# sum of variable importance over all levels of variable `Sex`
sum_Sex <- sum(ImpMeasure_xgb_fit2[grepl( "Sex" , ImpMeasure_xgb_fit2$Vars ),  ]$Overall)
# sum of variable importance over all levels of variable `SimpleColor`
sum_SimpleColor <- sum(ImpMeasure_xgb_fit2[grepl( "SimpleColor" , ImpMeasure_xgb_fit2$Vars ),  ]$Overall)
# sum of variable importance over all levels of variable `SimpleBreed`
sum_SimpleBreed <- sum(ImpMeasure_xgb_fit2[grepl( "SimpleBreed" , ImpMeasure_xgb_fit2$Vars ),  ]$Overall)
# Adding two corresponding "observations" to ImpMeasure_xgb_fit for (all) `Sex` and (all) `SimpleColor`
ImpMeasure_xgb_fit2 <- rbind(ImpMeasure_xgb_fit2, Sex = c(sum_Sex, 'Sex'))
ImpMeasure_xgb_fit2 <- rbind(ImpMeasure_xgb_fit2, SimpleColor = c(sum_SimpleColor, 'SimpleColor'))
ImpMeasure_xgb_fit2 <- rbind(ImpMeasure_xgb_fit2, SimpleBreed = c(sum_SimpleBreed, 'SimpleBreed'))
# Extract only the explanatory variables used in xgb_fit
ImpMeasure_xgb_fit2 <- ImpMeasure_xgb_fit2[which(ImpMeasure_xgb_fit2$Vars %in% c('Lifestagebaby', 'AnimalTypeDog', 'Sex', 'SimpleColor', 'SimpleBreed', 'IsMix')), ]
# plot variable importance
ggplot(ImpMeasure_xgb_fit2, aes(x=`Vars`, y=Overall)) + 
  geom_point(stat='identity', fill="red", size=2)  +
  geom_segment(aes(y = 0, 
                   x = `Vars`, 
                   yend = Overall, 
                   xend = `Vars`), 
               color = "cornflowerblue") +
  labs(x = "Explanatory variables used in model fit", y = "Variable Importance", title = "Variable Importance in 2nd gradient tree boosting fit")
```
Most important variables are Sex (IntactFemale >> all others), Lifestage (baby >> adult), AnimalType (dog >> cat), and Breed (Pit Bull >> all others). 


### Support Vector Machines

### SVM model

```{r}
###############################
#
# Do NOT run this block of code
#
###############################
y_train <- animals_train$OutcomeType
x_train <- model.matrix(~ 0 + AnimalType + Lifestage + IsMix + SimpleBreed + SexuponOutcome + SimpleColor, data = animals_train)
# Training Support Vector Machines model
svm_model <- svm(x = x_train, y = y_train, data = animals_train)
summary(svm_model)
saveRDS(svm_model, file = "svm_model.rds")
```


```{r}
##############################
#
# Only need to run this block
#
##############################
# define operator opposite to %in% 
'%notin%' <- Negate('%in%')
# Load SVM model from file
svm_model <- readRDS("svm_model.rds")
summary(svm_model)
x_test2 <- model.matrix(~ 0 + AnimalType + Lifestage + IsMix + SimpleBreed + SexuponOutcome + SimpleColor, data = animals_test %>% filter(SimpleColor != "Agouti", SimpleBreed %notin% c('Afghan Hound', 'English Setter', 'Treeing Tennesse Brindle', 'Otterhound', 'Entlebucher', 'Spinone Italiano')))
x_test <- subset(animals_test, select = c(OutcomeType, AnimalType, Lifestage, IsMix, SimpleBreed, SexuponOutcome, SimpleColor)) %>% filter(SimpleColor != "Agouti", SimpleBreed %notin% c('Afghan Hound', 'English Setter', 'Treeing Tennesse Brindle', 'Otterhound', 'Entlebucher', 'Spinone Italiano'))
# test set error rate
mean(x_test$OutcomeType != predict(svm_model, x_test2))
```
Test set error rate: 0.3866917.

### Reference
(1) (Kaggle) Megan Risdal Notebook - Quick and Dirty Random Forest