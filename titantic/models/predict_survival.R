library(imputation)
library(ggplot2)
library(mlbench)
library(caret)

df.train.raw <- read.csv("../data/train.csv")
df.train <- df.train.raw

########################################
# PRE-PROCESS DATA / FEATURE ENGINEERING
########################################

ship.regions <- c("A", "B", "C", "D", "E", "F", "G", "H")
get.region <- function(cabin){
    x1 <- gsub("[0-9]", "", cabin)
    x2 <- gsub(" ", "", x1)
    x2
    substring(x2, 1,1)
}

create.features <- function(df){
    df$region <- with(df, sapply(Cabin, get.region))
    df <- within(df,{
                   has.cabin <- as.numeric((Cabin != ""))
                   irish <- grepl("Mc", Name) | grepl("O\'", Name)
                   married <- grepl("Mrs", Name)
                   name.length <- nchar(as.character(Name))
                   ship.loc <- ifelse(region %in% c("A", "B", "C", "D", "E", "F"), region, "Other")
               })
    df
}


df.train <- create.features(df.train.raw)

###############
# FIT THE MODEL
###############

df.train <- df.train[, c("Survived", "Age", "Fare", "Pclass", "Sex", "has.cabin", "Parch", "SibSp", 
                                      "irish", "name.length", "ship.loc", "Embarked")] 

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

gbmFit1 <- train(Survived ~ ., data = df.train,
                 method = "gbm",
                 trControl = fitControl, verbose = FALSE)

##################
# GET TESTING DATA
##################
df.test <- create.features(read.csv("../data/test.csv"))


######################
# Age imputation model
#######################
m.age <- lm(Age ~ Pclass + Sex +  Fare, data =  merge(df.train, df.test))
df.test$Age.hat <- predict(m.age, newdata = df.test)
df.test$Age <- with(df.test, ifelse(is.na(Age), Age.hat, Age))
df.test$Fare[is.na(df.test$Fare)] <- mean(df.test$Fare, na.rm=TRUE)
df.test$Age.hat <- NULL

summary(df.test)

##################
# Make Predictions 
##################

df.test$Survived <- ifelse(predict(gbmFit1, newdata = df.test) > survival.rate, 1, 0) 
write.csv(df.test[,c("PassengerId", "Survived")], "../predictions/predictions_15AUG13_3.csv", row.names = FALSE)

