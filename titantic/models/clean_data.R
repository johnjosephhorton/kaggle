library(randomForest)
library(plyr)
library(stringr)
library(ggplot2)
library(Hmisc)

##########
# GET DATA
##########

df.train <- read.csv("../data/train.csv", stringsAsFactors = FALSE)
df.test <- read.csv("../data/test.csv", stringsAsFactors = FALSE)

# set default for outcome in test data; useful for doing model matrix construction later 
df.test$Survived <- 0


#----------
# EDA
#------------

EDA <- FALSE
if(EDA){
ggplot(data = df.train, aes(x = factor(Pclass), y = log1p(Fare), colour=factor(Survived))) +
    geom_violin() +
    facet_wrap(~Sex, ncol = 2) 

with(df.train, table(Sex, I(Fare == 0)))

head(df.train$Name[df.train$Sex == 'male' & df.train$Fare == 0])
}

#-----------------------
# Create data structures 
#-----------------------

df.full <- rbind(df.train.raw, df.test.raw)

# Encode all non-missing categorical variables as factors 
df.train <- within(df.train, {
    Sex <- as.factor(Sex)
    Pclass <- as.factor(Pclass)
})

df.test <- within(df.test, {
    Sex <- as.factor(Sex)
    Pclass <- as.factor(Pclass)
    Embarked <- as.factor(Embarked) 
})


df.full <- rbind(df.train, df.test)

# Impute Age for missing values 
m.age <- lm(Age ~ Fare + Sex + SibSp, data = df.full)

df.train$missing.age <- is.na(df.train$Age)
df.test$missing.age <- is.na(df.test$Age)

df.train$Age[is.na(df.train$Age)] <- predict(m.age, newdata = df.train)[is.na(df.train$Age)]
df.test$Age[is.na(df.test$Age)] <- predict(m.age, newdata = df.test)[is.na(df.test$Age)]

# For missing Fares, just use the median value 
df.train$Fare[is.na(df.train$Fare)] <- median(df.train$Fare, na.rm = TRUE)
df.test$Fare[is.na(df.test$Fare)] <- median(df.test$Fare, na.rm=TRUE)
df.train$Fare <- as.numeric(df.train$Fare)
df.test$Fare <- as.numeric(df.test$Fare)

df.train$Embarked[df.train$Embarked == ""] <- "S"
df.train$Embarked <- as.factor(df.train$Embarked)

# test for no missingness 
dim(na.omit(rbind(df.train, df.test)))
dim(rbind(df.train, df.test))

########################################
# PRE-PROCESS DATA / FEATURE ENGINEERING
########################################

get.sex.name <- function(name, lst){
    any( sapply(lst, function(x) grepl(x, name)) )
}


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
                   cabin.length <- nchar(as.character(Cabin))
                   miss = sapply(df$Name, function(x) get.sex.name(x, c("Miss", "Mlle", "Ms")))
                   mr = sapply(df$Name, function(x) get.sex.name(x, c("Capt","Col","Don","Dr","Jonkheer","Major","Mr","Rev")))
                   mrs = sapply(df$Name, function(x) get.sex.name(x, c("Countess", "Mme", "Mrs")))
                   master <- grepl("Master", Name)
                   irish <- grepl("Mc", Name) | grepl("O\'", Name)
                   name.length <- nchar(as.character(Name))
                   ship.loc <- ifelse(region %in% c("A", "B", "C", "D", "E", "F"), region, "Other")
                   non.numeric.ticket <- is.na(as.numeric(as.character(Ticket)))
               })
    df
}


df.train <- create.features(df.train)

df.test <- create.features(df.test)

#####################
# CREATE MODEL MATRIX 
#####################

formula <- as.formula("Survived ~ Age + Fare + Pclass + has.cabin + Parch +
SibSp + irish + name.length + ship.loc + mr + mrs + miss + master + missing.age + Embarked + non.numeric.ticket + master +
cabin.length")


training <- data.frame( model.matrix(lm(formula, data = df.train)))
training[, "X.Intercept."] <- NULL

training$Survived <- as.factor(with(df.train, ifelse(Survived == 1, "1", "0")))

testing <- data.frame(model.matrix(lm(formula, data = df.test)))
testing[, "X.Intercept."] <- NULL

###############
# FIT THE MODEL
###############

#-----------------
# Tree-based model 
#-----------------

m.rpart <- train(Survived ~ .,
              data = training,
              method = "rpart",
              trControl = fitControl)

m.rpart

summary(m.rpart$finalModel)

post(m.rpart$finalModel, filename = "tree.eps")

#-------------------------
# Gradient-Boosted Methods
#-------------------------

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

gbmGrid <-  expand.grid(.interaction.depth = c(9),  
                         .n.trees = c(100, 500),  
                         .shrinkage = c(0.05, 0.075, 0.1, 0.125)) 

m.gbm <- train(Survived ~ ., data = training,
               method = "gbm",
               trControl = fitControl,
               verbose = FALSE,
               tuneGrid  = gbmGrid
               )

m.gbm
summary(m.gbm)

plot(m.gbm)

#--------------
# Random Forest
#--------------

m.rf <- train(Survived ~ ., data = training,
               method = "rf",
               trControl = fitControl,
               verbose = FALSE
               )

importance(m.rf$finalModel)


m.rf.preproc <- train(Survived ~ ., data = training,
                      method = "rf",
                      trControl = fitControl,
                      preProc = c("center", "scale"), 
                      verbose = FALSE
                      )

summary(m.rf)

#############
# PREDICTIONS 
#############

reify <- function(x) as.numeric(as.character(x))

p.rpart <- reify(predict(m.rpart, newdata = testing))
p.gbm <- reify(predict(m.gbm, newdata = testing))
p.rf <- reify(predict(m.rf, newdata = testing))

results <- data.frame(cbind(p.rpart, p.gbm, p.rf))
results$PassengerId <- df.test$PassengerId

results$Survived <- with(results, round((p.rpart + p.rf + p.gbm)/3))

write.csv(results[,c("PassengerId", "Survived")], "../predictions/predictions_16AUG13_7.csv", row.names = FALSE)


df.test$Survived <- predict(m.gbm, newdata = testing)

write.csv(df.test[,c("PassengerId", "Survived")], "../predictions/predictions_15AUG13_6.csv", row.names = FALSE)

