library(imputation)
library(ggplot2)

df.train.raw <- read.csv("../data/train.csv")

predictors <- c("Pclass", "Sex", "Age", "Fare", "PassengerId")
outcome <- c("Survived")
df.train <- df.raw[, c(predictors, outcome)]

survival.rate <- mean(df.train$Survived)

df.train <- na.omit(df.train)

# Fit model 
m <- glm(Survived ~ factor(Pclass)*factor(Sex) + poly(Age, 3) +
         log1p(Fare), data = df, family = "binomial")

# load test data 
df.test.raw <- read.csv("../data/test.csv")
df.test <- df.test.raw[, predictors]
summary(df.test)

# Impute missing data 
m.age <- lm(Age ~ Pclass + Sex +  Fare, data =  merge(df.train, df.test))
df.test$Age.hat <- predict(m.age, newdata = df.test)
df.test$Age <- with(df.test, ifelse(is.na(Age), Age.hat, Age))
df.test$Fare[is.na(df.test$Fare)] <- mean(df.test$Fare, na.rm=TRUE)
df.test$Age.hat <- NULL

summary(df.test)


df.test$Survived <- ifelse(predict(m, newdata = df.test, type = "response") > survival.rate, 1, 0) 


write.csv(df.test[,c("PassengerId", "Survived")], "../predictions/predictions_15AUG13.csv", row.names = FALSE)

