#MASHABLE DATA...ONLINE MEDIA POPULARITY 
rm(list = ls())
library(tidyverse)
library(leaps)
library(glmnet)
library(glmnetUtils)
library(ElemStatLearn)
library(partykit)
library(magrittr)

#WTF does LDA mean?
#: the Latanet Dirichlet Allocation Algorithm was applied to all mashable texts in order to identify the top five most 
#relevant topics...then measure the closeness of each article to all top 5 topics in a percentage...all LDA's add up to 1

#Read-in Data and Load Libraries into current session
raw_data <- read.csv("~/Desktop/MGSC 310/MGSC 310 Final Project/OnlineNewsPopularityData.csv")
sapply(raw_data,is.numeric)
#All variables were conveniently numeric and binary dummy variables were already created

#******************** REMOVE NON-PREDICTIVE VARIABLES FOR VAR SELECTION********************
raw_data <- select(raw_data,-c(url,timedelta))
summary(raw_data$shares)
p10 <- ggplot(data = raw_data, aes(x=shares)) + geom_histogram(color = "green") + scale_x_continuous(limits = c(0,10000)) + 
  scale_y_continuous(limits = c(0,10000)) + labs(title = "Histogram: Shares")

plot(p10)



sd(raw_data$shares)
var(raw_data$shares)
#Make it a classification problem because it's far too difficult to predict actual number of shares for an individual article
raw_data <- raw_data %>% mutate(popular_article = ifelse(shares > 1400,1,0))

p11 <- ggplot(data = raw_data) + 
  stat_summary(
    mapping = aes(x = popular_article, y = shares),
    fun.ymin = min,
    fun.ymax = max,
    fun.y = median
  )
plot(p11)

#Take out "shares"...so we can iterate over whole data set without any problems
binomial_data <- select(raw_data, -c(shares))

#Here we decided to apply our variable selection methods using the binary/binomial predictor "popular_article".
#This is due to the fact that when we tried to run our variable selection methods predicting actual count/number of shares, often times
#the best methods (ridge, lasso, elastic-net) did not select any of our 58 predictor variables. We figured that this must be because of how
#difficult it would actually be to predict the exact number of shares for any given web article. 
#We re-ran our variable-selection tests using our binomial training set which contains only binomial target ('popular_article')...
#and all of our 58 potential predictors

#Avoid dealing with class-imbalance problems...go with binary/classification problem

#********************TEST AND TRAINING SPLIT********************
#We decided to continue with an exclusively logistic model 
set.seed(1861)
train_idx <- sample(1:nrow(binomial_data), size = floor(nrow(binomial_data) * .75))
binomial_train <- binomial_data %>% slice(train_idx)
binomial_test <- binomial_data %>% slice(-train_idx)




#********************CORRELATION MATRIX TO DETECT COLINEARITY********************
cor_matrix <- cor(binomial_train)
cor_matrix <- data.frame(round(cor_matrix, 3))
library(corrplot)


devtools::install_github("vsimko/corrplot")
corrplot(cor_matrix, type = "upper", tl.pos = "td",
         method = "circle", tl.cex = 0.5, tl.col = 'black',
         order = "hclust")



#Variance Inflation Factor (VIF)...anything over like 5 or 10 is problematic (regular OLS model)
library(olsrr)

lm_mod1 <- lm(shares ~.,
           data = raw_data)
summary(lm_mod1)
VIF_1 <- ols_vif_tol(lm_mod1)
VIF_1

#********************VARIABLE SELECTION METHODS********************

#Ridge-regression: Not using...the severity on the penalty term (lambda) is not strict enough...and in this case...
#not every variable matters a little...only a few matter a lot

#ridge_mod <- cv.glmnet(popular_article ~. ,
                       #data = binomial_train,
                       #alpha = 0)
#coef(ridge_mod)

#LASSO model: More Severe Penalty Term...using this because we are going for a more parsimonious/simple model
#For LASSO: the coefficients which are responsible for large variance are converted to zero
lasso_mod <- cv.glmnet(popular_article ~. ,
                       data = binomial_train,
                       family = 'binomial',
                       alpha = 1)
coef(lasso_mod)
plot(lasso_mod) 
#Coefficient table at lambda.min and lambda.1se for LASSO 
library(data.table)
lasso_coefs <- data.table(
  varnames = rownames(coef(lasso_mod, s = "lambda.min")),
  lasso_lambda_min = round(as.matrix(coef(lasso_mod, s = "lambda.min")), digits = 3), lasso_lambda_1se = round(as.matrix(coef(lasso_mod, s = "lambda.1se")), digits = 3)) 
print(lasso_coefs)

#Elastic Net Model
alpha_grid <- seq(0,1,len = 11)
alpha_grid

enet_fit <- cva.glmnet(popular_article ~. ,
                       data = binomial_train,
                       alpha = alpha_grid)

#plot
minlossplot(enet_fit)
plot(enet_fit)
#Min CV loss at .7
#matrix of coefficients at each alpha
enet_coefs <- data.frame(
  varname = rownames(coef(enet_fit,alpha = 0)),
  ridge = as.matrix(coef(enet_fit, alpha = 0)) %>% round(10),
  alpha01 = as.matrix(coef(enet_fit, alpha = 0.1)) %>% round(10),
  alpha02 = as.matrix(coef(enet_fit, alpha = 0.2)) %>% round(10),
  alpha03 = as.matrix(coef(enet_fit, alpha = 0.3)) %>% round(10),
  alpha04 = as.matrix(coef(enet_fit, alpha = 0.4)) %>% round(10),
  alpha05 = as.matrix(coef(enet_fit, alpha = 0.5)) %>% round(10),
  alpha06 = as.matrix(coef(enet_fit, alpha = 0.6)) %>% round(10),
  alpha07 = as.matrix(coef(enet_fit, alpha = 0.7)) %>% round(10),
  alpha08 = as.matrix(coef(enet_fit, alpha = 0.8)) %>% round(10),
  alpha09 = as.matrix(coef(enet_fit, alpha = 0.9)) %>% round(10),
  lasso = as.matrix(coef(enet_fit, alpha = 1)) %>% round(10)
) %>% rename(varname = 1, ridge = 2, alpha01 = 3, alpha02 = 4, alpha03 = 5, alpha04 = 6,
             alpha05 = 7, alpha06 = 8, alpha07 = 9, alpha08 = 10, alpha09 = 11, lasso = 12) %>% 
  remove_rownames()
head(enet_coefs)

#FOR COUNTING PURPOSES
#final_vars <- select(enet_coefs, c(varname, alpha07))
#dim(final_vars)
#final_vars <- final_vars %>% filter(alpha07 != 0)
#dim(final_vars)
#final_vars$varname
#********************SUMMARY STATS AND PLOTS AFTER VARIABLE SELECTION********************

#FINAL DATASET
#2 Potential Target Variables (shares & popular_article)
#21 Predictor Variables
#LDA: Latent Dirichlet Allocation Algorithm (LDA) was applied prior to downloading .csv file
#Identifies top 5 most relevant topics and then measures the closeness of each article to each relevant topic
#LDA_00 to LDA_04 (Didn't give specific topics)
#LDA's add up to 1 across a single observation/row


clean_data <- select(raw_data, c(popular_article,
                                 shares,
                                 num_hrefs,
                                 num_self_hrefs,
                                 num_keywords,
                                 data_channel_is_entertainment,
                                 data_channel_is_bus,
                                 data_channel_is_socmed,
                                 data_channel_is_tech,
                                 weekday_is_tuesday,
                                 weekday_is_wednesday,
                                 weekday_is_friday,
                                 weekday_is_saturday,
                                 is_weekend,
                                 global_subjectivity,
                                 min_positive_polarity,
                                 title_subjectivity,
                                 title_sentiment_polarity,
                                 abs_title_subjectivity,
                                 LDA_00,
                                 LDA_01,
                                 LDA_02,
                                 LDA_04))
summary(clean_data)

#Put it into data frame bc why not...
sum_stats <- data.frame(summary(clean_data))
sum_stats

#Group-by function to spit out summary stats of our selected variables...grouping by popular (1) vs unpopular (0)
by_popular <- clean_data %>% group_by(popular_article)
by_popular <- by_popular %>% summarise_all(list(min = min, 
                                                mean = mean, 
                                                median = median,
                                                max = max,
                                                sd = sd), na.rm = TRUE)
by_popular

#test and training split pt.2
set.seed(1861)
train_idx <- sample(1:nrow(clean_data), size = floor(nrow(clean_data) * .75))
mash_train <- clean_data %>% slice(train_idx)
mash_test <- clean_data %>% slice(-train_idx)


#Scatter of shares vs global subjectivity
p1 <- ggplot(mash_train, aes(x = global_subjectivity, y = shares)) + geom_point(mapping = aes(color = popular_article)) + 
  labs(x = "Global Subjectivity)", y = "Shares", title = "Shares vs Global Subjectivity")
plot(p1)
#This is interesting bc the articles with the most amount of shares appear to have the highest concetration arround a global subjectivity level of .5...Neutral overall text subjectivity

#Ridgline Density 
library(ggridges)
data_forplot <- mash_train %>% mutate(popular_article2 = ifelse( popular_article == 1,"Yes",
                                                                  "No"))
p2 <- ggplot(data_forplot, aes(x = num_hrefs, y = popular_article2, fill = popular_article2)) + 
  geom_density_ridges() + 
  scale_x_continuous(limits = c(0,60)) + 
  labs(x = "Number of Hrefs", y = "Popular Article", title = "Ridgeline Density Plot") 
plot(p2)
#more popular articles have a greater number of links in them

p3 <- ggplot(data_forplot, aes(x = is_weekend, y = popular_article2, fill = popular_article2)) + 
  geom_density_ridges() + 
  scale_x_continuous(limits = c(0,1)) + 
  labs(x = "Is Weekend", y = "Popular Article", title = "Ridgeline Density Plot") 
plot(p3)
#This plot is interesting because it appears that a larger proportion of popular articles were released during the weekend


p4 <- ggplot(data = mash_train) + 
  stat_summary(
    mapping = aes(x = popular_article, y = title_subjectivity),
    fun.ymin = min,
    fun.ymax = max,
    fun.y = median
  )
plot(p4)
#This plot highlights the difference in title subjectivity (the extent to which titles were based on personal sentiment/opinion) between popular vs non-popular articles

p5 <- ggplot(data = mash_train, aes(x=shares)) + geom_histogram(color = "green") + scale_x_continuous(limits = c(0,10000)) + 
  scale_y_continuous(limits = c(0,10000)) + labs(title = "Histogram: Shares")
plot(p5)


p6 <- ggplot(mash_train, aes(x = title_sentiment_polarity, y = shares)) + geom_point(mapping = aes(color = popular_article)) + 
  labs(x = "Title Polarity", y = "Shares", title = "Shares vs Title Polarity")
plot(p6)
#This plot is interesting because it shows that articles with neutral titles (title polarity of 0) seem to have the most amount of shares.

p7 <- ggplot(mash_train, aes(x = num_keywords, y = shares)) + geom_point(mapping = aes(color = popular_article)) + 
  labs(x = "Number of Keywords in Metadata", y = "Shares", title = "Shares vs Number of Keywords")
plot(p7)
#This plot is interesting because it seems that articles with more than 5 keywords have more shares and are more popular.


#the plots above and their variables were selected based off of the summary statistics we ran because we wanted to highlight what 
#we might find to be the key differences/key variables in predicting either the number of shares (OLS model) or whether or not an article
#would be popular or not (logistic model). 

#Further, we wanted to create a historgram/simple count of our number of shares to visualize the distrinution of the data. It appears to be skewed to the right.
#So, if we do decide to clean the data further and remove outliers, we may not be able to utilize z-scores because the data is not 
#normally distributed.

#***********************FIRST PREDICTIVE MODELS***********************

#OLS: NOT USING




ols_mod_train <- lm(shares ~ num_hrefs +
              num_self_hrefs +
              num_keywords +
              data_channel_is_entertainment +
              data_channel_is_bus +
              data_channel_is_socmed +
              data_channel_is_tech +
              weekday_is_tuesday +
              weekday_is_wednesday +
              weekday_is_friday +
              weekday_is_saturday +
              is_weekend +
              global_subjectivity +
              min_positive_polarity +
              title_subjectivity +
              title_sentiment_polarity +
              abs_title_subjectivity +
              LDA_00 +
              LDA_01 +
              LDA_02 +
              LDA_04,
              data = mash_train)
summary(ols_mod_train)
#Adjusted R-Squared is really low...this model is terrible



#Logistic: USING
logit_mod_train <- glm(popular_article ~ num_hrefs +
     num_self_hrefs +
     num_keywords +
     data_channel_is_entertainment +
     data_channel_is_bus +
     data_channel_is_socmed +
     data_channel_is_tech +
     weekday_is_tuesday +
     weekday_is_wednesday +
     weekday_is_friday +
     weekday_is_saturday +
     is_weekend +
     global_subjectivity +
     min_positive_polarity +
     title_subjectivity +
     title_sentiment_polarity +
     abs_title_subjectivity +
     LDA_00 +
     LDA_01 +
     LDA_02 +
     LDA_04,
     family = binomial,
   data = mash_train)

summary(logit_mod_train)
library(jtools)
summ(logit_mod_train, exp = TRUE)


#***********************Predictions for OLS and Logit Mods***********************
preds_DF_train <- data.frame(
  preds_ols = predict(ols_mod_train, data = mash_train),
  scores_logit = predict(logit_mod_train, data = mash_train, type = "response"),
  mash_train)
preds_DF_train[1:5,1:4]

preds_DF_train <- preds_DF_train %>% mutate(
  resids_ols = preds_DF_train$shares-preds_DF_train$preds_ols
)


#***********************Plots to show if heteroskedastic*********************** 
ggplot(preds_DF_train, aes(x = preds_ols, y = resids_ols)) + geom_point() + scale_x_continuous(limits = c(-1, 10000)) + 
  labs(title = "Heteroscedastic?", x = "Predictions OLS", y = "Residuals")

plot(ols_mod_train)


#Breusch-Pagan Test For Formal Analysis
library(lmtest)
#Sig. Threshold: p<.05
#Ho (Null): Our model is Homoskedastic
#Ha (Alternative): Our model is Heteroskedastic
bptest(ols_mod_train)
#p-value = 0.1932...fail to reject null hypothesis...model appears homoskedastic...we've eliminated a lot of the noise through our
#variable selection methods


#True Plot... actual observations against residuals
#ggplot(preds_DF_train, aes(x = shares, y = preds_ols)) + geom_point() + labs("True Plot", x = "Actual Shares", y = "Predictions")

#***********************ROC Plots***********************
#PART F: Predictions in the TEST Set...generating scores for logistic model only not OLS 


preds_DF_test <- data.frame(
  scores_logit = predict(logit_mod_train, newdata = mash_test, type = "response"),
  mash_test)

preds_DF_train <- preds_DF_train %>% mutate(PosNeg05 = 
                                              ifelse(scores_logit > 0.5 & popular_article == 1, "TP",
                                                     ifelse(scores_logit > 0.5 & popular_article == 0, "FP",
                                                            ifelse(scores_logit <= 0.5 & popular_article == 0, "TN",
                                                                   ifelse(scores_logit <= 0.5 & popular_article == 1, "FN",0)))))
preds_DF_test <- preds_DF_test %>% mutate(PosNeg05 = 
                                            ifelse(scores_logit > 0.5 & popular_article == 1, "TP",
                                                   ifelse(scores_logit > 0.5 & popular_article == 0, "FP",
                                                          ifelse(scores_logit <= 0.5 & popular_article == 0, "TN",
                                                                 ifelse(scores_logit <= 0.5 & popular_article == 1, "FN",0)))))

 

# Confusion Matrices for training and test sets

##For Training Set---------------------------------
preds_DF_train <- data.frame(class_pred05 = ifelse(preds_DF_train$scores_logit
                                                >.5,"Predicted Yes","Predicted No"), preds_DF_train)
matrix_train <- table(preds_DF_train$class_pred05,
                      preds_DF_train$popular_article)
matrix_train

train_accuracy <- data.frame(
  sensi_TPrate = matrix_train [2,2]/(matrix_train[2,2]+matrix_train[1,2]),
  speci_TNrate = matrix_train [1,1]/(matrix_train[1,1]+matrix_train[2,1]),
  FP_rate = matrix_train[2,1]/(matrix_train[1,1]+matrix_train[2,1]),
  accuracy = (matrix_train[2,2]+matrix_train[1,1])/nrow(preds_DF_train)
)
train_accuracy
#Accuracy is TP+TN/Total Negative + Total Positive...which is just the dataset total essentially

##For Test Set------------------------------------
preds_DF_test <- data.frame(class_pred05 = ifelse(preds_DF_test$scores_logit
                                               >.5,"Predicted Yes", "Predicted No"), preds_DF_test)
matrix_test <- table(preds_DF_test$class_pred05,
                     preds_DF_test$popular_article)
matrix_test
#Columns...whether or not it ACTUALLY was a PriceyHome
#Rows...whether or not we PREDICTED it was a PriceyHome

test_accuracy <- data.frame(
  sensi_TPrate = matrix_test[2,2]/(matrix_test[2,2]+matrix_test[1,2]),
  speci_TNrate = matrix_test [1,1]/(matrix_test[1,1]+matrix_test[2,1]),
  FP_rate = matrix_test[2,1]/(matrix_test[1,1]+matrix_test[2,1]), 
  accuracy = (matrix_test[2,2]+matrix_test[1,1])/nrow(preds_DF_test)
)
test_accuracy


#PART I: ROC plot
library('plotROC')


#Training Set
train_ROC <- ggplot(preds_DF_train,
                    aes(m = scores_logit,
                        d = popular_article)) + 
  geom_roc(labelsize = 3.5,
           cutoffs.at = c(.99,.9,.7,.6,.5,.4,.3,.1,0)) + labs(title = "Training Set")
plot(train_ROC)

#Test Set
test_ROC <- ggplot(preds_DF_test, 
                   aes(m = scores_logit, 
                       d = popular_article)) +
  geom_roc(labelsize = 3.5,
           cutoffs.at = c(.99,.9,.7,.6,.5,.4,.3,.1,0)) + labs(title = "Test Set")
plot(test_ROC)


#PART J: AUC
calc_auc(train_ROC)

calc_auc(test_ROC)

#In conclusion the logistic model didn't work very well, with AUC of .67 for both the test and training sets...
#We should attempt other methods...an AUC from .7-.8 would be considered acceptable...so this is relatiely low and not a whole whole whole whole lot better than chance
#The model is also slightly overfit since the test AUC is better than the training AUC 


#***********************Predictions for Lasso and Elastic-Net Models***********************

#Decided to go with Lasso because even after removing multi-colliniearity...we were still weary of having too highly correlated 
#predictors
#Often times if this is the case...elastic net will outperform lasso



#Training at lambda.min and alpha = .7
enet_preds_train <- data.frame(preds = predict(enet_fit,
                                                newdata = binomial_train,
                                                s = "lambda.min",
                                                alpha = .7))
enet_preds_train <- rename(enet_preds_train, preds_enet_train = X1) 
head(enet_preds_train)

#Test at lambda.min and alpha = .7
enet_preds_test <- data.frame(preds = predict(enet_fit,
                                                newdata = binomial_test,
                                                s = "lambda.min",
                                                alpha = .7))
enet_preds_test <- rename(enet_preds_test, preds_enet_test = X1) 
head(enet_preds_test)


binomial_train <- cbind.data.frame(enet_preds_train, binomial_train)
binomial_test <- cbind.data.frame(enet_preds_test,binomial_test)

train_ROC_enet <- ggplot(binomial_train, 
                   aes(m = preds_enet_train, 
                       d = popular_article)) +
  geom_roc(labelsize = 3.5,
           cutoffs.at = c(.99,.9,.7,.6,.5,.4,.3,.1,0)) + labs(title = "Train Elastic Net")
plot(train_ROC_enet)
calc_auc(train_ROC_enet)


#AUC is slightly better in this model...elastic net model...but only by about 2%

#***********************Random Forrests***********************
#The fundamental difference is that in Random forests, 
#only a subset of features are selected at random out of the total and the best split feature from the subset is used to split each node in a tree, 
#unlike in bagging where all features are considered for splitting a node.

#We used feature selection methods to remove variables that lacked statistical significance, exhibited collinearity, and that were confounding

library('randomForest')
library(randomForestExplainer)

binomial_train <- select(binomial_train,-c(preds_enet_train))


rf_fit <- randomForest(popular_article ~ . ,
                       data = binomial_train,
                       type = classification,
                       mtry = 3,
                       ntree = 500,
                       importance = TRUE,
                       localImp = TRUE)
rf_fit
plot(rf_fit, xlim = c(0,500), ylim = c(0,1))


varImpPlot(rf_fit)

importance(rf_fit)

plot_min_depth_distribution(rf_fit)

mash_test <- select(mash_test,-c(shares))
binomial_test <- select(binomial_test,-c(preds_enet_test))
rf_fit2 <- randomForest(popular_article ~ . ,
                        data = mash_train,
                        type = classification,
                        mtry = 3,
                        ntree = 500,
                        importance = TRUE,
                        localImp = TRUE)
rf_fit2
plot(rf_fit2, xlim = c(0,500), ylim = c(0,1))










