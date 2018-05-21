library(tidyverse)
library(h2o) # for H2O Machine Learning
library(lime) # for Machine Learning Interpretation
library(mlbench) # for Datasets


# Your lucky seed here ...
n_seed = 12345


data <- read.csv("stackoverflow.csv")
data %>% as_tibble()

# Is there any salary difference if you are in remote?
data %>% filter(Data.scientist) %>% group_by(Country) %>%  summarise(Remote_ratio=mean(Remote=="Remote"))

# How many data scientist filled the survey grouped by country?
data %>% filter(Data.scientist) %>% group_by(Country) %>% summarise(Count = n()) %>% ggplot(aes(x=Country, y=Count)) + geom_bar(stat = "identity") + theme_minimal()

# How many are them work from home?
data %>% filter(Data.scientist) %>% group_by(Country) %>%  summarise(Remote_ratio=mean(Remote=="Remote"))




glimpse(data)

data <- data %>% select(-Respondent)

target = "Salary" 
features = setdiff(colnames(data), target)
print(features)

# Start a local H2O cluster (JVM)
h2o.init()

h_data <- as.h2o(data)

# Now we have an H2O dataframe
str(h_data)

# Split Train/Test
h_split = h2o.splitFrame(h_data, ratios = 0.75, seed = n_seed)
h_train = h_split[[1]] # 75% for modelling
h_test = h_split[[2]] # 25% for evaluation


# Train a Default H2O GBM model
model_gbm = h2o.gbm(x = features,
                    y = target,
                    training_frame = h_train,
                    model_id = "my_gbm",
                    seed = n_seed)
print(model_gbm)


# Evaluate performance on test
h2o.performance(model_gbm, newdata = h_test)


## ------------------------------------------------------------------------------


model_automl = h2o.automl(x = features,
                          y = target,
                          training_frame = h_train,
                          nfolds = 5,               # Cross-Validation
                          max_runtime_secs = 120,   # Max time
                          max_models = 100,         # Max no. of models
                          stopping_metric = "RMSE", # Metric to optimize
                          project_name = "my_automl",
                          exclude_algos = NULL,     # If you want to exclude any algo 
                          seed = n_seed)


model_automl@leaderboard
h2o.performance(model_automl@leader, newdata = h_test) # lower RMSE = better



##### Let's try predicting the remote status, and we need to dael with the unbalanced data


target = "Remote" 
features = setdiff(colnames(data), target)
print(features)

# Start a local H2O cluster (JVM)
h2o.init()

h_data <- as.h2o(data)

# Now we have an H2O dataframe
str(h_data)

# Split Train/Test
h_split = h2o.splitFrame(h_data, ratios = 0.75, seed = n_seed)
h_train = h_split[[1]] # 75% for modelling
h_test = h_split[[2]] # 25% for evaluation


# Train a Default H2O GBM model
model_gbm2 <- h2o.gbm(x = features,
                    y = target,
                    training_frame = h_train,
                    model_id = "my_gbm2",
                    balance_classes = TRUE,
                    seed = n_seed)
print(model_gbm2)


# Evaluate performance on test
h2o.performance(model_gbm2, newdata = h_test)





model_automl <-  h2o.automl(x = features,
                          y = target,
                          training_frame = h_train,
                          nfolds = 5,               # Cross-Validation
                          max_runtime_secs = 120,   # Max time
                          max_models = 100,         # Max no. of models
                          stopping_metric = "AUC", # Metric to optimize
                          project_name = "my_automl2",
                          exclude_algos = NULL,     # If you want to exclude any algo 
                          seed = n_seed)


model_automl@leaderboard
h2o.performance(model_automl@leader, newdata = h_test) # lower RMSE = better
