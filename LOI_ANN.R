# Set working directory
setwd("C:\\LOI CSV")
getwd() #Directory set OK!

#1 Load dataset
happy_df <- read.csv('HAPPY.csv', stringsAsFactors = FALSE)

#2 Dataset structure
str(happy_df)

#3 Max, min, mean values.
summary(happy_df)

#Check for missing data
any_missing <- any(is.na(happy_df))
cat("Are there any missing values in the dataset? ", any_missing, "\n")

#Remove happy10 variable
happy_df <- happy_df[-8]
str(happy_df)

#Define normalize function
normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}

#Remove happy3 from dataset & Create input dataset
happy_df_input <- happy_df[-3]
str(happy_df_input)

#Normalize input dataset
happy_df_input_norm <- as.data.frame(lapply(happy_df_input, normalize))
summary(happy_df_input_norm)

#Output Not normalized (happy3)
happy_df_output_nn <- happy_df[3]
str(happy_df_output_nn)

#Create CSV file with Input Dataset and Output Not normalized)
write.csv(happy_df_input_norm, "normalized_data.csv", row.names = FALSE)
write.csv(happy_df_output_nn, "excluded_variable.csv", row.names = FALSE)

#Create a factor variable out of a categorical variable.
happy3_factor <- factor(happy_df$happy3)
str(happy3_factor)

#Add factor variable to normalized dataset
dataset <- cbind.data.frame(happy3_factor, happy_df_input_norm)
str(dataset)

#Training and Test dataset
#Make training set
x <- nrow(happy_df_input_norm)
str(x)
set.seed(1234) #Reproducibility of results
train_sample <- sample(x, x*0.8)
str(train_sample)

#Training and Testing dataset
set.seed(1234) #Reproducibility of results
happy_train <- dataset[train_sample,]
happy_test <- dataset[-train_sample,]
#Check training and test dataset!
str(happy_train)
str(happy_test)

#Training the model using factor variable and training dataset
install.packages("nnet")
library(nnet)
install.packages("devtools")
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

set.seed(1234) #Reproducibility of results
happy_model <- nnet(happy3_factor~ ., data = happy_train, maxit=5000, size=5, linout=FALSE)
summary(happy_model)
plot(happy_model)
dev.copy(png,'Happy_NeuralNetwork.png') #We save the plot to working directory!
# dev.copy(png,'comparison18.png') #We save the plot for comparison!
dev.off()

#Evaluate the model
set.seed(1234) #Reproducibility of results
pred <- predict(happy_model, happy_test, type = 'class')
predtable <- table(pred, happy_test$happy3_factor)
predtable

#Confusion Matrix
install.packages("caret")
library(caret)
confusion_matrix <- confusionMatrix(predtable)
confusion_matrix

#Initialize variables to print max accuracy and max size later!
max_accuracy <- 0
max_size <- 0

#Determine optimal size. maxiteration = 5000, size ranging from 1 to 15!
for (x in c(1:15)){
  set.seed(1234) #Reproducibility of results!
  l_model <- nnet(happy3_factor ~., data = happy_train, maxit=5000,size=x, linout=FALSE, trace=FALSE)
  pred <- predict(l_model, happy_test, type = 'class')
  x2 <- table(pred, happy_test$happy3_factor)
  cm <- confusionMatrix(x2)
  cat('Accuracy (size=',x,'):',cm$overall[1],'Kappa:',cm$overall[2],'\n')

  #We impose condition for max accuracy and size!
  if (cm$overall[1] > max_accuracy) {
  max_accuracy <- cm$overall[1]
  max_size <- x
  }
}

#Print size that gives  max accuracy.
cat('Maximum accuracy (size=', max_size, '):', max_accuracy, '\n')

#We train and test the model again w/ 5 nodes. Try different maxit values.
set.seed(1234)
model.final <- nnet(happy3_factor~ ., data = happy_train, maxit=500, size=5, linout=FALSE,trace=TRUE)
model.final.predict <- predict(model.final, happy_test, type = 'class')
mf.table <- table(model.final.predict, happy_test$happy3_factor)
mf.cm    <- confusionMatrix(mf.table); mf.cm