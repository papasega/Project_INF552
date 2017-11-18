train = read.csv("../Data/train.csv")
test = read.csv("../Data/test.csv")
# Add all the rows of test into train. This is to make sure that
# factor variables will be interpreted together so as to give correct
# factor levels.
target = rep(0,nrow(test))
test = cbind(target,test)
heap = rbind(train,test)

# All -1 in the original data are invalid
heap = as.matrix(heap)
heap[which(heap==-1)] = NA
heap = as.data.frame(heap)

# Correctly encode categorical variables
is_cat = function(x) return( (grepl("cat",x)) | (grepl("bin",x)) )
cat_indices = which(is_cat(colnames(heap)))
heap[,cat_indices] = lapply(heap[,cat_indices],as.factor)
# Reseparate training set and testing set
train = heap[1:nrow(train),]
range_test = (nrow(train)+1):nrow(heap)
test = heap[range_test,c(-2)]

# Delete columns with a lot of missing data. Then delete all missing data.
train = train[,c(-26,-28)]
test = test[,c(-25,-27)]
train = na.omit(train)

replace_NA = function(x,value) {
  # replacing all NA in x with value, where x is a column
  if (sum(is.na(x))==0) return(x)
  x[which(is.na(x))] = value
  return(x)
}

# Replacing all NA in test with its mean or its mode FROM TRANING
for (i in 2:ncol(test)) {
  if (is.factor(test[,i])) {
    mode_y = names(which.max(summary(train[,i+1])))[1]
    test[,i] = replace_NA(test[,i],mode_y)
  }
  else {
    mean_y = mean(train[,i+1])
    test[,i] = replace_NA(test[,i],mean_y)
  }
}

# Test data have new factor levels (bizarre...), so
# we remove all these new levels. WARNING: This code may cause error...
a = which((test$ps_car_11_cat==25)|(test$ps_car_11_cat==80))
test$ps_car_11_cat[a] = 1

#Data preprocessing completed.

# Learning by simple logistic regression
model = glm(target~.,family=binomial(link='logit'),data=train[,c(-1)])
result = predict(model,newdata=test[,c(-1)],type='response')
df = data.frame(test$id,result)
colnames(df) = c("id","target")
write.csv(df,"../Data/sub_Logistics_naive.csv",row.names=FALSE)
