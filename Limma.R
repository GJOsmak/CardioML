#data = read.csv('../../2020/cardiomyopathy/git/train_GSE36961.csv', sep = ',', row.names = 1)
#target = read.csv('../../2020/cardiomyopathy/git/train_GSE36961_target.csv', sep = ',', row.names = 1)

library(dplyr)
library(limma)


normalize_and_scale <- function (X_data) {
    #
    # INPUT: features are columns
    #
    
# > matrix(c(1,2,3,4), nrow =2, ncol=2)
#     [,1] [,2]
# [1,]    1    3
# [2,]    2    4
# > apply(matrix(c(1,2,3,4), nrow =2, ncol=2), 1, mean)
# [1] 2 3
# > apply(matrix(c(1,2,3,4), nrow =2, ncol=2), 2, mean)
# [1] 1.5 3.5
    
    X_cols = colnames(X_data)
    X_rows = row.names(X_data)
    
    # L2 norm: sqrt(sum(vec^2))
    # By rows, i.e., by samples
    X_data = apply(X_data, 1, function (x) x / sqrt(sum(x^2)))
    
    # Standard scale: scale
    # By columns, i.e., by features
    mean_X = mean(X_train, 2)
    sd_X = sd(X_train, 2)
    X_data = scale(X_data, mean_X, sd_X)
    
    colnames(X_data) = X_cols
    row.names(X_data) = X_rows
                   
    X_data
}


# Chained Normalizer and StandardScaler from scikit-learn
normalize_and_scale_all <- function (X_train, X_test) {
    #
    # INPUT: features are columns
    #
    
# > matrix(c(1,2,3,4), nrow =2, ncol=2)
#     [,1] [,2]
# [1,]    1    3
# [2,]    2    4
# > apply(matrix(c(1,2,3,4), nrow =2, ncol=2), 1, mean)
# [1] 2 3
# > apply(matrix(c(1,2,3,4), nrow =2, ncol=2), 2, mean)
# [1] 1.5 3.5
    
    X_train_cols = colnames(X_train)
    X_test_cols = colnames(X_test)
    X_train_rows = row.names(X_train)
    X_test_rows = row.names(X_test)
    
    # L2 norm: sqrt(sum(vec^2))
    # By rows, i.e., by samples
    X_train = apply(X_train, 1, function (x) x / sqrt(sum(x^2)))
    X_test = apply(X_test, 1, function (x) x / sqrt(sum(x^2)))
    
    # Standard scale: scale
    # By columns, i.e., by features
    mean_X = mean(X_train, 2)
    sd_X = sd(X_train, 2)
    X_train = scale(X_train, mean_X, sd_X)
    X_test = scale(X_test, mean_X, sd_X)
    
    colnames(X_train) = X_train_cols
    colnames(X_test) = X_test_cols
    row.names(X_train) = X_train_rows
    row.names(X_test) = X_test_rows
                   
    list(train = X_train, test = X_test)
}


downsample <- function (X_data, y_data, data_size = 0.8, random_state = 42) {
    set.seed(random_state)
    
    new_inds = sample(nrow(X_data), size = data_size * nrow(X_data), replace = TRUE)   
    
    list(X = X_data[new_inds,] y = y_data[new_inds])
}
                   

fit_all_loops <- function (X_data, y_data, n_ext_iter) {
    for (i_iter in 1:n_ext_iter) {
        cat("[", i_iter, "/", n_ext_iter, "] R: external loop fitting...\n")
        
        # Downsample
        X_data = downsample(X_data, y_data, data_size = 0.8, random_state = i_iter)
        
        # Normalize and scale
        X_data = normalize_and_scale(X_data)
        
        # Get significant features
        res = limma_anal(X_data, y_data)
    }
}


limma_anal = function(data, target, pvalue=0.05, foldChange=0.5) {
    #
    # INPUT: features are columns
    #
    
      #data = dataframe with row = genes, colums = samples
      #target = dataframe with one column = target
  
      target = factor(target[,1])
      data_log_norm = limma::normalizeBetweenArrays(log(data))
      
      design <- model.matrix(~ 0 + target)
      
      fit <- lmFit(data_log_norm, design)
      cont.matrix<-makeContrasts(vs1=target0 - target1, levels = design)
      fit2=contrasts.fit(fit,cont.matrix)
      fit2 <- eBayes(fit2)
      diff = topTable(fit2, coef=1, n=Inf, adjust="BH")
      
      diffSig = diff[(diff$adj.P.Val < padj & (diff$logFC>foldChange | diff$logFC<(-foldChange))),]
      
      return(diffSig)
}
