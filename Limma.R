library(limma)
library(dplyr)

#data = read.csv('../../2020/cardiomyopathy/git/train_GSE36961.csv', sep = ',', row.names = 1)
#target = read.csv('../../2020/cardiomyopathy/git/train_GSE36961_target.csv', sep = ',', row.names = 1)


limma_anal = function(data, target, pvalue=0.05, foldChange=0.5){
      
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
