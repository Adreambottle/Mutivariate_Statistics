# Author: Daniel Yanan ZHOU - 115010302@link.cuhk.edu.cn 


library(lattice)
path <- "/Users/meron/Desktop/MTB/Tutorial/04/wine.csv"
my.wines <- read.csv(path, header=TRUE)
my.wines
my.wines.data = my.wines[,-1]
summary(my.wines.data)


# Look at the correlations
data.cov = cov(my.wines.data)
data.cor = cor(my.wines.data, method="pearson")

data.cov
data.cor


library(gclus)
my.abs     <- abs(cor(my.wines.data))              # absolute value of the wine data 
my.colors  <- dmat.color(my.abs)                   # Colors a symmetric matrix  改变颜色
my.ordered <- order.single(cor(my.wines.data))     # Orders objects using hierarchical clustering
cpairs(my.wines.data, my.ordered, panel.colors=my.colors, gap=0.5)


# Calculating eigenvalue
eigen(data.cov)
eigen(data.cor)

lambdas = eigen(data.cor)$values                   # eigenvalues of the origin data
Alpha = eigen(data.cor)$vectors                    # eigenvactors of the origin data
cumulative.lambda = cumsum(lambdas)                # cumulative sum of the eigenvalues
cumulative.lambda
lambdas/sum(diag(data.cor))                        # proportion of each PC/eigenvalue
cumulative.lambda/sum(diag(data.cor))              # cumulative sum of the proportion

            
# Do the PCA 


my.prc <- prcomp(my.wines.data, center=TRUE, scale=TRUE)   # PCA function

screeplot(my.prc, main="Scree Plot", xlab="Components")
screeplot(my.prc, main="Scree Plot", type="line" )


# DotPlot PC1

load            <- my.prc$rotation                # `load` returns the rotations/PCs 
sorted.loadings <- load[order(load[, 1]), 1]      # `order` returns the order of the vector 
myTitle         <- "Loadings Plot for PC1" 
myXlab          <- "Variable Loadings"
dotplot(sorted.loadings, main=myTitle, xlab=myXlab, cex=1.5, col="red")

# DotPlot PC2

sorted.loadings <- load[order(load[, 2]), 2]
myTitle <- "Loadings Plot for PC2"
myXlab  <- "Variable Loadings"
dotplot(sorted.loadings, main=myTitle, xlab=myXlab, cex=1.5, col="red")


# Reform the data
PC1 = my.prc$rotation[, 1]
PC2 = my.prc$rotation[, 2]

data1 = as.matrix(my.wines.data) %*% as.matrix(PC1)
data2 = as.matrix(my.wines.data) %*% as.matrix(PC2)

plot(data1, data2, cex=1.5, col='red', pch=16)


# Now draw the BiPlot
biplot(my.prc, cex=c(1, 0.7))
