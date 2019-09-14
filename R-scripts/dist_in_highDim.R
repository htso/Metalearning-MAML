# dist_in_highDim.R -- Curse of Dimensionality, or explore the properties of random points in high dim space
# Aug 22, 2018
# (c) Horace W. Tso
# 
# Ref :
# https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions
#
# https://bib.dbvis.de/uploadedFiles/155.pdf
#
# https://mathoverflow.net/questions/128786/history-of-the-high-dimensional-volume-paradox/128881#128881
#
# On curse of dimensionality ....
#
# "[O]ur intuitions, which come from a three-dimensional world, often do not apply in high-dimensional
# ones. In high dimensions, most of the mass of a multivariate Gaussian distribution is not near 
# the mean, but in an increasingly distant “shell” around it; and most of the volume of a 
# high-dimensional orange is in the skin, not the pulp. If a constant number of examples is 
# distributed uniformly in a high-dimensional hypercube, beyond some dimensionality most examples 
# are closer to a face of the hypercube than to their nearest neighbor. And if we approximate 
# a hypersphere by inscribing it in a hypercube, in high dimensions almost all the volume of 
# the hypercube is outside the hypersphere. This is bad news for machine learning, where shapes 
# of one type are often approximated by shapes of another."
# 
# 

library(goodies)
library(doParallel)
library(foreach)

cl = makeCluster(3)
registerDoParallel(cl)

nr = 1000 # number of obs
nc = 1000 # dim
# Obs matrix
x = matrix(rnorm(nr*nc, mean=0, sd=1.0), nrow = nr, ncol=nc)
# euclidean distance
dmat = dist(x, method="euclidean", diag=FALSE, upper=TRUE)
mat = as.matrix(dmat)
diag(mat) <- NA
mat[upper.tri(mat)] <- NA
sd(as.vector(mat), na.rm=TRUE)

# Variance of distances as a function of dimension ==============
# That is, the concentration of points on the surface of a sphere
# as implied by theory.
pts.sd = 0.1
dim.v = 1000:2000
#spread = double(len(dim.v))
#mean.dist = double(len(dim.v))

p = 2
# NOTE : takes > one hr to finish !!
res <- foreach ( i=dim.v[1]:dim.v[len(dim.v)], .combine=rbind ) %dopar%
{
  x = matrix(rnorm(nr*i, mean=0, sd=pts.sd), nrow=nr, ncol=i)
  mat = as.matrix(dist(x, method="minkowski", p=p))
  diag(mat) <- NA
  mat[upper.tri(mat)] <- NA
  mean.dist = mean(as.vector(mat), na.rm=TRUE)
  spread = sd(as.vector(mat), na.rm=TRUE)
  c(mean.dist, spread)
}
colnames(res) = c("mean.dist", "spread")
dim(res)

#X11();
pdf(file=paste("MinkowskiDist_in_high_dim_p=", round(p,1), ".pdf", sep=""), width=18, height=11)
par(mfrow=c(1,2))
plot(dim.v, res[,"spread"], type="l", xlab="dim", ylab="stdev",  col="red", main=paste("Variability of p=", round(p,1), "-norm Minkowski Distances vs dim", sep=""))
plot(dim.v, res[,"mean.dist"], type="l", col="red", xlab="dim", ylab="ave dist", main=paste("Ave p=", round(p,1), "-norm Minkowski Pairwise Distance vs dim", sep=""))
dev.off()

# Observation : 1) variability of distances betwn points is very tight, and quite stable, 
#               not changing much as dimensionality increases. Is that expected ?
#               2) average distance betwn points is an (monotonic?) incresing function of dimension.
#               




