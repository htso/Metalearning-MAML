# fun_generator.R : Generate, plot, and save harmonic functional shapes
# This is part of the project to examine the meta learning capability of maml (Finn et al 2017)

library(goodies)

path = "/mnt/WanChai/Dropbox/Tensorflow-Mostly/MetaLearning/maml_ht"
setwd(path)

# special case when W1 = W2
x = seq(from=0, to=6*pi, by=0.1)
W1 = 0.2
W2 = 0.1
Ph1 = pi
Ph2 = pi
y = sin(2*pi*W1*x + Ph1) * cos(2*pi*W2*x + Ph2) # product of sin, cos
y = sin(2*pi*W1*x + Ph1) * sin(2*pi*W2*x + Ph2) # product of sin
y = sin(2*pi*W1*x + Ph1) + cos(2*pi*W2*x + Ph2) # sum of sin, cos
y = sin(2*pi*W1*x + Ph1) + sin(2*pi*W2*x + Ph2) # sum of sin
X11();plot(x, y, type="l", col="red")



x = seq(from=-5, to=5.0, by=0.1)

w1 = seq(from=0.1, to=2.5, length.out=4)
w2 = seq(from=0.1, to=2.5, length.out=4)
ph1 = seq(from=0.1*pi, to=0.6*pi, length.out=4)
ph2 = seq(from=0.4*pi, to=0.9*pi, length.out=4)
Grid = expand.grid(w1=w1, w2=w2, ph1=ph1, ph2=ph2)
ix = which(Grid[,"w1"] == Grid[,"w2"])
Grid = Grid[-ix,]
dim(Grid)

pdf("fun-shapes.pdf", width=11, height=8.5)
par(mfrow=c(5,5), mar=c(2,2,2,2))
for ( i in 1:nrow(Grid) ) {
  W1 = Grid[i, "w1"]
  W2 = Grid[i, "w2"]
  Ph1 = Grid[i, "ph1"]
  Ph2 = Grid[i, "ph2"]
  y = sin(W1*x + Ph1) * cos(W2*x + Ph2)
  plot(x, y, type="l", col="red")
  mtext(paste("w:", W1, " ", W2, "ph:", round(Ph1,1), " ", round(Ph2,1)), 
            side=3, line=-2, cex=0.8)
}
dev.off()




