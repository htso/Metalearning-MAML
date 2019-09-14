
ff = seq(from=1, to=5, by=0.5)
tt = seq(from=0.0, to=1, by=0.01)

X11()
par(mfrow=c(3,3), mar=c(1,1,1,1))
for ( i in 1:9) {
  plot(tt, sin(2*pi*tt*ff[i]), type="l")  
}

