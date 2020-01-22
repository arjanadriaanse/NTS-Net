library(ggplot2)
library(tikzDevice)

data <- read.csv("log.csv")
colnames(data) <- c("epoch", "trainloss", "trainacc", "testloss", "testacc")

tikz("epochsacc.tex",width=3,height=3)
acc.plot <- ggplot(data, aes(epoch)) +
    geom_line(aes(y = testacc, colour="testing")) +
    geom_line(aes(y = trainacc, colour="training")) +
    coord_cartesian(ylim=c(0,1), xlim=c(1,50)) +
    ylab("Accuracy") + xlab("Epochs") +
    theme_bw() + theme(legend.position="bottom") + scale_color_discrete(name = "Set")
acc.plot
dev.off()

tikz("epochsloss.tex",width=3,height=3)
loss.plot <- ggplot(data, aes(epoch)) +
    geom_line(aes(y = testloss, colour="testing")) +
    geom_line(aes(y = trainloss, colour="training")) +
    coord_cartesian(xlim=c(1,50)) +
    ylab("Loss") + xlab("Epochs") +
    theme_bw() + theme(legend.position="bottom") + scale_color_discrete(name = "Set")
loss.plot
dev.off()
