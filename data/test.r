setwd("E:/p2020/分子指纹自编码器")
library(tidyverse)
test.file <- readLines("test.csv")

feature.list <- list()
for (Line in test.file) {
    Line <- str_split(Line, ",")
    names(Line) <- Line[[1]][1]
    Line[[1]] <- Line[[1]][-1]
    Line[[1]] <- Line[[1]][Line[[1]] != ""]
    feature.list <- append(feature.list, Line)
}


feature.binary <- as.data.frame.matrix(t(table(stack(feature.list))))

write.csv(feature.binary, "feature.binary.csv")

