setwd("E:\\program\\software\\cygwin\\home\\YCKJ2939\\project\\Stacked-AutoEncoder-Model\\data")
library(tidyverse)
test.file <- readLines("jerry.csv")

feature.list <- list()
for (Line in test.file) {
    Line <- str_split(Line, ",")
    names(Line) <- Line[[1]][1]
    Line[[1]] <- Line[[1]][-1]
    Line[[1]] <- Line[[1]][Line[[1]] != ""]
    feature.list <- append(feature.list, Line)
}


feature.binary <- as.data.frame.matrix(t(table(stack(feature.list))))

write.csv(feature.binary, "train_base_jerry.binary.csv")

