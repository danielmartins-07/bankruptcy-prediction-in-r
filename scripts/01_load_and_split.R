data_files <- file.path("data", paste0(1:5, "year.arff"))
if (!all(file.exists(data_files))) {
  stop("Download the five UCI ARFF files into data/. See README.md.", call. = FALSE)
}

# UCI numbers the files by observation year. The first file has a five-year forecast horizon.
statements <- do.call(rbind, lapply(seq_along(data_files), function(file_index) {
  rows <- farff::readARFF(data_files[file_index])
  rows$forecast_horizon <- 6L - file_index
  rows
}))
statements$class <- factor(statements$class, levels = c("0", "1"))
ratio_names <- paste0("Attr", 1:64)

# Group exact raw ratio vectors before dropping or imputing any features.
ratio_values <- lapply(statements[ratio_names], function(values) sprintf("%a", values + 0))
ratio_key <- do.call(paste, c(ratio_values, sep = "|"))
unique_keys <- unique(ratio_key)
group_id <- match(ratio_key, unique_keys)
groups <- seq_along(unique_keys)
positive_groups <- unique(group_id[statements$class == "1"])
group_label <- factor(ifelse(groups %in% positive_groups, "1", "0"))

set.seed(123)
train_groups <- groups[caret::createDataPartition(group_label, p = 0.6, list = FALSE)]
holdout_groups <- setdiff(groups, train_groups)
validation_groups <- holdout_groups[
  caret::createDataPartition(group_label[holdout_groups], p = 0.5, list = FALSE)
]
partition_by_group <- rep("test", length(groups))
partition_by_group[train_groups] <- "train"
partition_by_group[validation_groups] <- "validation"
partition <- factor(partition_by_group[group_id], levels = c("train", "validation", "test"))

stopifnot(length(unique(partition)) == 3L, all(table(partition, statements$class) > 0L))
print(table(partition, bankruptcy = statements$class))
cat("Statements:", nrow(statements), "| Unique raw ratio vectors:", length(groups), "\n")
