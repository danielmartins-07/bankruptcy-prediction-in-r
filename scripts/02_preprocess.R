training_rows <- partition == "train"
training_ratios <- statements[training_rows, ratio_names]
missing_rate <- colMeans(is.na(training_ratios))
missing_cutoff <- 0.4
top_missing <- head(sort(missing_rate, decreasing = TRUE), 10L) * 100
dir.create("outputs", showWarnings = FALSE)
write.csv(data.frame(ratio = names(top_missing), missing_percent = as.numeric(top_missing)),
          "outputs/training_missingness_top10.csv", row.names = FALSE)

png("outputs/missing_values_top10.png", width = 1000, height = 620, res = 120)
par(mar = c(4.4, 5, 4, 1))
bars <- rev(top_missing)
cutoff_percent <- missing_cutoff * 100
bar_centers <- barplot(
  bars, horiz = TRUE, las = 1, border = NA,
  col = ifelse(bars > cutoff_percent, "#AC371B", "#21668A"),
  xlim = c(0, max(c(bars, cutoff_percent)) + 8),
  main = "Financial ratios with the most missing data",
  xlab = "Missing values in training (%)"
)
abline(v = cutoff_percent, col = "gray40", lty = 2)
text(bars, bar_centers, labels = sprintf("%.1f%%", bars), pos = 4, cex = 0.85)
legend("bottomright", "40% removal cutoff", lty = 2, col = "gray40", bty = "n", cex = 0.85)
invisible(dev.off())

usable_ratios <- names(missing_rate[missing_rate <= missing_cutoff])
usable_ratios <- usable_ratios[vapply(training_ratios[usable_ratios], function(values) {
  length(unique(values[!is.na(values)])) > 1L
}, logical(1))]

correlations <- cor(training_ratios[usable_ratios], use = "pairwise.complete.obs")
redundant_ratios <- caret::findCorrelation(correlations, cutoff = 0.9, names = TRUE)
selected_ratios <- setdiff(usable_ratios, redundant_ratios)
predictor_names <- c(selected_ratios, "forecast_horizon")
train_predictors <- statements[training_rows, predictor_names]
training_medians <- vapply(train_predictors, median, numeric(1), na.rm = TRUE)

impute_predictors <- function(predictors, training) {
  # Start the imputation chain with training medians. PMM then draws real donor values.
  initialized <- predictors
  for (column in names(training_medians)) {
    initialized[[column]][is.na(initialized[[column]])] <- training_medians[[column]]
  }

  completed <- mice::mice(
    predictors, ignore = !training, data.init = initialized,
    method = "pmm", m = 1L, maxit = 5L, seed = 123L, printFlag = FALSE,
    remove.constant = FALSE, remove.collinear = FALSE
  )
  if (!is.null(completed$loggedEvents)) {
    print(completed$loggedEvents)
  }
  result <- mice::complete(completed)
  stopifnot(!anyNA(result))
  result
}

# Keep test rows out of this call. The outcome is absent from every imputation input.
development_rows <- partition != "test"
development_predictors <- impute_predictors(
  statements[development_rows, predictor_names], training_rows[development_rows]
)
development <- data.frame(development_predictors, class = statements$class[development_rows])
development_partition <- partition[development_rows]
train <- development[development_partition == "train", ]
validation <- development[development_partition == "validation", ]

cat("Removed for missingness or no variation:", paste(setdiff(ratio_names, usable_ratios), collapse = ", "), "\n")
cat("Removed for correlation:", length(redundant_ratios), "| Retained ratios:", length(selected_ratios), "\n")
