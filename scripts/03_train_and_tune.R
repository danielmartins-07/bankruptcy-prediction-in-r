# SMOTE uses nearest neighbours, so scale the ratios using training statistics.
# Generate synthetic rows within each horizon to preserve its integer meaning.
ratio_centers <- vapply(train[selected_ratios], mean, numeric(1))
ratio_scales <- vapply(train[selected_ratios], sd, numeric(1))
stopifnot(all(is.finite(ratio_scales)), all(ratio_scales > 0))
set.seed(123)
synthetic_rows <- do.call(rbind, lapply(split(train, train$forecast_horizon), function(rows) {
  scaled_ratios <- scale(rows[selected_ratios], center = ratio_centers, scale = ratio_scales)
  synthetic <- smotefamily::SMOTE(as.data.frame(scaled_ratios), rows$class, K = 5, dup_size = 3)$syn_data
  ratios <- sweep(as.matrix(synthetic[selected_ratios]), 2L, ratio_scales, "*")
  ratios <- sweep(ratios, 2L, ratio_centers, "+")
  data.frame(
    ratios, forecast_horizon = rows$forecast_horizon[1],
    class = factor(synthetic$class, levels = levels(rows$class))
  )
}))
train_smote <- rbind(train, synthetic_rows)
cat("Bankruptcy share in training:", mean(train$class == "1"),
    "| After SMOTE:", mean(train_smote$class == "1"), "\n")

training_sets <- list(baseline = train, smote = train_smote)
models <- lapply(training_sets, function(rows) {
  set.seed(123)
  randomForest::randomForest(class ~ ., data = rows, ntree = 100L)
})

score_bankruptcy <- function(model, rows) {
  predict(model, newdata = rows, type = "prob")[, "1"]
}

threshold_metrics <- function(scores, actual, threshold) {
  predicted <- scores >= threshold
  bankrupt <- actual == "1"
  true_positives <- sum(predicted & bankrupt)
  false_positives <- sum(predicted & !bankrupt)
  false_negatives <- sum(!predicted & bankrupt)
  data.frame(
    threshold = threshold,
    recall = true_positives / sum(bankrupt),
    precision = if (any(predicted)) true_positives / sum(predicted) else NA_real_,
    f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives),
    accuracy = mean(predicted == bankrupt),
    true_positives = true_positives,
    false_positives = false_positives,
    false_negatives = false_negatives,
    true_negatives = sum(!predicted & !bankrupt)
  )
}

threshold_grid <- seq(10, 90, by = 5) / 100
validation_results <- lapply(models, function(model) {
  scores <- score_bankruptcy(model, validation)
  do.call(rbind, lapply(threshold_grid, function(threshold) {
    threshold_metrics(scores, validation$class, threshold)
  }))
})
selected_thresholds <- vapply(validation_results, function(results) {
  results$threshold[which.max(results$f1)]
}, numeric(1))
print(selected_thresholds)
