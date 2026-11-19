test_predictors <- statements[partition == "test", predictor_names]
test_inputs <- rbind(train_predictors, test_predictors)
test_imputed <- impute_predictors(test_inputs, seq_len(nrow(test_inputs)) <= nrow(train_predictors))
test <- tail(test_imputed, nrow(test_predictors))
test_actual <- statements$class[partition == "test"]

# Reuse the frozen forests and one set of scores per model for the tables and plots.
test_scores <- lapply(models, score_bankruptcy, rows = test)
roc_curves <- lapply(test_scores, function(scores) {
  pROC::roc(test_actual, scores, levels = c("0", "1"), direction = "<", quiet = TRUE)
})
pr_curves <- lapply(test_scores, function(scores) {
  PRROC::pr.curve(scores[test_actual == "1"], scores[test_actual == "0"], curve = TRUE)
})

report <- do.call(rbind, lapply(names(models), function(model_name) {
  thresholds <- c(default = 0.5, tuned_on_validation = selected_thresholds[[model_name]])
  do.call(rbind, lapply(names(thresholds), function(rule) {
    data.frame(
      model = model_name, rule = rule,
      threshold_metrics(test_scores[[model_name]], test_actual, thresholds[[rule]]),
      roc_auc = as.numeric(pROC::auc(roc_curves[[model_name]])),
      pr_auc = pr_curves[[model_name]]$auc.integral
    )
  }))
}))
print(report, row.names = FALSE)

write.csv(report, "outputs/test_metrics.csv", row.names = FALSE)
write.csv(as.data.frame.matrix(table(partition, statements$class)),
          "outputs/partition_counts.csv")
for (model_name in names(models)) {
  write.csv(validation_results[[model_name]],
            file.path("outputs", paste0("validation_thresholds_", model_name, ".csv")),
            row.names = FALSE)
}
writeLines(capture.output(sessionInfo()), "outputs/session_info.txt")

palette <- c(baseline = "#21668A", smote = "#AC371B")
class_counts <- table(statements$class)
png("outputs/class_distribution.png", width = 900, height = 630)
bar_centers <- barplot(class_counts, names.arg = c("Non-bankrupt", "Bankrupt"),
                      col = palette, ylim = c(0, max(class_counts) * 1.12),
                      ylab = "Financial statements", main = "Bankruptcy labels in the UCI dataset")
text(bar_centers, class_counts, labels = formatC(class_counts, format = "d", big.mark = ","),
     pos = 3)
invisible(dev.off())

png("outputs/roc_curve.png", width = 900, height = 630)
plot(roc_curves$baseline, legacy.axes = TRUE, col = palette[["baseline"]], lwd = 2,
     main = "ROC on the held-out test set")
pROC::lines.roc(roc_curves$smote, col = palette[["smote"]], lwd = 2)
legend("bottomright", legend = c("Random Forest", "Random Forest + SMOTE"),
       col = palette, lwd = 2, bty = "n")
invisible(dev.off())

png("outputs/precision_recall_curve.png", width = 900, height = 630)
plot(pr_curves$baseline$curve[, 1:2], type = "l", xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Recall", ylab = "Precision", col = palette[["baseline"]], lwd = 2,
     main = "Precision and recall on the held-out test set")
lines(pr_curves$smote$curve[, 1:2], col = palette[["smote"]], lwd = 2)
abline(h = mean(test_actual == "1"), lty = 2, col = "gray50")
legend("topright", legend = c("Random Forest", "Random Forest + SMOTE", "Bankruptcy prevalence"),
       col = c(palette, "gray50"), lwd = c(2, 2, 1), lty = c(1, 1, 2), bty = "n")
invisible(dev.off())
