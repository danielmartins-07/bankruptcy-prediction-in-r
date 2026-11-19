required_packages <- c("farff", "caret", "mice", "smotefamily", "randomForest", "pROC", "PRROC")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  command <- paste(shQuote(missing_packages), collapse = ", ")
  stop("Install the missing packages with install.packages(c(", command, "))", call. = FALSE)
}

scripts <- sort(list.files("scripts", pattern = "^[0-9].*\\.R$", full.names = TRUE))
if (!length(scripts)) {
  stop("Run this command from the project root.", call. = FALSE)
}
for (script in scripts) {
  message("\n", basename(script))
  source(script, encoding = "UTF-8")
}
