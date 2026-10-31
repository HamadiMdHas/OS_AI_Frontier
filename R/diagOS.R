top_n    <- 2
n_neg    <- 1
alpha    <- 0.05

DATA_FILE  <- file.path("data", "dataor.xlsx")
OUTPUT_DIR <- "outputs"

if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl")
library(readxl)

if (!file.exists(DATA_FILE)) stop("Could not find ", DATA_FILE, ". Run this script from the project root folder.")
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)

dat <- as.data.frame(read_excel(DATA_FILE, sheet = 1))

c_vars     <- paste0("c", 1:6)
topic_vars <- paste0("Topic_", 1:40)

missing_vars <- setdiff(c(c_vars, topic_vars), names(dat))
if (length(missing_vars) > 0)
  stop("Missing columns in sheet 1: ", paste(missing_vars, collapse = ", "))

dat[c(c_vars, topic_vars)] <- lapply(dat[c(c_vars, topic_vars)], as.numeric)

sig_symbol <- function(p) {
  ifelse(is.na(p), "",
         ifelse(p < 0.001, "***",
                ifelse(p < 0.01,  "**",
                       ifelse(p < 0.05,  "*",
                              ifelse(p < 0.10,  "\u2020", "")))))
}

cor_row <- function(x, y, xname, yname) {
  ok <- complete.cases(x, y)
  n  <- sum(ok)
  out <- data.frame(Var1 = xname, Var2 = yname, N = n,
                    Estimate = NA_real_, SE = NA_real_, t = NA_real_,
                    p = NA_real_, stringsAsFactors = FALSE)
  if (n < 3 || sd(x[ok]) == 0 || sd(y[ok]) == 0) return(out)
  ct <- cor.test(x[ok], y[ok], method = "pearson")
  r  <- unname(ct$estimate)
  out$Estimate <- r
  out$SE       <- sqrt((1 - r^2) / (n - 2))
  out$t        <- unname(ct$statistic)
  out$p        <- ct$p.value
  out
}

format_p <- function(p) ifelse(is.na(p), "NA",
                               ifelse(p < 0.001, "<0.001", sprintf("%.4f", p)))

fmt_table <- function(df) {
  out <- data.frame(Var1 = df$Var1, Var2 = df$Var2, N = df$N,
                    Estimate = sprintf("%.4f", df$Estimate),
                    SE       = sprintf("%.4f", df$SE),
                    t        = sprintf("%.3f", df$t),
                    p        = format_p(df$p),
                    Sig      = sig_symbol(df$p),
                    stringsAsFactors = FALSE)
  if ("Pick" %in% names(df)) out$Pick <- df$Pick
  out
}

table_lines <- function(df) {
  if (nrow(df) == 0) return("  (none)")
  capture.output(print(fmt_table(df), row.names = FALSE, right = FALSE))
}

legend <- "Significance: *** p<0.001 | ** p<0.01 | * p<0.05 | \u2020 p<0.10"

cc_pairs <- combn(c_vars, 2, simplify = FALSE)
cc_res <- do.call(rbind, lapply(cc_pairs, function(pr)
  cor_row(dat[[pr[1]]], dat[[pr[2]]], pr[1], pr[2])))

ct_res <- do.call(rbind, lapply(c_vars, function(cv)
  do.call(rbind, lapply(topic_vars, function(tv)
    cor_row(dat[[cv]], dat[[tv]], cv, tv)))))

cat("\n==== Correlations among c1-c6 (all pairs) ====\n")
print(fmt_table(cc_res), row.names = FALSE, right = FALSE)

cat("\n==== Correlations of each c with Topic_1 ... Topic_40 (all tests) ====\n")
for (cv in c_vars) {
  cat("\n---", cv, "vs topics ---\n")
  print(fmt_table(ct_res[ct_res$Var1 == cv, ]), row.names = FALSE, right = FALSE)
}
cat("\n", legend, "\n", sep = "")

top_list <- lapply(c_vars, function(cv) {
  sub <- ct_res[ct_res$Var1 == cv & !is.na(ct_res$p) & ct_res$p < alpha, ]
  sub$Pick <- character(nrow(sub))
  if (nrow(sub) == 0) return(sub)
  
  sub <- sub[order(-sub$Estimate), ]
  top_pos <- head(sub, top_n)
  top_pos$Pick <- rep("highest estimate", nrow(top_pos))
  
  neg <- sub[sub$Estimate < 0 & !(sub$Var2 %in% top_pos$Var2), ]
  neg <- neg[order(neg$Estimate), ]
  neg <- head(neg, n_neg)
  if (nrow(neg) > 0) neg$Pick <- rep("strongest negative", nrow(neg))
  
  rbind(top_pos, neg)
})
names(top_list) <- c_vars

out_file <- file.path(OUTPUT_DIR, "summary_correlations.txt")

lines <- c(
  "SUMMARY OF CORRELATIONS (Pearson; phi coefficient for 0/1 pairs)",
  paste("Data file: dataor.xlsx (sheet 1) | Run:", format(Sys.time(), "%Y-%m-%d %H:%M")),
  paste("Topic selection: p <", alpha, "| per c variable:", top_n,
        "highest-estimate topics + ", n_neg,
        "most negative topic (largest magnitude)"),
  legend, "",
  "=== A) c1-c6 pool: correlations between all c_i and c_j pairs ===",
  table_lines(cc_res), "",
  "=== B) Top topics per c variable (highest estimate, significant) ==="
)
for (cv in c_vars) {
  lines <- c(lines, "", paste0("--- ", cv, " ---"), table_lines(top_list[[cv]]))
}
writeLines(lines, out_file, useBytes = FALSE)

cat("\n==== Top topics per c (p <", alpha, ", highest estimate) ====\n")
for (cv in c_vars) {
  cat("\n---", cv, "---\n")
  if (nrow(top_list[[cv]]) == 0) cat("  (none)\n") else
    print(fmt_table(top_list[[cv]]), row.names = FALSE, right = FALSE)
}
cat("\nSummary saved to:", normalizePath(out_file), "\n")