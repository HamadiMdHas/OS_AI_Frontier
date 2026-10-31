INPUT_FILE   <- file.path("data", "dataor.xlsx")
SHEET_NAME   <- "Sheet1"
OUTPUT_FILE  <- INPUT_FILE
OUTPUT_DIR   <- "outputs"
TXT_FILE     <- file.path(OUTPUT_DIR, "input.txt")
XLSX_FILE    <- file.path(OUTPUT_DIR, "input.xlsx")
K            <- 40
MIN_DOC_FREQ <- 5
STEM         <- TRUE
CUSTOM_STOPWORDS <- c()
N_WORDS      <- 15
SEED         <- 12345

COL_CLEAN <- "content_cleaned"
COL_STOP  <- "content_no_stopwords"
COL_FINAL <- "content_preprocessed"

pkgs <- c("stm", "readxl", "openxlsx", "stringi", "tm", "SnowballC")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

if (!file.exists(INPUT_FILE)) stop("Could not find ", INPUT_FILE, ". Run this script from the project root folder.")
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)

raw <- as.data.frame(read_excel(INPUT_FILE, sheet = SHEET_NAME))
orig_names <- names(raw)
raw$.row_id <- seq_len(nrow(raw))
cat("Rows read:", nrow(raw), "\n")

req_cols <- c("content", "score", "period")
if (!all(req_cols %in% names(raw)))
  stop("Missing column(s): ", paste(setdiff(req_cols, names(raw)), collapse = ", "))

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- iconv(x, from = "UTF-8", to = "UTF-8", sub = " ")
  x <- gsub("[\u2018\u2019\u201B`\u00B4]", "'", x)
  x <- tolower(x)
  x <- gsub("<[^>]+>", " ", x)
  x <- gsub("(https?://|www\\.)\\S+", " ", x)
  x <- gsub("\\S+@\\S+", " ", x)
  x <- stri_replace_all_regex(x, "[^\\p{ASCII}]", " ")
  x <- gsub("\\b\\d+\\b", " ", x)
  x <- gsub("[^a-z' ]", " ", x)
  x <- gsub("(^|\\s)'+|'+(\\s|$)", " ", x)
  x <- gsub("(.)\\1{2,}", "\\1\\1", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

extra_stop <- c("dont", "doesnt", "didnt", "cant", "couldnt", "wouldnt", "shouldnt",
                "wont", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt",
                "im", "ive", "youre", "thats", "theres", "lets")
all_stop <- unique(c(stopwords("en"), extra_stop, tolower(CUSTOM_STOPWORDS)))

remove_stop <- function(x) {
  vapply(strsplit(x, " +"), function(w) {
    w <- w[w != "" & !(w %in% all_stop)]
    paste(w, collapse = " ")
  }, character(1))
}

raw[[COL_CLEAN]] <- clean_text(raw$content)
raw[[COL_STOP]]  <- remove_stop(raw[[COL_CLEAN]])

raw$score  <- suppressWarnings(as.numeric(raw$score))
raw$period <- suppressWarnings(as.numeric(as.character(raw$period)))

keep <- nchar(raw[[COL_STOP]]) > 0 & complete.cases(raw[, c("score", "period")])
cat("Rows dropped before modeling (empty text or missing covariates):", sum(!keep), "\n")
dat <- raw[keep, ]

processed <- textProcessor(
  documents         = dat[[COL_STOP]],
  metadata          = dat[, c(".row_id", "score", "period")],
  lowercase         = TRUE,
  removestopwords   = FALSE,
  removenumbers     = TRUE,
  removepunctuation = TRUE,
  onlycharacter     = TRUE,
  stem              = STEM,
  wordLengths       = c(3, Inf),
  language          = "en"
)

out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
                     lower.thresh = MIN_DOC_FREQ)
cat("Documents retained:", length(out$documents),
    "| Vocabulary size:", length(out$vocab), "\n")

final_text <- function(txt) {
  w <- strsplit(txt, " +")[[1]]
  w <- gsub("[^a-z]", "", w)
  w <- w[w != ""]
  if (STEM) w <- SnowballC::wordStem(w, language = "english")
  paste(w[w %in% out$vocab], collapse = " ")
}
raw[[COL_FINAL]] <- NA_character_
raw[[COL_FINAL]][out$meta$.row_id] <- vapply(raw[[COL_STOP]][out$meta$.row_id],
                                             final_text, character(1), USE.NAMES = FALSE)

prev_formula <- ~ score + period + score:period

cat("\nFitting STM with K =", K, "...\n")
set.seed(SEED)
model <- stm(out$documents, out$vocab, K = K,
             prevalence = prev_formula, data = out$meta,
             init.type = "Spectral", max.em.its = 500, seed = SEED, verbose = TRUE)

theta <- model$theta
colnames(theta) <- paste0("Topic_", seq_len(K))
theta_full <- matrix(NA_real_, nrow = nrow(raw), ncol = K, dimnames = list(NULL, colnames(theta)))
theta_full[out$meta$.row_id, ] <- theta
theta_full <- as.data.frame(theta_full)

lab <- labelTopics(model, n = N_WORDS)
make_rows <- function(mat, type) {
  df <- as.data.frame(mat, stringsAsFactors = FALSE)
  names(df) <- paste0("Word_", seq_len(ncol(df)))
  cbind(Topic = seq_len(nrow(df)), Word_Type = type, df, stringsAsFactors = FALSE)
}
words_df <- rbind(make_rows(lab$prob,  "prob"),
                  make_rows(lab$frex,  "frex"),
                  make_rows(lab$lift,  "lift"),
                  make_rows(lab$score, "score"))
words_df <- words_df[order(words_df$Topic,
                           match(words_df$Word_Type, c("prob", "frex", "lift", "score"))), ]

backup <- sub("\\.xlsx$", "_backup.xlsx", INPUT_FILE)
if (!file.exists(backup)) file.copy(INPUT_FILE, backup)

new_cols <- cbind(raw[, c(COL_CLEAN, COL_STOP, COL_FINAL)], theta_full)

wb <- loadWorkbook(INPUT_FILE)
sheet_names <- names(read_excel(INPUT_FILE, sheet = SHEET_NAME, n_max = 1))
stale <- which(sheet_names %in% c(COL_CLEAN, COL_STOP, COL_FINAL) |
                 grepl("^Topic_\\d+$", sheet_names))
if (length(stale) > 0) {
  deleteData(wb, SHEET_NAME, cols = stale, rows = 1:(nrow(raw) + 1), gridExpand = TRUE)
  start_col <- min(stale)
} else {
  start_col <- length(sheet_names) + 1
}
writeData(wb, SHEET_NAME, new_cols, startCol = start_col, startRow = 1)

if ("words" %in% names(wb)) removeWorksheet(wb, "words")
addWorksheet(wb, "words")
writeData(wb, "words", words_df)
setColWidths(wb, "words", cols = 1:ncol(words_df), widths = "auto")

saveWorkbook(wb, OUTPUT_FILE, overwrite = TRUE)
cat("\nSaved to ", normalizePath(OUTPUT_FILE), "\n",
    "  '", SHEET_NAME, "': 3 text columns + ", K, " topic probability columns\n",
    "  'words': topic word lists\n", sep = "")

set.seed(SEED)
eff <- estimateEffect(1:K ~ score + period + score:period, stmobj = model,
                      metadata = out$meta, uncertainty = "Global")
sm  <- summary(eff, topics = 1:K)

sig_symbol <- function(p) {
  ifelse(is.na(p), "",
         ifelse(p < 0.001, "***",
                ifelse(p < 0.01,  "**",
                       ifelse(p < 0.05,  "*",
                              ifelse(p < 0.1,   "\u2020", "")))))
}

res <- do.call(rbind, lapply(seq_len(K), function(k) {
  tab <- sm$tables[[k]]
  data.frame(Topic = k, Term = rownames(tab),
             Estimate = tab[, 1], Std.Error = tab[, 2],
             t.value = tab[, 3], p.value = tab[, 4],
             Sig = sig_symbol(tab[, 4]), row.names = NULL, stringsAsFactors = FALSE)
}))

stats_lines <- capture.output({
  cat("=================== Effects on topic prevalence ===================\n")
  cat("Prevalence ~ score + period + score:period\n")
  cat("Signif.: *** p<0.001  ** p<0.01  * p<0.05  \u2020 p<0.1\n\n")
  for (k in seq_len(K)) {
    r <- res[res$Topic == k, ]
    top <- paste(lab$frex[k, 1:5], collapse = ", ")
    cat("Topic ", k, "  (top FREX: ", top, ")\n", sep = "")
    show <- data.frame(Term = r$Term,
                       Estimate  = sprintf("%.5f", r$Estimate),
                       Std.Error = sprintf("%.5f", r$Std.Error),
                       t.value   = sprintf("%.3f", r$t.value),
                       p.value   = ifelse(r$p.value < 0.0001, "<0.0001", sprintf("%.4f", r$p.value)),
                       Sig       = r$Sig)
    print(show, row.names = FALSE, right = FALSE)
    cat("\n")
  }
})

cat("\n")
cat(stats_lines, sep = "\n")

frex_lines <- c(
  "=================== Topics and FREX words ===================",
  sprintf("(K = %d, %d FREX words per topic)", K, N_WORDS),
  "",
  vapply(seq_len(K), function(k)
    sprintf("Topic %d: %s", k, paste(lab$frex[k, ], collapse = ", ")),
    character(1)),
  "",
  ""
)

con <- file(TXT_FILE, open = "w", encoding = "UTF-8")
writeLines(c(frex_lines, stats_lines), con)
close(con)

frex_df <- data.frame(Topic = seq_len(K),
                      FREX_words = vapply(seq_len(K), function(k)
                        paste(lab$frex[k, ], collapse = ", "), character(1)),
                      stringsAsFactors = FALSE)
effects_df <- data.frame(res[, c("Topic", "Term")],
                         Estimate  = res$Estimate,
                         Std.Error = res$Std.Error,
                         t.value   = res$t.value,
                         p.value   = res$p.value,
                         Sig       = res$Sig,
                         stringsAsFactors = FALSE)
wb2 <- createWorkbook()
addWorksheet(wb2, "FREX");    writeData(wb2, "FREX", frex_df)
addWorksheet(wb2, "Effects"); writeData(wb2, "Effects", effects_df)
setColWidths(wb2, "FREX",    cols = 1:2, widths = c(8, 100))
setColWidths(wb2, "Effects", cols = 1:ncol(effects_df), widths = "auto")
saveWorkbook(wb2, XLSX_FILE, overwrite = TRUE)

cat("Saved text report to  ", normalizePath(TXT_FILE),  "\n", sep = "")
cat("Saved Excel report to ", normalizePath(XLSX_FILE), "\n", sep = "")

cat("Done.\n")

N_DOCS <- 4

kept_rows <- out$meta$.row_id
orig_text <- as.character(raw$content[kept_rows])

thoughts <- findThoughts(model, texts = orig_text, n = N_DOCS, topics = 1:K)

cat("\n=================== Representative documents ===================\n")
cat(sprintf("(%d highest-probability reviews per topic)\n\n", N_DOCS))

for (k in seq_len(K)) {
  idx <- thoughts$index[[k]]
  cat("Topic ", k, "  (top FREX: ", paste(lab$frex[k, 1:5], collapse = ", "), ")\n", sep = "")
  for (i in seq_along(idx)) {
    cat(sprintf("  [%d] (theta = %.3f | score = %s | period = %s | sheet row %d)\n      %s\n",
                i, theta[idx[i], k], out$meta$score[idx[i]], out$meta$period[idx[i]],
                kept_rows[idx[i]] + 1, gsub("\\s+", " ", orig_text[idx[i]])))
  }
  cat("\n")
}