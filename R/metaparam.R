packages <- c("readxl", "dplyr", "tidyr", "ggplot2",
              "viridis", "scales", "patchwork")

new_packages <- packages[
  !packages %in% installed.packages()[, "Package"]
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

invisible(lapply(packages, library, character.only = TRUE))


input_file <- file.path("data", "dataor.xlsx")

if (!file.exists(input_file)) {
  stop("Could not find ", input_file,
       ". Run this script from the project root folder.")
}

output_folder <- file.path("outputs", "Analysis_Results")

if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}


data <- read_excel(input_file, sheet = "Sheet1")

i_cols <- paste0("i", 1:6)
c_cols <- paste0("c", 1:6)
k_cols <- paste0("k", 1:6)

required_cols <- c(i_cols, c_cols, "cfit",
                   k_cols, "kfit")

missing_cols <- setdiff(required_cols, names(data))

if (length(missing_cols) > 0) {
  stop(paste("Missing columns:", 
             paste(missing_cols, collapse = ", ")))
}

all_cols <- c(i_cols, c_cols, "cfit", k_cols, "kfit")

data[all_cols] <- lapply(data[all_cols], function(x) {
  as.numeric(as.character(x))
})

data[all_cols] <- lapply(data[all_cols], function(x) {
  x[is.na(x)] <- 0
  x
})

if (any(as.matrix(data[i_cols]) < 0)) {
  stop("Topic proportions cannot be negative.")
}

if (any(as.matrix(data[c(c_cols, k_cols, "cfit", "kfit")]) 
        %in% c(-1, 2))) {
  stop("Categorization columns must contain binary 0/1 values.")
}

N <- nrow(data)

cat("Number of documents:", N, "\n")


IC_matrix <- matrix(
  NA, nrow = 6, ncol = 6,
  dimnames = list(i_cols, c_cols)
)

for (i in 1:6) {
  for (j in 1:6) {
    
    numerator <- sum(data[[i_cols[i]]] * data[[c_cols[j]]])
    
    denominator <- sum(data[[i_cols[i]]])
    
    IC_matrix[i, j] <- ifelse(
      denominator > 0,
      numerator / denominator,
      0
    )
  }
}

IC_df <- as.data.frame(as.table(IC_matrix))

names(IC_df) <- c("Topic", "Concept", "Association")

p1 <- ggplot(IC_df, aes(x = Concept, y = Topic,
                        fill = Association)) +
  
  geom_tile(color = "white", linewidth = 1) +
  
  geom_text(aes(label = sprintf("%.2f", Association)),
            size = 4) +
  
  scale_fill_viridis(
    option = "C",
    limits = c(0, 1),
    name = "Association"
  ) +
  
  labs(
    title = "Topic Modeling vs Categorization C",
    subtitle = "Weighted topic-concept association",
    x = "Concepts (C)",
    y = "Topics (I)"
  ) +
  
  theme_minimal(base_size = 13) +
  
  theme(
    plot.title = element_text(face = "bold"),
    axis.text = element_text(color = "black")
  )


IK_matrix <- matrix(
  NA, nrow = 6, ncol = 6,
  dimnames = list(i_cols, k_cols)
)

for (i in 1:6) {
  for (j in 1:6) {
    
    numerator <- sum(data[[i_cols[i]]] * data[[k_cols[j]]])
    
    denominator <- sum(data[[i_cols[i]]])
    
    IK_matrix[i, j] <- ifelse(
      denominator > 0,
      numerator / denominator,
      0
    )
  }
}

IK_df <- as.data.frame(as.table(IK_matrix))

names(IK_df) <- c("Topic", "Concept", "Association")

p2 <- ggplot(IK_df, aes(x = Concept, y = Topic,
                        fill = Association)) +
  
  geom_tile(color = "white", linewidth = 1) +
  
  geom_text(aes(label = sprintf("%.2f", Association)),
            size = 4) +
  
  scale_fill_viridis(
    option = "C",
    limits = c(0, 1),
    name = "Association"
  ) +
  
  labs(
    title = "Topic Modeling vs Categorization K",
    subtitle = "Weighted topic-concept association",
    x = "Concepts (K)",
    y = "Topics (I)"
  ) +
  
  theme_minimal(base_size = 13) +
  
  theme(
    plot.title = element_text(face = "bold"),
    axis.text = element_text(color = "black")
  )


CK_matrix <- matrix(
  NA, nrow = 6, ncol = 6,
  dimnames = list(c_cols, k_cols)
)

for (i in 1:6) {
  for (j in 1:6) {
    
    intersection <- sum(
      data[[c_cols[i]]] == 1 &
        data[[k_cols[j]]] == 1
    )
    
    union <- sum(
      data[[c_cols[i]]] == 1 |
        data[[k_cols[j]]] == 1
    )
    
    CK_matrix[i, j] <- ifelse(
      union > 0,
      intersection / union,
      0
    )
  }
}

CK_df <- as.data.frame(as.table(CK_matrix))

names(CK_df) <- c("Concept_C", "Concept_K", "Jaccard")

p3 <- ggplot(CK_df, aes(x = Concept_K, y = Concept_C,
                        fill = Jaccard)) +
  
  geom_tile(color = "white", linewidth = 1) +
  
  geom_text(aes(label = sprintf("%.2f", Jaccard)),
            size = 4) +
  
  scale_fill_viridis(
    option = "C",
    limits = c(0, 1),
    name = "Jaccard"
  ) +
  
  labs(
    title = "Categorization C vs Categorization K",
    subtitle = "Jaccard similarity between concepts",
    x = "Concepts (K)",
    y = "Concepts (C)"
  ) +
  
  theme_minimal(base_size = 13) +
  
  theme(
    plot.title = element_text(face = "bold"),
    axis.text = element_text(color = "black")
  )


coverage <- data.frame(
  Analysis = c("Categorization C", "Categorization K"),
  Coverage = c(
    mean(data$cfit > 0),
    mean(data$kfit > 0)
  )
)

p4 <- ggplot(coverage,
             aes(x = Analysis, y = Coverage,
                 fill = Analysis)) +
  
  geom_col(width = 0.6) +
  
  geom_text(
    aes(label = percent(Coverage, accuracy = 0.1)),
    vjust = -0.5,
    size = 5
  ) +
  
  scale_y_continuous(
    limits = c(0, 1.1),
    labels = percent
  ) +
  
  scale_fill_viridis_d(option = "C") +
  
  labs(
    title = "Classification Coverage",
    x = NULL,
    y = "Percentage of documents classified"
  ) +
  
  theme_minimal(base_size = 13) +
  
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold")
  )


coverage_status <- data.frame(
  Status = c(
    "Both C and K",
    "Only C",
    "Only K",
    "Neither C nor K"
  ),
  
  Documents = c(
    sum(data$cfit > 0 & data$kfit > 0),
    sum(data$cfit > 0 & data$kfit == 0),
    sum(data$cfit == 0 & data$kfit > 0),
    sum(data$cfit == 0 & data$kfit == 0)
  )
)

coverage_status$Percentage <-
  coverage_status$Documents / N

p5 <- ggplot(coverage_status,
             aes(x = Status, y = Percentage,
                 fill = Status)) +
  
  geom_col(width = 0.7) +
  
  geom_text(
    aes(label = paste0(
      Documents, " (",
      percent(Percentage, accuracy = 0.1), ")"
    )),
    vjust = -0.5,
    size = 3.5
  ) +
  
  scale_y_continuous(
    limits = c(0, 1.15),
    labels = percent
  ) +
  
  scale_fill_viridis_d(option = "C") +
  
  labs(
    title = "Overlap in Classification Coverage",
    x = NULL,
    y = "Percentage of documents"
  ) +
  
  theme_minimal(base_size = 12) +
  
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )


ggsave(
  file.path(output_folder, "01_Topic_vs_C.png"),
  p1, width = 9, height = 7, dpi = 300
)

ggsave(
  file.path(output_folder, "02_Topic_vs_K.png"),
  p2, width = 9, height = 7, dpi = 300
)

ggsave(
  file.path(output_folder, "03_C_vs_K.png"),
  p3, width = 9, height = 7, dpi = 300
)

ggsave(
  file.path(output_folder, "04_Coverage.png"),
  p4, width = 8, height = 6, dpi = 300
)

ggsave(
  file.path(output_folder, "05_Coverage_Overlap.png"),
  p5, width = 9, height = 6, dpi = 300
)


write.csv(
  IC_matrix,
  file.path(output_folder, "Topic_C_Association.csv")
)

write.csv(
  IK_matrix,
  file.path(output_folder, "Topic_K_Association.csv")
)

write.csv(
  CK_matrix,
  file.path(output_folder, "C_K_Jaccard.csv")
)

write.csv(
  coverage_status,
  file.path(output_folder, "Coverage_Status.csv"),
  row.names = FALSE
)


print(p1)
print(p2)
print(p3)
print(p4)
print(p5)

cat("\n====================================\n")
cat("ANALYSIS COMPLETED SUCCESSFULLY\n")
cat("Results saved in:", normalizePath(output_folder), "\n")
cat("====================================\n")