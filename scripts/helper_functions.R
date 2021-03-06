LoadPackages <- function(packages) {
  # Load or install packages if they aren't already loaded.
  #
  # Args:
  #   packages: a vector of package names
  #
  for (package in packages) {
    if (!require(package, character.only=T, quietly=T)) {
      if (!package %in% installed.packages()) install.packages(package)
      library(package, character.only=T)
    }
  }
}

summarize_df <- function(df, r = 4) {
  # creates a dataframe that summarizes the input data
  require(dplyr)
  summary_names <- c("col_name", 
                     "type", 
                     "num_unq",
                     "mode", 
                     "mode_ratio",                     
                     "num_missing", 
                     "num_na", 
                     "num_inf", 
                     "num_nan",
                     "min", 
                     "q1", 
                     "median", 
                     "mean", 
                     "q3", 
                     "max", 
                     "std_dev")
  col_summary <- data.frame(matrix(ncol = length(summary_names), nrow = ncol(df)))
  names(col_summary) <- summary_names
  
  for(i in 1:ncol(df)) {
    col <- df[,i]
    not_inf <- df[!is.infinite(col), i]
    freq_table <- sort(table(not_inf), decreasing=TRUE)[1]
    
    col_name <- names(df)[i]
    type <- class(col)
    num_inf <- length(df[is.infinite(col),i])
    num_nan <- length(df[is.nan(col),i])
    num_na <- length(df[is.na(col),i]) - num_nan
    num_missing <- num_na + num_inf + num_nan
    num_unq <- length(unique(not_inf[!is.na(not_inf)])) # NAs and INF values not included
    mode <- names(freq_table) # NAs and INF values not included
    mode_ratio <- unname(freq_table) / (length(col) - num_missing)
    
    
    if(is.numeric(col) == TRUE) {
      min <- min(col, na.rm = TRUE)
      q1 <- quantile(not_inf, .25, na.rm = TRUE) %>% unname()
      median <- median(not_inf, na.rm = TRUE)
      mean <- mean(not_inf, na.rm = TRUE)
      q3 <- quantile(not_inf, .75, na.rm = TRUE) %>% unname()
      max <- max(not_inf, na.rm = TRUE)
      std_dev <- sd(not_inf, na.rm = TRUE)
    } else {
      min <- NA
      q1 <- NA
      median <- NA
      mean <- NA
      q3 <- NA
      max <- NA
      std_dev <- NA
    }
    
    col_summary[i,] <- c(col_name, 
                         type, 
                         num_unq,
                         mode,
                         mode_ratio,
                         num_missing, 
                         num_na,
                         num_inf, 
                         num_nan,
                         min, 
                         q1, 
                         median, 
                         mean, 
                         q3, 
                         max, 
                         std_dev)
  }
  numerics <- dplyr::setdiff(names(col_summary),c("col_name", "type", "mode")) 
  col_summary <- dplyr::mutate_at(col_summary, numerics, funs(as.double(.)))
  col_summary <- dplyr::mutate_at(col_summary, numerics, funs(round(., r)))
  
  return(col_summary)
}
