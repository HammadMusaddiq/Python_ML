library(readxl)
library(imputeTS) ## install.packages("imputeTS")
library(ggplot2) ## install.packages("ggplot2")
library(corrplot) ## install.packages("corrplot")

# reading Excel file
data <- read_excel("C:\\Users\\Hammad\\Desktop\\Python New Code\\R Project\\3.-Investment-Data_258099571.xlsx")

# column data type
column_data_type <- class(data$DivYield)
print(paste("Data type of the column:", column_data_type))

# converting column value to numeric
data$DivYield <- as.numeric(data$DivYield)

# replacing missing values with mean
data_filled <- data
for (col in names(data_filled)) {
  if (any(is.na(data_filled[[col]]))) {
    data_filled[[col]][is.na(data_filled[[col]])] <- mean(data_filled[[col]], na.rm = TRUE)
  }
}

# checking if there are any missing values left
missing_values <- any(is.na(data_filled))
if (missing_values) {
  cat("There are still missing values in the data.")
} else {
  cat("All missing values have been filled.")
}


# Exploratory Data Analysis (EDA)
summary(data_filled)

par(mfrow=c(2, 2))
hist(data_filled$MktPrice, main="Market Price Histogram")

png("market_price_histogram.png", width = 7, height = 5, units = "in", res = 300)
hist(data_filled$MktPrice, main = "Market Price Histogram")
dev.off()


hist(data_filled$DivYield, main="Dividend Yield Histogram")
boxplot(data_filled$PERatio, main="PE Ratio Boxplot")


png("dividend_yield_histogram.png", width = 7, height = 5, units = "in", res = 300)
hist(data_filled$DivYield, main = "Dividend Yield Histogram")
dev.off()

# Save the PE Ratio boxplot using png()
png("pe_ratio_boxplot.png", width = 7, height = 5, units = "in", res = 300)
boxplot(data_filled$PERatio, main = "PE Ratio Boxplot")
dev.off()

scatterplot <- ggplot(data_filled, aes(x=TotalSales17, y=TotalSales18)) +
  geom_point() +
  labs(title="Scatter Plot of Total Sales (2017 vs 2018)")
print(scatterplot)


# # Task 1: Find which variables give the best measures of the investment potential
# Correlation Analysis
cor_matrix <- cor(data_filled[, c("MktPrice", "TotMktCap", "DivYield", "PERatio", "Beta", "TotalSales17", "TotalSales18", "CapEmp", "Dividend", "MktBook", "Ret17", "Ret18")])
#corrplot::corrplot(cor_matrix, method = "circle")
correlation_plot <- corrplot(cor_matrix, method = "circle")
# save plot
#ggsave("correlation_plot.png", plot = last_plot(), device = "png")
ggsave("correlation_plot1.png", plot = correlation_plot, device = "png")

png("correlation_plot1.png", width = 5.75, height = 3.48, units = "in", res = 300)
corrplot(cor_matrix, method = "circle")
dev.off()


# Task 2: Draw conclusions on which types of stock would provide the best investment
# Create a hypothetical grouping variable named 'Category'
data_filled$Category <- sample(c("Large", "Medium", "Small"), nrow(data_filled), replace = TRUE)

# Boxplot based on the hypothetical 'Category' variable
ggplot(data_filled, aes(x = factor(Category), y = MktPrice)) +
  geom_boxplot() +
  labs(title = "Boxplot of Market Price by Category")
# save plot
ggsave("boxplot.png", plot = last_plot(), device = "png")


# Task 3: Assess the validity of any predictions made
# Example: Scatter plot of Total Market Cap vs. Total Sales for validity assessment
ggplot(data_filled, aes(x = TotalSales17 + TotalSales18, y = TotMktCap)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Scatter Plot of Total Sales vs. Total Market Cap",
       subtitle = "Linear Regression Line for Validity Assessment")
# Save plot
ggsave("scatter_plot.png", plot = last_plot(), device = "png")
