# 🚀 Data Pipeline Capstone Project

A comprehensive, interactive web-based data pipeline demonstrating the complete data science workflow: **Ingest → Transform → Analyze → Model**

## 🏀 Basketball Stats Data Pipeline
### 🎯 Overview
This capstone project demonstrates a complete data pipeline using web scraping, data wrangling, and exploratory analysis techniques. It pulls live NBA player statistics from Basketball Reference, processes the data into a structured format, and sets the foundation for deeper statistical insights or modeling.

### 🎓 Educational Goals
- Practice end-to-end data ingestion from live web sources
- Understand HTML parsing and table extraction using BeautifulSoup
- Clean and validate real-world sports data
- Prepare data for visualization or modeling tasks
- Develop foundational skills in working with tabular datasets

### 🔥 Core Functionality
- **🌐 Live Web Scraping**: Extracts per-game NBA player stats from Basketball Reference
- **🧹 Clean Data Pipeline**: Removes missing rows, renames columns, and converts to DataFrame
- **📊 Preview Output**: Displays top rows of cleaned data
- **📁 CSV Export Ready**: Easy extension to save output for later use
- **📉 Analysis-Ready Format**: Compatible with pandas, NumPy, and visualization libraries

### 🔄 Pipeline Stages
#### 1️⃣ Data Ingest
- **Target URL**: Scrapes data from the 2023–24 NBA per-game stats page
- **HTML Request**: Uses requests and BeautifulSoup to pull and parse the HTML
- **Table Detection**: Locates the per_game_stats table by ID
- **Header + Row Parsing**: Extracts header columns and player rows

#### 2️⃣ Data Transform
- **Row Filtering**: Removes duplicate header rows and missing players
- **DataFrame Conversion**: Converts structured data to a pandas DataFrame
- **Column Cleanup**: Strips whitespace and cleans column names
- **Data Types**: Converts numeric fields for analysis

#### 3️⃣ Data Analysis
- **Preview**: Prints shape and first 5 rows of the dataset
- **Ready for Modeling**: Compatible with scikit-learn or other ML libraries

### 🚀 Getting Started
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local development server (recommended)
- Text editor or IDE (VS Code recommended)

#### Python HTTP Server
```bash
# Navigate to project directory
cd data-pipeline

# Start server (Python 3)
python -m http.server 8000

# Open browser to http://localhost:8000
```

## E-Commerce Data

### 🎯 Overview

This capstone project demonstrates a complete data science pipeline using modern web technologies. Built with vanilla JavaScript and interactive visualizations, it processes e-commerce data through four distinct stages, providing hands-on experience with real-world data science workflows.

### 🎓 Educational Goals
- Understand the end-to-end data science process
- Learn data cleaning and transformation techniques
- Practice exploratory data analysis (EDA)
- Implement and evaluate machine learning models
- Build interactive data visualizations

### 🔥 Core Functionality
- **📊 Interactive Dashboard**: Real-time progress tracking and visual feedback
- **📁 Flexible Data Input**: Upload CSV files or generate sample datasets
- **🔄 Data Transformation**: Automated cleaning, normalization, and feature engineering
- **📈 Rich Visualizations**: Dynamic charts using Chart.js library
- **🤖 Machine Learning**: Linear regression model with performance metrics
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

### 🔄 Pipeline Stages

#### 1️⃣ Data Ingest
- **File Upload**: Support for CSV file uploads with validation
- **Sample Data**: Generate synthetic e-commerce dataset (1000+ records)
- **Data Validation**: Integrity checks and format verification
- **Preview**: Display data structure and column information

#### 2️⃣ Data Transform
- **Data Cleaning**: Remove null values, duplicates, and invalid entries
- **Normalization**: Scale numerical features to [0,1] range
- **Feature Engineering**: Create derived features like price categories and seasonal indicators
- **Data Quality**: Comprehensive data quality assessment

#### 3️⃣ Data Analysis
- **Exploratory Analysis**: Calculate key business metrics and statistics
- **Visualizations**: Interactive charts showing revenue distribution by category
- **Statistical Analysis**: Correlation analysis between variables
- **Insights Generation**: Automated insights and pattern detection

#### 4️⃣ ML Modeling
- **Model Preparation**: Feature selection and train/test split (80/20)
- **Training**: Linear regression model for price prediction
- **Evaluation**: RMSE, MAE, and prediction accuracy metrics
- **Visualization**: Scatter plot of predictions vs. actual values

### 🚀 Getting Started
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local development server (recommended)
- Text editor or IDE (VS Code recommended)

#### Python HTTP Server
```bash
# Navigate to project directory
cd data-pipeline

# Start server (Python 3)
python -m http.server 8000

# Open browser to http://localhost:8000
```
