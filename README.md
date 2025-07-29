# 🚀 Data Pipeline Capstone Project (Basketball Reference and E-Commerce)

A comprehensive, interactive web-based data pipeline demonstrating the complete data science workflow: **Ingest → Transform → Analyze → Model**

## E-Commerce Data

### 🎯 Overview

This capstone project demonstrates a complete data science pipeline using modern web technologies. Built with vanilla JavaScript and interactive visualizations, it processes e-commerce data through four distinct stages, providing hands-on experience with real-world data science workflows.

### 🎓 Educational Goals
- Understand the end-to-end data science process
- Learn data cleaning and transformation techniques
- Practice exploratory data analysis (EDA)
- Implement and evaluate machine learning models
- Build interactive data visualizations

### ✨ Features

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
