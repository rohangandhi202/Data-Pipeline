#!/usr/bin/env python3
"""
🏀 NBA Data Pipeline: Web Scraping to Machine Learning
==================================================
A complete pipeline that scrapes NBA player stats, cleans the data,
performs exploratory analysis, and builds predictive models.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class NBADataPipeline:
    def __init__(self):
        self.url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
        self.raw_data = None
        self.clean_data = None
        self.features = None
        self.target = None
        
    def ingest_data(self):
        """🔁 Step 1: Scrape NBA player statistics"""
        print("🔁 INGESTING DATA...")
        print(f"📡 Scraping: {self.url}")
        
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table', {'id': 'per_game_stats'})
            if not table:
                raise ValueError("Could not find the stats table")
            
            # Extract headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                headers.append(th.text.strip())
            
            # Extract data rows
            rows = []
            tbody = table.find('tbody')
            for tr in tbody.find_all('tr'):
                # Skip header rows that appear in the middle
                if tr.find('th') and tr.find('th').get('scope') == 'col':
                    continue
                    
                row = []
                for td in tr.find_all(['td', 'th']):
                    row.append(td.text.strip())
                
                if len(row) == len(headers):
                    rows.append(row)
            
            # Create DataFrame
            self.raw_data = pd.DataFrame(rows, columns=headers)
            print(f"✅ Successfully scraped {len(self.raw_data)} player records")
            print(f"📊 Columns: {list(self.raw_data.columns)}")
            
        except Exception as e:
            print(f"❌ Error scraping data: {e}")
            # Fallback: create sample data for demonstration
            print("🎯 Creating sample data for demonstration...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample NBA data for demonstration"""
        np.random.seed(42)
        n_players = 150
        
        players = [f"Player_{i}" for i in range(1, n_players + 1)]
        teams = ['LAL', 'GSW', 'BOS', 'MIA', 'PHX', 'MIL', 'DEN', 'NYK'] * (n_players // 8 + 1)
        positions = ['PG', 'SG', 'SF', 'PF', 'C'] * (n_players // 5 + 1)
        
        data = {
            'Player': players[:n_players],
            'Tm': teams[:n_players],
            'Pos': positions[:n_players],
            'Age': np.random.randint(19, 40, n_players),
            'G': np.random.randint(10, 82, n_players),
            'GS': np.random.randint(0, 82, n_players),
            'MP': np.round(np.random.uniform(8, 40, n_players), 1),
            'FG': np.round(np.random.uniform(1, 12, n_players), 1),
            'FGA': np.round(np.random.uniform(2, 25, n_players), 1),
            'FG%': np.round(np.random.uniform(0.3, 0.7, n_players), 3),
            '3P': np.round(np.random.uniform(0, 5, n_players), 1),
            '3PA': np.round(np.random.uniform(0, 12, n_players), 1),
            '3P%': np.round(np.random.uniform(0.2, 0.5, n_players), 3),
            'FT': np.round(np.random.uniform(0, 8, n_players), 1),
            'FTA': np.round(np.random.uniform(0, 10, n_players), 1),
            'FT%': np.round(np.random.uniform(0.5, 0.95, n_players), 3),
            'ORB': np.round(np.random.uniform(0, 5, n_players), 1),
            'DRB': np.round(np.random.uniform(0, 12, n_players), 1),
            'TRB': np.round(np.random.uniform(0, 15, n_players), 1),
            'AST': np.round(np.random.uniform(0, 12, n_players), 1),
            'STL': np.round(np.random.uniform(0, 3, n_players), 1),
            'BLK': np.round(np.random.uniform(0, 3, n_players), 1),
            'TOV': np.round(np.random.uniform(0, 5, n_players), 1),
            'PF': np.round(np.random.uniform(0, 5, n_players), 1),
            'PTS': np.round(np.random.uniform(2, 35, n_players), 1)
        }
        
        self.raw_data = pd.DataFrame(data)
        print(f"✅ Created sample dataset with {len(self.raw_data)} players")
    
    def transform_data(self):
        """🔄 Step 2: Clean and structure the data"""
        print("\n🔄 TRANSFORMING DATA...")
        
        df = self.raw_data.copy()
        
        # Convert numeric columns
        numeric_cols = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                       'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.dropna(subset=['PTS', 'MP'])  # Drop if missing key stats
        df = df.fillna(0)  # Fill other missing values with 0
        
        # Create derived features
        df['Efficiency'] = (df['PTS'] + df['TRB'] + df['AST']) / df['MP']
        df['Usage_Rate'] = (df['FGA'] + df['FTA'] * 0.44 + df['TOV']) / df['MP']
        df['TS%'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        
        # Create position categories
        df['Position_Group'] = df['Pos'].map({
            'PG': 'Guard', 'SG': 'Guard', 'G': 'Guard',
            'SF': 'Forward', 'PF': 'Forward', 'F': 'Forward',
            'C': 'Center'
        }).fillna('Guard')
        
        # Create performance tiers
        df['Scoring_Tier'] = pd.cut(df['PTS'], 
                                   bins=[0, 5, 10, 15, 20, float('inf')],
                                   labels=['Bench', 'Role Player', 'Starter', 'Star', 'Superstar'])
        
        # Remove outliers (players with very low minutes)
        df = df[df['MP'] >= 5]
        
        self.clean_data = df
        print(f"✅ Data cleaned: {len(df)} players remaining")
        print(f"📊 New features created: Efficiency, Usage_Rate, TS%, Position_Group, Scoring_Tier")
        
    def analyze_data(self):
        """📊 Step 3: Explore and analyze the data"""
        print("\n📊 ANALYZING DATA...")
        
        df = self.clean_data
        
        # Basic statistics
        print("\n🔍 BASIC STATISTICS:")
        print(f"Players: {len(df)}")
        print(f"Teams: {df['Tm'].nunique()}")
        print(f"Average PPG: {df['PTS'].mean():.1f}")
        print(f"Average RPG: {df['TRB'].mean():.1f}")
        print(f"Average APG: {df['AST'].mean():.1f}")
        
        # Top performers
        print("\n🏆 TOP PERFORMERS:")
        print("Scoring Leaders:")
        print(df.nlargest(5, 'PTS')[['Player', 'Tm', 'PTS', 'FG%', 'MP']].to_string(index=False))
        
        print("\nEfficiency Leaders (min 20 MPG):")
        eff_leaders = df[df['MP'] >= 20].nlargest(5, 'Efficiency')
        print(eff_leaders[['Player', 'Tm', 'Efficiency', 'PTS', 'TRB', 'AST']].to_string(index=False))
        
        # Position analysis
        print("\n📍 POSITION ANALYSIS:")
        pos_stats = df.groupby('Position_Group').agg({
            'PTS': 'mean',
            'TRB': 'mean', 
            'AST': 'mean',
            'FG%': 'mean'
        }).round(2)
        print(pos_stats)
        
        # Create visualizations
        self._create_visualizations()
        
    def _create_visualizations(self):
        """Create data visualizations"""
        df = self.clean_data
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🏀 NBA Player Statistics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Points distribution
        axes[0,0].hist(df['PTS'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0,0].set_title('Points Per Game Distribution')
        axes[0,0].set_xlabel('Points Per Game')
        axes[0,0].set_ylabel('Number of Players')
        axes[0,0].axvline(df['PTS'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["PTS"].mean():.1f}')
        axes[0,0].legend()
        
        # 2. Position comparison
        sns.boxplot(data=df, x='Position_Group', y='PTS', ax=axes[0,1])
        axes[0,1].set_title('Scoring by Position')
        axes[0,1].set_ylabel('Points Per Game')
        
        # 3. Efficiency vs Usage
        scatter = axes[1,0].scatter(df['Usage_Rate'], df['Efficiency'], 
                                   c=df['PTS'], cmap='viridis', alpha=0.6)
        axes[1,0].set_title('Efficiency vs Usage Rate')
        axes[1,0].set_xlabel('Usage Rate')
        axes[1,0].set_ylabel('Efficiency')
        plt.colorbar(scatter, ax=axes[1,0], label='Points Per Game')
        
        # 4. Scoring tiers
        tier_counts = df['Scoring_Tier'].value_counts()
        axes[1,1].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Player Distribution by Scoring Tier')
        
        plt.tight_layout()
        plt.show()
        
    def build_models(self):
        """🤖 Step 4: Build predictive models"""
        print("\n🤖 BUILDING MODELS...")
        
        df = self.clean_data.copy()
        
        # Prepare features
        feature_cols = ['Age', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                       'FT', 'FTA', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        
        # Add encoded categorical features
        position_dummies = pd.get_dummies(df['Position_Group'], prefix='Pos')
        
        X = pd.concat([df[feature_cols], position_dummies], axis=1)
        X = X.fillna(0)
        
        # Model 1: Regression - Predict Points Per Game
        print("\n🎯 MODEL 1: POINTS PREDICTION (Regression)")
        y_reg = df['PTS']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_reg.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"✅ RMSE: {rmse:.2f} points")
        print(f"📊 Model explains {rf_reg.score(X_test_scaled, y_test):.3f} of variance")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 TOP FEATURES FOR SCORING:")
        print(feature_importance.head(8).to_string(index=False))
        
        # Model 2: Classification - Predict All-Star potential
        print("\n⭐ MODEL 2: ALL-STAR PREDICTION (Classification)")
        
        # Create All-Star label (top performers)
        df['All_Star'] = ((df['PTS'] >= 20) & (df['Efficiency'] >= df['Efficiency'].quantile(0.8))).astype(int)
        
        y_clf = df['All_Star']
        X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_clf = rf_clf.predict(X_test_scaled)
        
        print(f"✅ Accuracy: {rf_clf.score(X_test_scaled, y_test):.3f}")
        print(f"📊 All-Star candidates identified: {sum(y_pred_clf)}/{len(y_pred_clf)}")
        
        print("\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred_clf, target_names=['Regular', 'All-Star']))
        
        # Show predicted All-Stars
        test_players = df.iloc[X_test.index]
        predicted_stars = test_players[y_pred_clf == 1]
        
        if len(predicted_stars) > 0:
            print("\n⭐ PREDICTED ALL-STAR CANDIDATES:")
            print(predicted_stars[['Player', 'Tm', 'PTS', 'TRB', 'AST', 'Efficiency']].to_string(index=False))
        
    def run_pipeline(self):
        """🚀 Run the complete pipeline"""
        print("🚀 STARTING NBA DATA PIPELINE")
        print("=" * 50)
        
        self.ingest_data()
        self.transform_data()
        self.analyze_data()
        self.build_models()
        
        print("\n" + "=" * 50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("💡 This pipeline demonstrated:")
        print("   • Web scraping with error handling")
        print("   • Data cleaning and feature engineering")
        print("   • Exploratory data analysis")
        print("   • Predictive modeling (regression + classification)")


# 🚀 RUN THE PIPELINE
if __name__ == "__main__":
    pipeline = NBADataPipeline()
    pipeline.run_pipeline()