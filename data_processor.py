import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataProcessor:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all XAU data files"""
        timeframes = ['5m', '15m', '1h', '4h']
        for tf in timeframes:
            file_path = os.path.join(self.data_dir, f'XAU_{tf}_data.csv')
            if os.path.exists(file_path):
                self.data[tf] = self.load_csv(file_path)
                print(f"✓ Loaded {tf}: {len(self.data[tf])} candles")
            else:
                print(f"✗ File not found: {file_path}")
    
    def load_csv(self, file_path):
        """Load CSV with proper parsing"""
        df = pd.read_csv(file_path, sep=';')
        df['DateTime'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': int})
        df = df.sort_values('DateTime').reset_index(drop=True)
        return df
    
    def add_indicators(self, df, period=14):
        """Add technical indicators to dataframe"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Lower'] = sma - (std * 2)
        df['BB_Middle'] = sma
        
        # ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift()))
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        return df.fillna(0)
    
    def get_processed_data(self, timeframe='1h'):
        """Get data with all indicators"""
        if timeframe not in self.data:
            return None
        df = self.data[timeframe].copy()
        df = self.add_indicators(df)
        return df
    
    def get_time_features(self, df):
        """Extract time-based features"""
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['DayOfMonth'] = df['DateTime'].dt.day
        df['Month'] = df['DateTime'].dt.month
        df['IsAsiaSession'] = df['Hour'].isin(range(0, 9)).astype(int)
        df['IsEuropeSession'] = df['Hour'].isin(range(8, 17)).astype(int)
        df['IsUSSession'] = df['Hour'].isin(range(13, 22)).astype(int)
        return df
    
    def normalize_data(self, df, columns):
        """Normalize columns to 0-1 range"""
        df_norm = df.copy()
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        return df_norm
