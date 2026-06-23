import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    def __init__(self, lookback=60, forecast_horizon=5):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_data(self, df, target_column='Close'):
        """Prepare LSTM training data"""
        data = df[target_column].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback - self.forecast_horizon):
            X.append(scaled_data[i:i+self.lookback])
            y.append(scaled_data[i+self.lookback:i+self.lookback+self.forecast_horizon])
        
        return np.array(X), np.array(y).reshape(len(y), -1)
    
    def build_model(self, input_shape):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      callbacks=[early_stop], verbose=0)
        print(f"✓ LSTM Model trained: {self.model.count_params()} parameters")
    
    def predict(self, X_test):
        """Make predictions"""
        predictions = self.model.predict(X_test, verbose=0)
        return self.scaler.inverse_transform(predictions)

class PatternDetector:
    def __init__(self):
        self.patterns = {
            'bullish_engulfing': self.detect_bullish_engulfing,
            'bearish_engulfing': self.detect_bearish_engulfing,
            'hammer': self.detect_hammer,
            'shooting_star': self.detect_shooting_star,
            'morning_star': self.detect_morning_star,
            'evening_star': self.detect_evening_star
        }
    
    def detect_bullish_engulfing(self, df, idx):
        if idx < 1: return 0
        if df.iloc[idx-1]['Close'] >= df.iloc[idx-1]['Open'] and \
           df.iloc[idx]['Open'] < df.iloc[idx-1]['Close'] and \
           df.iloc[idx]['Close'] > df.iloc[idx-1]['Open']:
            return 1
        return 0
    
    def detect_bearish_engulfing(self, df, idx):
        if idx < 1: return 0
        if df.iloc[idx-1]['Open'] >= df.iloc[idx-1]['Close'] and \
           df.iloc[idx]['Close'] < df.iloc[idx-1]['Open'] and \
           df.iloc[idx]['Open'] > df.iloc[idx-1]['Close']:
            return 1
        return 0
    
    def detect_hammer(self, df, idx):
        body = abs(df.iloc[idx]['Close'] - df.iloc[idx]['Open'])
        lower_wick = df.iloc[idx]['Open'] - df.iloc[idx]['Low'] if df.iloc[idx]['Open'] < df.iloc[idx]['Close'] else df.iloc[idx]['Close'] - df.iloc[idx]['Low']
        if lower_wick > body * 2 and body > 0:
            return 1
        return 0
    
    def detect_shooting_star(self, df, idx):
        body = abs(df.iloc[idx]['Close'] - df.iloc[idx]['Open'])
        upper_wick = df.iloc[idx]['High'] - (df.iloc[idx]['Open'] if df.iloc[idx]['Open'] > df.iloc[idx]['Close'] else df.iloc[idx]['Close'])
        if upper_wick > body * 2 and body > 0:
            return 1
        return 0
    
    def detect_morning_star(self, df, idx):
        if idx < 2: return 0
        if df.iloc[idx-2]['Close'] < df.iloc[idx-2]['Open'] and \
           df.iloc[idx]['Close'] > df.iloc[idx]['Open'] and \
           df.iloc[idx]['Close'] > df.iloc[idx-2]['Open']:
            return 1
        return 0
    
    def detect_evening_star(self, df, idx):
        if idx < 2: return 0
        if df.iloc[idx-2]['Close'] > df.iloc[idx-2]['Open'] and \
           df.iloc[idx]['Close'] < df.iloc[idx]['Open'] and \
           df.iloc[idx]['Close'] < df.iloc[idx-2]['Open']:
            return 1
        return 0
    
    def detect_all_patterns(self, df):
        """Detect all patterns in dataframe"""
        for pattern_name, pattern_func in self.patterns.items():
            df[f'Pattern_{pattern_name}'] = df.index.map(lambda idx: pattern_func(df, idx))
        return df

class SupportResistance:
    @staticmethod
    def find_support_resistance(df, lookback=50):
        """Find support and resistance levels"""
        df['Local_High'] = df['High'].rolling(window=lookback, center=True).max()
        df['Local_Low'] = df['Low'].rolling(window=lookback, center=True).min()
        
        # Recent support and resistance
        recent_high = df['High'].tail(lookback).max()
        recent_low = df['Low'].tail(lookback).min()
        
        return {
            'recent_resistance': recent_high,
            'recent_support': recent_low,
            'level_difference': recent_high - recent_low
        }

class VolatilityAnalyzer:
    @staticmethod
    def calculate_volatility(df, period=20):
        """Calculate volatility metrics"""
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=period).std() * 100
        df['High_Vol'] = (df['Volatility'] > df['Volatility'].mean() + df['Volatility'].std()).astype(int)
        return df
    
    @staticmethod
    def get_current_volatility(df):
        """Get current volatility status"""
        current_vol = df['Volatility'].iloc[-1]
        avg_vol = df['Volatility'].mean()
        return {
            'current': current_vol,
            'average': avg_vol,
            'trend': 'High' if current_vol > avg_vol else 'Low'
        }

class TrendAnalyzer:
    @staticmethod
    def identify_trend(df, period=20):
        """Identify trend direction"""
        sma_fast = df['Close'].rolling(window=5).mean()
        sma_slow = df['Close'].rolling(window=period).mean()
        
        df['Trend'] = np.where(sma_fast > sma_slow, 1, -1)
        df['TrendStrength'] = abs(sma_fast - sma_slow) / sma_slow * 100
        
        return df
    
    @staticmethod
    def get_current_trend(df):
        """Get current trend information"""
        trend = df['Trend'].iloc[-1]
        strength = df['TrendStrength'].iloc[-1]
        
        return {
            'direction': 'Uptrend' if trend == 1 else 'Downtrend',
            'strength': strength,
            'signal': 'Strong' if strength > 1 else 'Weak'
        }
