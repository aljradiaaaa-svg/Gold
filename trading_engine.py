import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradingEngine:
    def __init__(self, initial_balance=10000, risk_percent=2):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percent = risk_percent
        self.trades = []
        self.active_trade = None
        self.trade_history = []
        self.equity_curve = [initial_balance]
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk"""
        risk_amount = self.balance * (self.risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    
    def open_trade(self, signal, entry_price, stop_loss, take_profit, datetime_str, confidence=0.5):
        """Open a new trade"""
        if self.active_trade is not None:
            return False
        
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        if position_size <= 0:
            return False
        
        self.active_trade = {
            'signal': signal,  # 'BUY' or 'SELL'
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime_str,
            'position_size': position_size,
            'confidence': confidence,
            'pnl': 0,
            'pnl_percent': 0
        }
        
        self.trades.append({
            'type': 'OPEN',
            'signal': signal,
            'price': entry_price,
            'size': position_size,
            'time': datetime_str,
            'balance': self.balance
        })
        
        return True
    
    def close_trade(self, exit_price, exit_reason, datetime_str):
        """Close active trade"""
        if self.active_trade is None:
            return False
        
        trade = self.active_trade
        
        if trade['signal'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - exit_price) * trade['position_size']
        
        pnl_percent = (pnl / (self.balance - pnl)) * 100 if (self.balance - pnl) != 0 else 0
        
        self.balance += pnl
        
        self.trade_history.append({
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'entry_time': trade['entry_time'],
            'exit_time': datetime_str,
            'signal': trade['signal'],
            'size': trade['position_size'],
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'reason': exit_reason,
            'confidence': trade['confidence']
        })
        
        self.equity_curve.append(self.balance)
        self.active_trade = None
        
        self.trades.append({
            'type': 'CLOSE',
            'price': exit_price,
            'pnl': pnl,
            'time': datetime_str,
            'reason': exit_reason,
            'balance': self.balance
        })
        
        return True
    
    def update_trade(self, current_price, datetime_str):
        """Update active trade P&L"""
        if self.active_trade is None:
            return
        
        if self.active_trade['signal'] == 'BUY':
            pnl = (current_price - self.active_trade['entry_price']) * self.active_trade['position_size']
        else:
            pnl = (self.active_trade['entry_price'] - current_price) * self.active_trade['position_size']
        
        self.active_trade['pnl'] = pnl
        self.active_trade['pnl_percent'] = (pnl / self.balance) * 100
    
    def get_statistics(self):
        """Calculate trading statistics"""
        if len(self.trade_history) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_loss': 0,
                'net_profit': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'return_percent': 0
            }
        
        trades_df = pd.DataFrame(self.trade_history)
        
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        total_profit = winning['pnl'].sum()
        total_loss = abs(losing['pnl'].sum())
        net_profit = total_profit - total_loss
        
        win_rate = (len(winning) / len(trades_df) * 100) if len(trades_df) > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'average_win': winning['pnl'].mean() if len(winning) > 0 else 0,
            'average_loss': losing['pnl'].mean() if len(losing) > 0 else 0,
            'largest_win': winning['pnl'].max() if len(winning) > 0 else 0,
            'largest_loss': losing['pnl'].min() if len(losing) > 0 else 0,
            'return_percent': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'current_balance': self.balance
        }

class SignalGenerator:
    def __init__(self, pattern_detector, trend_analyzer, volatility_analyzer, lstm_predictor=None):
        self.pattern_detector = pattern_detector
        self.trend_analyzer = trend_analyzer
        self.volatility_analyzer = volatility_analyzer
        self.lstm_predictor = lstm_predictor
    
    def generate_signals(self, df, current_idx):
        """Generate trading signals based on multiple factors"""
        signals = []
        confidences = []
        
        if current_idx < 5:
            return None, 0
        
        current_row = df.iloc[current_idx]
        prev_row = df.iloc[current_idx - 1]
        
        # Pattern signals
        bullish_patterns = sum([
            current_row.get('Pattern_bullish_engulfing', 0),
            current_row.get('Pattern_hammer', 0),
            current_row.get('Pattern_morning_star', 0)
        ])
        
        bearish_patterns = sum([
            current_row.get('Pattern_bearish_engulfing', 0),
            current_row.get('Pattern_shooting_star', 0),
            current_row.get('Pattern_evening_star', 0)
        ])
        
        # Trend signal
        trend = df['Trend'].iloc[current_idx]
        trend_strength = df['TrendStrength'].iloc[current_idx]
        
        # RSI signal
        rsi = current_row['RSI']
        rsi_signal = 0
        if rsi < 30:
            rsi_signal = 1  # Oversold - BUY
        elif rsi > 70:
            rsi_signal = -1  # Overbought - SELL
        
        # MACD signal
        macd_signal = 0
        if current_row['MACD'] > current_row['Signal'] and prev_row['MACD'] <= prev_row['Signal']:
            macd_signal = 1  # BUY
        elif current_row['MACD'] < current_row['Signal'] and prev_row['MACD'] >= prev_row['Signal']:
            macd_signal = -1  # SELL
        
        # Bollinger Bands signal
        bb_signal = 0
        if current_row['Close'] < current_row['BB_Lower']:
            bb_signal = 1  # BUY
        elif current_row['Close'] > current_row['BB_Upper']:
            bb_signal = -1  # SELL
        
        # Combine signals
        buy_signals = bullish_patterns + max(0, rsi_signal) + max(0, macd_signal) + max(0, bb_signal) + max(0, trend)
        sell_signals = bearish_patterns + max(0, -rsi_signal) + max(0, -macd_signal) + max(0, -bb_signal) + max(0, -trend)
        
        confidence = max(buy_signals, sell_signals) / 8.0
        
        if buy_signals > sell_signals:
            return 'BUY', confidence
        elif sell_signals > buy_signals:
            return 'SELL', confidence
        else:
            return None, 0
    
    def get_entry_exit_points(self, df, current_idx, signal):
        """Calculate entry, stop loss, and take profit levels"""
        current_price = df['Close'].iloc[current_idx]
        atr = df['ATR'].iloc[current_idx]
        
        if signal == 'BUY':
            entry = current_price
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        else:  # SELL
            entry = current_price
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        
        return entry, stop_loss, take_profit
