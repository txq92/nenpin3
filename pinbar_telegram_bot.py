#!/usr/bin/env python3
"""
Pin Bar Detection Bot - Optimized Version
Chỉ phát hiện Pin Bar ở khung 5m và 15m, gửi qua Telegram
Không sử dụng DB, không ghi log
"""

import asyncio
import pandas as pd
import numpy as np
import ta
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional
from telegram import Bot
from telegram.constants import ParseMode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
if not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_CHAT_ID not found in .env file")

# Trading pairs để monitor
TRADING_PAIRS = ['SUIUSDT', 'ETHUSDT', 'HYPEUSDT', 'SOLUSDT', 'ADAUSDT']

# Timeframes để scan
TIMEFRAMES = ['3m', '15m']  # 3m for signals, 15m for trend confirmation

# Technical Indicators Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Pin Bar Detection Parameters
PIN_BAR_MIN_BODY_RATIO = 0.3   # Body <= 30% tổng chiều dài nến
PIN_BAR_MIN_TAIL_RATIO = 2.0   # Tail >= 2x body  
PIN_BAR_MIN_RANGE_RATIO = 1.5  # Range >= 1.5x ATR

# Scan interval (seconds)
SCAN_INTERVAL = 120  # 2 phút - phù hợp với timeframe 3m

# ==================== BINANCE CLIENT ====================
class SimpleBinanceClient:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu nến từ Binance"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception:
            return None

# ==================== TECHNICAL ANALYSIS ====================
class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict:
        """Tính toán các chỉ số kỹ thuật"""
        try:
            # RSI và Stochastic RSI
            rsi = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
            stoch_rsi_k = stoch_rsi.rolling(3).mean()  # %K line
            stoch_rsi_d = stoch_rsi_k.rolling(3).mean()  # %D line
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            macd_histogram = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_middle = bb.bollinger_mavg()
            bb_lower = bb.bollinger_lband()
            
            # EMA
            ema20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            ema50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            # ATR
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume SMA
            volume_sma = df['volume'].rolling(window=20).mean()
            
            # Support/Resistance (simplified)
            recent_high = df['high'].rolling(window=20).max().iloc[-1]
            recent_low = df['low'].rolling(window=20).min().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            return {
                'rsi': rsi.iloc[-1] if len(rsi) > 0 else None,
                'stoch_rsi_k': stoch_rsi_k.iloc[-1] if len(stoch_rsi_k) > 0 else None,
                'stoch_rsi_d': stoch_rsi_d.iloc[-1] if len(stoch_rsi_d) > 0 else None,
                'macd': macd_line.iloc[-1] if len(macd_line) > 0 else None,
                'macd_signal': macd_signal.iloc[-1] if len(macd_signal) > 0 else None,
                'macd_histogram': macd_histogram.iloc[-1] if len(macd_histogram) > 0 else None,
                'bb_upper': bb_upper.iloc[-1] if len(bb_upper) > 0 else None,
                'bb_middle': bb_middle.iloc[-1] if len(bb_middle) > 0 else None,
                'bb_lower': bb_lower.iloc[-1] if len(bb_lower) > 0 else None,
                'ema20': ema20.iloc[-1] if len(ema20) > 0 else None,
                'ema50': ema50.iloc[-1] if len(ema50) > 0 else None,
                'atr': atr.iloc[-1] if len(atr) > 0 else None,
                'volume_ratio': df['volume'].iloc[-1] / volume_sma.iloc[-1] if len(volume_sma) > 0 and volume_sma.iloc[-1] > 0 else 1,
                'resistance': recent_high,
                'support': recent_low,
                'current_price': current_price
            }
        except:
            return {}
    
    @staticmethod
    def detect_pin_bar(df: pd.DataFrame) -> Dict:
        """Phát hiện Pin Bar pattern"""
        try:
            candle = df.iloc[-1]
            
            open_price = candle['open']
            high_price = candle['high']
            low_price = candle['low']
            close_price = candle['close']
            
            body = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            if total_range == 0 or body == 0:
                return {'is_pin_bar': False}
            
            body_ratio = body / total_range
            
            # Tính ATR
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else total_range
            range_ratio = total_range / current_atr if current_atr > 0 else 1
            
            # Kiểm tra điều kiện Pin Bar
            is_small_body = body_ratio <= PIN_BAR_MIN_BODY_RATIO
            is_significant_range = range_ratio >= PIN_BAR_MIN_RANGE_RATIO
            
            pin_bar_type = None
            tail_ratio = 0
            
            if is_small_body and is_significant_range:
                # Bullish Pin Bar
                if lower_shadow > 0 and lower_shadow / body >= PIN_BAR_MIN_TAIL_RATIO and lower_shadow > upper_shadow:
                    pin_bar_type = 'BULLISH'
                    tail_ratio = lower_shadow / body
                # Bearish Pin Bar
                elif upper_shadow > 0 and upper_shadow / body >= PIN_BAR_MIN_TAIL_RATIO and upper_shadow > lower_shadow:
                    pin_bar_type = 'BEARISH'
                    tail_ratio = upper_shadow / body
            
            if pin_bar_type:
                strength = min(100, (tail_ratio * 20) + (range_ratio * 20) + ((1 - body_ratio) * 60))
                return {
                    'is_pin_bar': True,
                    'type': pin_bar_type,
                    'strength': round(strength, 1),
                    'tail_ratio': round(tail_ratio, 2)
                }
            
            return {'is_pin_bar': False}
            
        except:
            return {'is_pin_bar': False}

# ==================== SIGNAL ANALYZER ====================
class SignalAnalyzer:
    @staticmethod
    def analyze_trend_15m(df_15m: pd.DataFrame) -> Dict:
        """Phân tích xu hướng trên khung 15m"""
        try:
            indicators = TechnicalAnalysis.calculate_indicators(df_15m)
            if not indicators:
                return {'trend': None}
            
            current_price = indicators['current_price']
            ema20 = indicators.get('ema20')
            ema50 = indicators.get('ema50')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            
            # Xác định xu hướng
            trend = None
            trend_strength = 0
            reasons = []
            
            if ema20 and ema50:
                if ema20 > ema50 and current_price > ema20:
                    trend = 'UPTREND'
                    trend_strength += 40
                    reasons.append("Price > EMA20 > EMA50 (15m)")
                elif ema20 < ema50 and current_price < ema20:
                    trend = 'DOWNTREND'
                    trend_strength += 40
                    reasons.append("Price < EMA20 < EMA50 (15m)")
            
            if macd and macd_signal:
                if macd > macd_signal:
                    if trend == 'UPTREND':
                        trend_strength += 20
                    elif not trend:
                        trend = 'UPTREND'
                        trend_strength += 30
                    reasons.append("MACD above signal (15m)")
                elif macd < macd_signal:
                    if trend == 'DOWNTREND':
                        trend_strength += 20
                    elif not trend:
                        trend = 'DOWNTREND'
                        trend_strength += 30
                    reasons.append("MACD below signal (15m)")
            
            return {
                'trend': trend,
                'strength': trend_strength,
                'reasons': reasons
            }
        except:
            return {'trend': None}
    
    @staticmethod
    def analyze_signal(pin_bar: Dict, indicators: Dict, trend_15m: Dict = None) -> Optional[Dict]:
        """Phân tích và đánh giá tín hiệu"""
        if not pin_bar.get('is_pin_bar'):
            return None
        
        confidence = 50  # Base score
        signal_type = None
        reasons = []
        
        pin_bar_type = pin_bar.get('type')
        strength = pin_bar.get('strength', 0)
        
            # Pin Bar strength bonus
        confidence += min(20, strength * 0.25)
        reasons.append(f"{pin_bar_type} Pin Bar (strength: {strength}%)")
        
        # Trend 15m analysis và tỷ lệ thắng dự kiến
        win_rate = 50  # Tỷ lệ cơ bản
        trend_status = "NO_TREND"
        
        if trend_15m and trend_15m.get('trend'):
            trend_align = (
                (pin_bar_type == 'BULLISH' and trend_15m['trend'] == 'UPTREND') or
                (pin_bar_type == 'BEARISH' and trend_15m['trend'] == 'DOWNTREND')
            )
            if trend_align:
                confidence += min(25, trend_15m['strength'])
                win_rate += min(30, trend_15m['strength'] * 0.6)  # Tăng tỷ lệ thắng khi thuận xu hướng
                reasons.extend([f"Aligned with {trend_15m['trend']} (15m)"] + trend_15m['reasons'])
                trend_status = "WITH_TREND"
            else:
                confidence -= 20  # Penalty for trading against the trend
                win_rate -= 15  # Giảm tỷ lệ thắng khi nghịch xu hướng
                reasons.append(f"Warning: Against {trend_15m['trend']} (15m)")
                trend_status = "AGAINST_TREND"
        
        # Điều chỉnh tỷ lệ thắng dựa trên các chỉ báo khác
        if pin_bar.get('strength', 0) > 80:
            win_rate += 10
        if volume_ratio > 2:
            win_rate += 5
            
        rsi = indicators.get('rsi')
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        volume_ratio = indicators.get('volume_ratio', 1)
        
        current_price = indicators['current_price']
        
        # BULLISH Signal Analysis
        if pin_bar_type == 'BULLISH':
            signal_type = 'LONG'
            
            # RSI và Stochastic RSI Analysis
            if rsi and rsi < RSI_OVERSOLD:
                confidence += 15
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi and rsi < 50:
                confidence += 8
            
            stoch_k = indicators.get('stoch_rsi_k')
            stoch_d = indicators.get('stoch_rsi_d')
            if stoch_k and stoch_d and stoch_k > stoch_d and stoch_k < 20:
                confidence += 12
                reasons.append("Stoch RSI bullish crossover in oversold")
            
            # MACD Analysis
            if macd and macd_signal and macd > macd_signal:
                confidence += 10
                reasons.append("MACD bullish crossover")
            
            # Bollinger Bands Analysis
            bb_lower = indicators.get('bb_lower')
            if bb_lower and current_price <= bb_lower:
                confidence += 12
                reasons.append("Price at/below BB lower band")
            
            # EMA Analysis
            ema20 = indicators.get('ema20')
            ema50 = indicators.get('ema50')
            if ema20 and ema50:
                if current_price > ema20:
                    confidence += 8
                    if ema20 > ema50:
                        confidence += 7
                        reasons.append("Price > EMA20 > EMA50")
            
            # Volume Analysis
            if volume_ratio > 1.5:
                confidence += 10
                reasons.append(f"High volume ({volume_ratio:.1f}x)")
        
        # BEARISH Signal Analysis
        elif pin_bar_type == 'BEARISH':
            signal_type = 'SHORT'
            
            # RSI và Stochastic RSI Analysis
            if rsi and rsi > RSI_OVERBOUGHT:
                confidence += 15
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif rsi and rsi > 50:
                confidence += 8
            
            stoch_k = indicators.get('stoch_rsi_k')
            stoch_d = indicators.get('stoch_rsi_d')
            if stoch_k and stoch_d and stoch_k < stoch_d and stoch_k > 80:
                confidence += 12
                reasons.append("Stoch RSI bearish crossover in overbought")
            
            # MACD Analysis
            if macd and macd_signal and macd < macd_signal:
                confidence += 10
                reasons.append("MACD bearish crossover")
            
            # Bollinger Bands Analysis
            bb_upper = indicators.get('bb_upper')
            if bb_upper and current_price >= bb_upper:
                confidence += 12
                reasons.append("Price at/above BB upper band")
            
            # EMA Analysis
            ema20 = indicators.get('ema20')
            ema50 = indicators.get('ema50')
            if ema20 and ema50:
                if current_price < ema20:
                    confidence += 8
                    if ema20 < ema50:
                        confidence += 7
                        reasons.append("Price < EMA20 < EMA50")
            
            # Volume Analysis
            if volume_ratio > 1.5:
                confidence += 10
                reasons.append(f"High volume ({volume_ratio:.1f}x)")
        
        # Return signal for both trend-aligned and counter-trend signals
        if confidence >= 40:  # Giảm ngưỡng để cho phép cả tín hiệu nghịch xu hướng
            win_rate = min(90, max(35, win_rate))  # Giới hạn tỷ lệ thắng từ 35-90%
            return {
                'type': signal_type,
                'confidence': min(100, confidence),
                'reasons': reasons,
                'win_rate': round(win_rate, 1),
                'trend_status': trend_status
            }
        
        return None

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
    
    async def send_signal(self, symbol: str, timeframe: str, signal: Dict, indicators: Dict, pin_bar: Dict):
        """Gửi tín hiệu qua Telegram"""
        try:
            # Format message
            emoji = "🟢" if signal['type'] == 'LONG' else "🔴"
            arrow = "📈" if signal['type'] == 'LONG' else "📉"
            
            # Thêm emoji cho trạng thái xu hướng
            trend_emoji = "↔️"
            if signal.get('trend_status') == "WITH_TREND":
                trend_emoji = "✅"
            elif signal.get('trend_status') == "AGAINST_TREND":
                trend_emoji = "⚠️"

            message = f"{emoji} <b>PIN BAR SIGNAL</b> {arrow}\n\n"
            message += f"💰 <b>Symbol:</b> {symbol}\n"
            message += f"⏰ <b>Timeframe:</b> {timeframe}\n"
            message += f"📊 <b>Signal:</b> <b>{signal['type']}</b> {trend_emoji}\n"
            message += f"🎯 <b>Confidence:</b> {signal['confidence']}%\n"
            message += f"📈 <b>Win Rate:</b> {signal.get('win_rate', 50)}%\n"
            message += f"💵 <b>Price:</b> ${indicators['current_price']:.4f}\n\n"
            
            # Technical details
            message += f"📈 <b>Indicators:</b>\n"
            if indicators.get('rsi'):
                message += f"• RSI: {indicators['rsi']:.1f}\n"
            if indicators.get('macd'):
                message += f"• MACD: {indicators['macd']:.4f}\n"
            if indicators.get('volume_ratio'):
                message += f"• Volume: {indicators['volume_ratio']:.1f}x\n\n"
            
            # Signal reasons
            message += f"✅ <b>Reasons:</b>\n"
            for reason in signal['reasons']:
                message += f"• {reason}\n"
            
            # Risk levels
            if signal['type'] == 'LONG':
                stop_loss = indicators['current_price'] * 0.98
                take_profit = indicators['current_price'] * 1.04
            else:
                stop_loss = indicators['current_price'] * 1.02
                take_profit = indicators['current_price'] * 0.96
            
            message += f"\n⚠️ <b>Risk Management:</b>\n"
            message += f"• Stop Loss: ${stop_loss:.4f}\n"
            message += f"• Take Profit: ${take_profit:.4f}\n\n"
            
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send message
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            return True
            
        except Exception:
            return False

# ==================== MAIN BOT ====================
class PinBarTelegramBot:
    def __init__(self):
        self.binance = SimpleBinanceClient()
        self.telegram = TelegramNotifier()
        self.last_signals = {}  # Prevent duplicate signals
        self.is_running = False
    
    async def scan_symbol(self, symbol: str, timeframe: str):
        """Scan một symbol và timeframe"""
        # Chỉ tìm Pin Bar trên timeframe 3m
        if timeframe != '3m':
            return
            
        # Get market data cho cả 3m và 15m
        df_3m = self.binance.get_klines(symbol, '3m', limit=100)
        df_15m = self.binance.get_klines(symbol, '15m', limit=100)
        
        if df_3m is None or len(df_3m) < 50 or df_15m is None or len(df_15m) < 50:
            return
        
        # Detect Pin Bar trên 3m
        pin_bar = TechnicalAnalysis.detect_pin_bar(df_3m)
        if not pin_bar.get('is_pin_bar'):
            return
        
        # Calculate indicators cho 3m
        indicators = TechnicalAnalysis.calculate_indicators(df_3m)
        if not indicators:
            return
            
        # Analyze trend trên 15m
        trend_15m = SignalAnalyzer.analyze_trend_15m(df_15m)
        
        # Analyze signal với xác nhận xu hướng từ 15m
        signal = SignalAnalyzer.analyze_signal(pin_bar, indicators, trend_15m)
        if not signal:
            return
        
        # Check for duplicate
        signal_key = f"{symbol}_{timeframe}_{signal['type']}"
        current_time = datetime.now()
        
        if signal_key in self.last_signals:
            last_time = self.last_signals[signal_key]
            if (current_time - last_time).total_seconds() < 3600:  # 1 hour cooldown
                return
        
        # Send signal
        sent = await self.telegram.send_signal(symbol, timeframe, signal, indicators, pin_bar)
        if sent:
            self.last_signals[signal_key] = current_time
            print(f"✅ Signal sent: {symbol} {timeframe} {signal['type']} ({signal['confidence']}%)")
    
    async def scan_all(self):
        """Scan tất cả symbols và timeframes"""
        tasks = []
        for symbol in TRADING_PAIRS:
            for timeframe in TIMEFRAMES:
                tasks.append(self.scan_symbol(symbol, timeframe))
        
        await asyncio.gather(*tasks)
    
    async def start(self):
        """Bắt đầu bot monitoring"""
        self.is_running = True
        print("🤖 Pin Bar Telegram Bot Started")
        print(f"📊 Monitoring: {', '.join(TRADING_PAIRS)}")
        print(f"⏰ Timeframes: {', '.join(TIMEFRAMES)}")
        print(f"🔄 Scan interval: {SCAN_INTERVAL} seconds")
        print("-" * 50)
        
        # Send startup message
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text="🤖 <b>Pin Bar Bot Started</b>\n\n"
                     f"📊 Monitoring: {', '.join(TRADING_PAIRS)}\n"
                     f"⏰ Timeframes: {', '.join(TIMEFRAMES)}\n"
                     f"🔄 Scanning every {SCAN_INTERVAL//60} minutes",
                parse_mode='HTML'
            )
        except:
            pass
        
        while self.is_running:
            try:
                await self.scan_all()
                await asyncio.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception:
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        print("\n🛑 Bot stopped")

# ==================== MAIN FUNCTION ====================
async def main():
    bot = PinBarTelegramBot()
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
