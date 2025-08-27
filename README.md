# Pin Bar Detection Bot

Bot tự động phát hiện mô hình nến Pin Bar và gửi tín hiệu qua Telegram.

## Cách Hoạt Động

### 1. Timeframes và Scanning
- **Chart 3 phút**: Sử dụng để phát hiện Pin Bar và tín hiệu giao dịch
- **Chart 15 phút**: Sử dụng để xác định xu hướng thị trường
- **Tần suất quét**: 2 phút/lần
- **Cặp tiền được theo dõi**: SUIUSDT, ETHUSDT, HYPEUSDT, SOLUSDT, ADAUSDT

### 2. Điều Kiện Pin Bar
- Body nhỏ (≤ 30% tổng chiều dài nến)
- Bóng nến dài (≥ 2x body)
- Range đủ lớn (≥ 1.5x ATR)

### 3. Phân Tích Xu Hướng (15 phút)
Bot phân tích xu hướng dựa trên:
- EMA20 và EMA50
- MACD
- Độ mạnh xu hướng được tính dựa trên các yếu tố trên

### 4. Điều Kiện Gửi Tín Hiệu

#### 4.1 Khi Nào Gửi Tín Hiệu
Bot sẽ gửi tín hiệu khi:
- Phát hiện Pin Bar hợp lệ trên chart 3 phút
- Độ tin cậy (confidence) ≥ 40%
- Không có tín hiệu trùng lặp trong vòng 1 giờ qua

#### 4.2 Phân Loại Tín Hiệu
1. **Tín Hiệu Thuận Xu Hướng** (✅)
   - Pin Bar xuất hiện cùng chiều xu hướng 15 phút
   - Độ tin cậy và tỷ lệ thắng cao hơn
   - Thêm 25% độ tin cậy từ xu hướng

2. **Tín Hiệu Nghịch Xu Hướng** (⚠️)
   - Pin Bar xuất hiện ngược chiều xu hướng 15 phút
   - Giảm 20% độ tin cậy
   - Giảm 15% tỷ lệ thắng

### 5. Tính Toán Điểm Số

#### Độ Tin Cậy (Confidence)
- Điểm cơ bản: 50%
- Pin Bar strength: +20% (max)
- Xu hướng 15m: +25% (thuận) hoặc -20% (nghịch)
- RSI: +15% (oversold/overbought)
- MACD: +10% (crossover)
- Bollinger Bands: +12% (price at bands)
- EMA: +15% (price & trend alignment)
- Volume: +10% (high volume)

#### Tỷ Lệ Thắng (Win Rate)
- Cơ bản: 50%
- Thuận xu hướng: +30% (max)
- Nghịch xu hướng: -15%
- Pin Bar mạnh (>80%): +10%
- Volume cao (>2x): +5%
- Giới hạn: 35-90%

### 6. Thông Tin Trong Tin Nhắn Telegram
```
🟢/🔴 PIN BAR SIGNAL 📈/📉

💰 Symbol: BTCUSDT
⏰ Timeframe: 3m
📊 Signal: LONG/SHORT ✅/⚠️
🎯 Confidence: XX%
📈 Win Rate: XX%
💵 Price: $XX,XXX

📈 Indicators:
• RSI: XX
• MACD: XX
• Volume: XXx

✅ Reasons:
• Pin Bar strength: XX%
• [Các lý do khác]

⚠️ Risk Management:
• Stop Loss: $XX,XXX
• Take Profit: $XX,XXX

⏰ YYYY-MM-DD HH:MM:SS
```

### 7. Cài Đặt và Cấu Hình

#### Yêu Cầu
- Python 3.x
- Các thư viện: requirements.txt

#### Cấu Hình (.env)
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

#### Chạy Bot
```bash
python pinbar_telegram_bot.py
```
