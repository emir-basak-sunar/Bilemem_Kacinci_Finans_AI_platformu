# FinAI Platform — Kapsamlı Sistem Mimarisi

Bu doküman, FinAI Platformu'nun tüm mikroservislerini, veritabanı şemalarını, veri boru hatlarını (data pipelines) ve API uç noktalarını (endpoints) içermektedir. Yeni bir geliştirici (veya yapay zeka asistanı) sisteme katıldığında tüm yapısal bilgiyi buradan alabilir.

## 1. Sistem Mimarisine Genel Bakış

FinAI Platformu; asenkron, olay güdümlü (event-driven) ve mikroservis odaklı modern bir altyapıya sahiptir.

- **Frontend:** Next.js (React 19), Tailwind CSS, Zustand, Lightweight Charts.
- **Java Spring Backend (Core API):** Portföy, yetkilendirme, cüzdan, log ve bildirimleri yönetir.
- **Python AI Backend (Data Ingestion & ML):** Piyasa verilerini çeker, makine öğrenmesi modellerini çalıştırır ve LLM ile alım/satım (BUY/SELL) kararları üretir.
- **Mesajlaşma (Queue):** Apache Kafka (KRaft modunda). Tüm asenkron iletişim ve sinyaller Kafka topic'lerinden akar.
- **Veritabanları:** 
  - **TimescaleDB:** (PostgreSQL üzerine eklenti). Hem klasik ilişkisel tabloları hem de büyük zaman serisi verilerini (hypertables) tutar.
  - **Redis:** Caching (Önbellekleme), Rate Limiting (İstek sınırlama) ve ML Feature Cache (Özellik Önbelleği) için kullanılır.

### 1.1. Veri Akışı Şeması (Data Pipeline)
1. **Veri Toplama (Ingestion):** Python servisi, Alpaca/Finnhub üzerinden WebSocket ile piyasa verilerini alır ve FRED'den makro verileri çeker. Bunları `raw-prices` ve `macro-signals` Kafka topic'lerine yazar.
2. **Makine Öğrenmesi (ML Models):** Kafka'dan okunan ham veriler, zaman serisi (Time Series) modellerine girer (ARIMA, LSTM, TFT vs.). Üretilen tahminler `ts-model-signals` Kafka topic'ine yazılır.
3. **Signal Aggregator:** Bu sinyaller toplanıp Redis'e context olarak kaydedilir.
4. **LLM Kararı:** İnce ayar (fine-tune) yapılmış Büyük Dil Modeli, bu context'i kullanarak nihai `BUY`, `SELL`, `HOLD` kararlarını üretir ve `llm-decisions` Kafka topic'ine atar.
5. **Java Tüketimi:** Java Spring Backend `llm-decisions` topic'ini dinler, karar geldiğinde kullanıcılara bildirim gönderir ve cüzdan/portföy işlemlerini gerçekleştirir.

---

## 2. Veritabanı Şeması (TimescaleDB / PostgreSQL)

Veritabanı migration'ları Spring Boot üzerinden **Flyway** (`db/migration` klasörü) ile yönetilmektedir.

### 2.1 İlişkisel Tablolar (V1__init_schema.sql)
- **`users`:** Kullanıcı kimlik, yetki ve güvenlik bilgileri (email, password_hash (BCrypt), role).
- **`wallets`:** Kullanıcının bakiyelerini tutan cüzdan tablosu (user_id ile 1:1 ilişki, bakiye, para birimi).
- **`transactions`:** Para/Varlık transferi ve alım satım işlemleri geçmişi.
- **`subscription_plans`:** Abonelik paketleri (FREE, PRO, ENTERPRISE).
- **`user_subscriptions`:** Hangi kullanıcının hangi pakette olduğu.
- **`ai_usage`:** Kullanıcıların yapay zeka limit harcamalarının kaydı.
- **`notifications`:** Sistem içi bildirimler.
- **`audit_logs`:** Güvenlik ve aktivite logları.

### 2.2 Zaman Serisi Tabloları / Hypertables (V2__timescale_schema.sql)
TimescaleDB performans optimizasyonu (`create_hypertable`) uygulanmış özel tablolardır.
- **`price_bars`:** 1 günlük partitionlarla bölünmüş ham fiyat verileri (symbol, open, high, low, close, volume). 
  - Ayrıca `price_bars_1h` adında sürekli toplanan (continuous aggregate) bir Materialized View bulunur.
- **`macro_signals`:** Makroekonomik ve haber sentiment sinyalleri.
- **`model_signals`:** Bireysel makine öğrenmesi modellerinin ürettiği öngörüler.
- **`llm_decisions`:** LLM'in verdiği nihai alım/satım kararları.
- **`fine_tune_examples`:** (Standart tablo) LLM'in kendini yeniden eğitmesi için ayrılmış onaylanmış veri setleri.

---

## 3. Java Spring Boot Backend

### 3.1 Servisler ve Görevleri (`src/main/java/com/finai/`)
- **`auth/`:** JWT kullanarak `users` tablosu üzerinden yetkilendirme sağlar. Login, Kayıt, Şifre sıfırlama işlemleri.
- **`wallet/`:** Kullanıcı bakiyelerini günceller, işlem (`transaction`) geçmişi kaydeder. İyimser kilitleme (Optimistic Locking) veya idempontency mekanizmaları içerir.
- **`notification/`:** Sistem içi mesajları asenkron gönderir. **Kafka:** `finai.notifications` topic'ine mesaj yollar ve okur.
- **`audit/`:** Kullanıcı aktivitelerini (login, hatalı şifre denemesi vs.) Kafka (`finai.audit`) üzerinden arka planda veritabanına kaydeder.
- **`subscription/`:** Kullanıcıların PRO, ENTERPRISE paketlerini ve AI kotalarını (`ai_usage`) yönetir.
- **`ai/` & `market/`:** Python tarafındaki FastAPI'ye senkron istek (WebFlux WebClient ile) atması gereken durumlarda aracı görevi görür.

### 3.2 Kafka Konfigürasyonu
- `KafkaConfig.java` tarafında `finai.notifications` ve `finai.audit` topicleri tanımlıdır. 
- Servisler `KafkaTemplate<String, Object>` kullanarak mesaj atar, `@KafkaListener` anatasyonu ile mesajları dinler. JSON serileştirme aktiftir.

---

## 4. Python AI Backend (FastAPI)

### 4.1 Modüller (`backend/services/`)
- **`market_data.py`:** İstenilen enstrümanların borsa/kripto verilerini çeker (Alpaca, yfinance vs.).
- **`ai_models.py`:** Gelen veriler üzerinde XGBoost, LightGBM, CatBoost veya LSTM çalıştırıp sonraki bar tahminini oluşturur.
- **`redis_cache.py`:** Redis üzerinden caching, stampede protection (lock) işlemlerini yapar. Modeller her barda çalışmamak için veriyi buradan alır.

---

## 5. Altyapı ve Konteynerler (`docker-compose.yml`)
- **`postgres` (TimescaleDB):** Port: `5433:5432`. Tüm verinin tutulduğu ana bileşen.
- **`redis`:** Port: `6379`. Cache ve Rate Limiting (Spring Boot tarafında).
- **`kafka`:** Port: `9092`. Mesaj kuyruğu. KRaft modunda, zookeeper gereksinimi olmadan çalışır.
- **`kafka-ui`:** Port: `8081`. Kafka arayüzüne tarayıcıdan erişmek için.
- **`kafka-init`:** İlk kurulumda Kafka topiclerini (`raw-prices`, vb.) otomatik oluşturan betik.

---

## 6. Güvenlik ve Performans Notları
1. **Güvenlik:**
   - Stateless JWT Token sistemi aktiftir. Secret `application.yml` içindedir.
   - Şifreler BCrypt ile şifrelenir. 
2. **Rate Limiting:** Redis üzerinden API bazlı dakika kısıtlamaları uygulanır (Market: 60/dk, AI: 20/dk, Auth: 10/dk).
3. **Senkronizasyon:** Spring Boot'un yavaşlamaması için Audit Log ve Bildirim kayıt işlemleri **Kafka üzerinden Fire-and-Forget** (gönder-unut) asenkron mantığıyla çalışır.

_NOT: Yeni bir yapay zeka ajanının projeye dahil olduğunda ilk olarak `V1__init_schema.sql` ve `V2__timescale_schema.sql` dosyalarına, ardından Spring `application.yml`'e bakması önerilir._
