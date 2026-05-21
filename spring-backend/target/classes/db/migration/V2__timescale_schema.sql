-- =============================================
-- FinAI Platform — TimescaleDB Schema Migration
-- =============================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. price_bars
CREATE TABLE price_bars (
    "timestamp" TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    "open" DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    "close" DOUBLE PRECISION,
    volume BIGINT,
    source VARCHAR(50)
);
SELECT create_hypertable('price_bars', by_range('timestamp', INTERVAL '1 day'));

-- 2. macro_signals
CREATE TABLE macro_signals (
    collected_at TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL,
    signal_key VARCHAR(100) NOT NULL,
    value JSONB,
    raw_text TEXT
);
SELECT create_hypertable('macro_signals', by_range('collected_at', INTERVAL '7 days'));

-- 3. model_signals
CREATE TABLE model_signals (
    generated_at TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    horizon_hours INT,
    metadata JSONB
);
SELECT create_hypertable('model_signals', by_range('generated_at', INTERVAL '7 days'));

-- 4. llm_decisions
CREATE TABLE llm_decisions (
    decided_at TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    decision VARCHAR(10) CHECK (decision IN ('BUY', 'SELL', 'HOLD')),
    confidence DOUBLE PRECISION,
    reasoning TEXT,
    signals_snapshot JSONB,
    actual_outcome VARCHAR(50),
    outcome_recorded_at TIMESTAMPTZ
);
SELECT create_hypertable('llm_decisions', by_range('decided_at', INTERVAL '30 days'));

-- 5. fine_tune_examples (Regular table)
CREATE TABLE fine_tune_examples (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(10) NOT NULL,
    date_range TSRANGE,
    input_context JSONB,
    model_decision VARCHAR(10),
    actual_outcome VARCHAR(50),
    outcome_pct_change DOUBLE PRECISION,
    is_validated BOOLEAN DEFAULT false
);

-- Appropriate indexes as requested
CREATE INDEX idx_price_bars_symbol_time ON price_bars (symbol, "timestamp" DESC);
CREATE INDEX idx_model_signals_symbol_time ON model_signals (symbol, generated_at DESC);
CREATE INDEX idx_llm_decisions_symbol_time ON llm_decisions (symbol, decided_at DESC);

-- Continuous aggregate for 1h OHLCV bars
CREATE MATERIALIZED VIEW price_bars_1h
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', "timestamp") AS bucket,
       symbol,
       first("open", "timestamp") AS "open",
       max(high) AS high,
       min(low) AS low,
       last("close", "timestamp") AS "close",
       sum(volume) AS volume
FROM price_bars
GROUP BY bucket, symbol
WITH NO DATA;
