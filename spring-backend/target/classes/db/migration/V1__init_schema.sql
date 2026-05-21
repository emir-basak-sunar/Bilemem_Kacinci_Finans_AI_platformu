-- =============================================
-- FinAI Platform — Initial Schema Migration
-- =============================================

-- Users
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    full_name       VARCHAR(100),
    role            VARCHAR(20)  NOT NULL DEFAULT 'USER',
    email_verified  BOOLEAN      NOT NULL DEFAULT FALSE,
    locked          BOOLEAN      NOT NULL DEFAULT FALSE,
    failed_attempts INT          NOT NULL DEFAULT 0,
    lock_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Wallets (1:1 with users)
CREATE TABLE wallets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    balance     DECIMAL(19,4) NOT NULL DEFAULT 0.0000,
    currency    VARCHAR(3)    NOT NULL DEFAULT 'USD',
    version     BIGINT        NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- Transactions
CREATE TABLE transactions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key     VARCHAR(64) UNIQUE NOT NULL,
    sender_wallet_id    UUID REFERENCES wallets(id),
    receiver_wallet_id  UUID REFERENCES wallets(id),
    amount              DECIMAL(19,4) NOT NULL CHECK (amount > 0),
    type                VARCHAR(20) NOT NULL,
    status              VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    description         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_txn_sender     ON transactions(sender_wallet_id, created_at DESC);
CREATE INDEX idx_txn_receiver   ON transactions(receiver_wallet_id, created_at DESC);
CREATE INDEX idx_txn_status     ON transactions(status);
CREATE INDEX idx_txn_idempotency ON transactions(idempotency_key);

-- Subscription Plans
CREATE TABLE subscription_plans (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name              VARCHAR(50)    NOT NULL UNIQUE,
    display_name      VARCHAR(100)   NOT NULL,
    monthly_price     DECIMAL(10,2)  NOT NULL DEFAULT 0.00,
    ai_quota_monthly  INT            NOT NULL DEFAULT 10,
    features          JSONB,
    active            BOOLEAN        NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

-- Seed default plans
INSERT INTO subscription_plans (name, display_name, monthly_price, ai_quota_monthly, features) VALUES
    ('FREE',       'Free Plan',       0.00,   10,   '{"market_data": true,  "ai_models": ["ensemble"], "export": false}'),
    ('PRO',        'Pro Plan',       29.99,  200,   '{"market_data": true,  "ai_models": ["ensemble","xgboost","lightgbm","catboost","lstm"], "export": true}'),
    ('ENTERPRISE', 'Enterprise Plan', 99.99, 2000,  '{"market_data": true,  "ai_models": ["ensemble","xgboost","lightgbm","catboost","lstm"], "export": true, "priority_support": true}');

-- User Subscriptions
CREATE TABLE user_subscriptions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_id     UUID NOT NULL REFERENCES subscription_plans(id),
    status      VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    starts_at   TIMESTAMPTZ  NOT NULL,
    expires_at  TIMESTAMPTZ  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sub_user   ON user_subscriptions(user_id, status);
CREATE INDEX idx_sub_expiry ON user_subscriptions(expires_at);

-- AI Usage Tracking
CREATE TABLE ai_usage (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_type  VARCHAR(20)  NOT NULL,
    symbol      VARCHAR(10)  NOT NULL,
    credits     INT          NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_ai_usage_user_month ON ai_usage(user_id, created_at);

-- Notifications
CREATE TABLE notifications (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type        VARCHAR(30)  NOT NULL,
    title       VARCHAR(200) NOT NULL,
    body        TEXT,
    read        BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_notif_user ON notifications(user_id, read, created_at DESC);

-- Audit Logs
CREATE TABLE audit_logs (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID,
    action      VARCHAR(50)  NOT NULL,
    entity_type VARCHAR(50),
    entity_id   VARCHAR(100),
    details     JSONB,
    ip_address  VARCHAR(45),
    user_agent  TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_audit_user ON audit_logs(user_id, created_at DESC);
CREATE INDEX idx_audit_action ON audit_logs(action, created_at DESC);

-- Refresh Tokens (for tracking/revocation)
CREATE TABLE refresh_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  VARCHAR(255) UNIQUE NOT NULL,
    device_info VARCHAR(255),
    expires_at  TIMESTAMPTZ  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_refresh_user ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_expiry ON refresh_tokens(expires_at);
