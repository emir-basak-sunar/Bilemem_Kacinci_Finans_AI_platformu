-- =============================================
-- FinAI Platform — Watchlist
-- =============================================

CREATE TABLE IF NOT EXISTS watchlist (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol      VARCHAR(10)  NOT NULL,
    name        VARCHAR(100),
    position    INT          NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id, position);
