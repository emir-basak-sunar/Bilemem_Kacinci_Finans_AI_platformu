-- =============================================
-- FinAI Platform — Seed Test User
-- =============================================

-- We insert a test user with email 'admin@finai.com' and password 'Test1234!'
-- The BCrypt hash for 'Test1234!' is '$2a$10$8.UnVuG9HHgffUDAlk8qfOuVGkqRzgVymGe07xd00DMxs.TsphxXK'
-- Note: You can generate BCrypt hashes via various online tools or Spring Security's BCryptPasswordEncoder.

INSERT INTO users (id, email, password_hash, full_name, role, email_verified)
VALUES (
    '11111111-1111-1111-1111-111111111111', 
    'admin@finai.com', 
    '$2a$10$8.UnVuG9HHgffUDAlk8qfOuVGkqRzgVymGe07xd00DMxs.TsphxXK', 
    'Admin User', 
    'ADMIN', 
    true
)
ON CONFLICT (email) DO NOTHING;

-- Create wallet for the test user
INSERT INTO wallets (id, user_id, balance, currency)
VALUES (
    '22222222-2222-2222-2222-222222222222', 
    '11111111-1111-1111-1111-111111111111', 
    10000.0000, 
    'USD'
)
ON CONFLICT (user_id) DO NOTHING;

-- Give the test user the PRO subscription plan
INSERT INTO user_subscriptions (id, user_id, plan_id, status, starts_at, expires_at)
SELECT 
    '33333333-3333-3333-3333-333333333333',
    '11111111-1111-1111-1111-111111111111',
    p.id,
    'ACTIVE',
    NOW(),
    NOW() + INTERVAL '1 year'
FROM subscription_plans p
WHERE p.name = 'PRO'
ON CONFLICT DO NOTHING;
