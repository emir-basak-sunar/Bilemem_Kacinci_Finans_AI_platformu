package com.finai.ai.service;

import com.finai.common.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class QuotaService {

    private final StringRedisTemplate redisTemplate;

    private static final String QUOTA_PREFIX = "quota:ai:";

    @Value("${rate-limit.ai.requests-per-minute:20}")
    private int aiRateLimit;

    /**
     * Check and consume 1 AI credit. Returns remaining credits.
     * Uses Redis INCR for atomic counter.
     */
    public long consumeCredit(UUID userId, int monthlyLimit) {
        String key = QUOTA_PREFIX + userId + ":" + currentMonthKey();

        Long current = redisTemplate.opsForValue().increment(key);
        if (current != null && current == 1) {
            // First usage this month — set expiry to end of month + 1 day buffer
            redisTemplate.expire(key, Duration.ofDays(32));
        }

        if (current != null && current > monthlyLimit) {
            throw new BusinessException(
                    "AI quota exceeded. Used " + current + "/" + monthlyLimit + " this month.",
                    HttpStatus.TOO_MANY_REQUESTS
            );
        }

        log.info("AI credit consumed for user {}: {}/{}", userId, current, monthlyLimit);
        return monthlyLimit - (current != null ? current : 0);
    }

    public long getUsage(UUID userId) {
        String key = QUOTA_PREFIX + userId + ":" + currentMonthKey();
        String val = redisTemplate.opsForValue().get(key);
        return val != null ? Long.parseLong(val) : 0;
    }

    /**
     * Rate limit check per minute (sliding window).
     */
    public void checkRateLimit(UUID userId) {
        String key = "ratelimit:ai:" + userId + ":minute";
        Long count = redisTemplate.opsForValue().increment(key);

        if (count != null && count == 1) {
            redisTemplate.expire(key, Duration.ofMinutes(1));
        }

        if (count != null && count > aiRateLimit) {
            throw new BusinessException("Rate limit exceeded. Max " + aiRateLimit + " AI requests per minute.",
                    HttpStatus.TOO_MANY_REQUESTS);
        }
    }

    private String currentMonthKey() {
        var now = java.time.YearMonth.now();
        return now.getYear() + ":" + String.format("%02d", now.getMonthValue());
    }
}
