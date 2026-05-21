package com.finai.ai.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.finai.ai.dto.PredictionRequest;
import com.finai.common.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.util.retry.Retry;

import java.time.Duration;
import java.util.Map;
import java.util.UUID;

/**
 * AI Gateway Service — proxies prediction requests to the Python FastAPI backend.
 *
 * Security layers:
 * 1. Rate limiting (per-minute via Redis sliding window)
 * 2. Monthly quota enforcement (per-user credit system)
 * 3. Redis cache (avoid redundant AI calls)
 * 4. Retry with backoff (resilience against transient Python service failures)
 * 5. Circuit breaker pattern via timeout + error handling
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AiGatewayService {

    private final WebClient.Builder webClientBuilder;
    private final StringRedisTemplate redisTemplate;
    private final QuotaService quotaService;
    private final ObjectMapper objectMapper;

    @Value("${ai-service.base-url}")
    private String aiServiceUrl;

    @Value("${ai-service.timeout:30000}")
    private int timeout;

    private static final String CACHE_PREFIX = "ai:prediction:";

    /**
     * Execute a prediction request with full security pipeline.
     */
    public Map<String, Object> predict(UUID userId, PredictionRequest request, int monthlyQuota) {
        // 1. Check Redis cache FIRST — cache hits skip both rate limit and quota
        String cacheKey = buildCacheKey(request);
        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            log.debug("AI prediction cache HIT (no quota consumed): {}", cacheKey);
            try {
                Map<String, Object> result = objectMapper.readValue(cached, new TypeReference<>() {});
                result.put("_cache", "gateway_hit");
                return result;
            } catch (Exception e) {
                log.warn("Failed to deserialize AI cache, proceeding to Python service", e);
                redisTemplate.delete(cacheKey);
            }
        }

        // 2. Rate limit check — only on cache miss (actual computation)
        quotaService.checkRateLimit(userId);

        // 3. Consume monthly quota — only on cache miss
        quotaService.consumeCredit(userId, monthlyQuota);

        // 4. Call Python AI service with retry
        long startTime = System.currentTimeMillis();
        try {
            WebClient client = webClientBuilder.baseUrl(aiServiceUrl).build();

            String response = client.post()
                    .uri("/api/predict")
                    .bodyValue(Map.of(
                            "symbol", request.getSymbol(),
                            "model_type", request.getModelType(),
                            "period", request.getPeriod(),
                            "horizon", request.getHorizon()
                    ))
                    .retrieve()
                    .bodyToMono(String.class)
                    .retryWhen(Retry.backoff(2, Duration.ofMillis(500))
                            .filter(ex -> ex instanceof WebClientResponseException.ServiceUnavailable
                                    || ex instanceof WebClientResponseException.BadGateway)
                            .onRetryExhaustedThrow((spec, signal) ->
                                    new BusinessException("AI service unavailable after retries", HttpStatus.BAD_GATEWAY)))
                    .timeout(Duration.ofMillis(timeout))
                    .block();

            if (response == null) {
                throw new BusinessException("AI service returned empty response", HttpStatus.BAD_GATEWAY);
            }

            Map<String, Object> result = objectMapper.readValue(response, new TypeReference<>() {});

            // Check if Python returned an error
            if (result.containsKey("error")) {
                String errorMsg = String.valueOf(result.get("error"));
                log.warn("AI service returned error: {}", errorMsg);
                throw new BusinessException("AI prediction error: " + errorMsg, HttpStatus.UNPROCESSABLE_ENTITY);
            }

            long elapsed = System.currentTimeMillis() - startTime;
            result.put("_gateway_time_ms", elapsed);

            // 5. Cache successful result — 30 min TTL (same params = same result within market session)
            try {
                redisTemplate.opsForValue().set(cacheKey, response, Duration.ofMinutes(30));
                log.info("AI prediction cached 30min: {} ({}ms)", cacheKey, elapsed);
            } catch (Exception cacheEx) {
                log.warn("Failed to cache prediction (non-fatal): {}", cacheEx.getMessage());
            }

            return result;

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - startTime;
            log.error("AI service call failed after {}ms: {}", elapsed, e.getMessage());
            throw new BusinessException("AI prediction service unavailable", HttpStatus.BAD_GATEWAY);
        }
    }

    /**
     * Get quota usage info for a user.
     */
    public Map<String, Object> getQuotaInfo(UUID userId, int monthlyLimit) {
        long used = quotaService.getUsage(userId);
        return Map.of(
                "used", used,
                "limit", monthlyLimit,
                "remaining", Math.max(0, monthlyLimit - used)
        );
    }

    /**
     * List available models from the Python service.
     */
    public Map<String, Object> listModels() {
        try {
            WebClient client = webClientBuilder.baseUrl(aiServiceUrl).build();
            String response = client.get()
                    .uri("/api/models")
                    .retrieve()
                    .bodyToMono(String.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();

            if (response == null) return Map.of();
            return objectMapper.readValue(response, new TypeReference<>() {});
        } catch (Exception e) {
            log.warn("Failed to fetch model list: {}", e.getMessage());
            return Map.of("error", "Model list unavailable");
        }
    }

    private String buildCacheKey(PredictionRequest req) {
        return CACHE_PREFIX + req.getSymbol() + ":"
                + req.getModelType() + ":"
                + req.getPeriod() + ":"
                + req.getHorizon();
    }
}
