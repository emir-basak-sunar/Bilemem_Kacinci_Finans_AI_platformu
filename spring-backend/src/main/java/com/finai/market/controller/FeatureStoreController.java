package com.finai.market.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.common.exception.BusinessException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;

/**
 * Feature Store Controller
 *
 * Proxies feature store requests to the Python backend.
 * Public endpoints — no auth required.
 *
 * GET  /feature-store/registry          — full registry summary
 * GET  /feature-store/views             — list feature views
 * POST /feature-store/materialize/{sym} — compute + store features
 * GET  /feature-store/features/{sym}    — get online features
 * GET  /feature-store/features/{sym}/stats    — offline stats
 * GET  /feature-store/features/{sym}/validate — drift check
 */
@RestController
@RequestMapping("/feature-store")
@RequiredArgsConstructor
@Slf4j
public class FeatureStoreController {

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;

    @Value("${ai-service.base-url:http://localhost:8000}")
    private String pythonBaseUrl;

    private static final String CACHE_PREFIX = "spring:fs:";

    // ── Registry & Views (long cache — rarely changes) ────────────────────────

    @GetMapping("/registry")
    public ResponseEntity<ApiResponse<Object>> getRegistry() {
        return proxy("/api/feature-store/registry", "registry", Duration.ofMinutes(10));
    }

    @GetMapping("/views")
    public ResponseEntity<ApiResponse<Object>> listViews() {
        return proxy("/api/feature-store/views", "views", Duration.ofMinutes(10));
    }

    // ── Materialization (no cache — always fresh) ─────────────────────────────

    @PostMapping("/materialize/{symbol}")
    public ResponseEntity<ApiResponse<Object>> materialize(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "1y") String period) {

        String sym = symbol.toUpperCase();
        try {
            WebClient client = WebClient.create(pythonBaseUrl);
            String response = client.post()
                    .uri(uriBuilder -> uriBuilder
                            .path("/api/features/materialize/{symbol}")
                            .queryParam("period", period)
                            .build(sym))
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body -> {
                                log.warn("Feature store materialize error [{}]: {}", sym, body);
                                return new BusinessException("Materialization failed", HttpStatus.BAD_GATEWAY);
                            })
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(60));  // materialization can take time

            if (response == null) throw new BusinessException("Empty response", HttpStatus.BAD_GATEWAY);

            // Invalidate cached features for this symbol
            redisTemplate.delete(CACHE_PREFIX + "features:" + sym);
            redisTemplate.delete(CACHE_PREFIX + "stats:" + sym);

            Object data = objectMapper.readValue(response, Object.class);
            log.info("Features materialized for {}/{}", sym, period);
            return ResponseEntity.ok(ApiResponse.ok(data));

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("Materialization failed [{}]: {}", sym, e.getMessage());
            throw new BusinessException("Feature materialization failed", HttpStatus.BAD_GATEWAY);
        }
    }

    // ── Online Features (short cache — 5 min) ────────────────────────────────

    @GetMapping("/features/{symbol}")
    public ResponseEntity<ApiResponse<Object>> getFeatures(
            @PathVariable String symbol,
            @RequestParam(required = false) String views) {

        String sym = symbol.toUpperCase();
        String cacheKey = CACHE_PREFIX + "features:" + sym + (views != null ? ":" + views : "");

        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            try {
                return ResponseEntity.ok(ApiResponse.ok(objectMapper.readValue(cached, Object.class)));
            } catch (Exception ignored) {}
        }

        try {
            WebClient client = WebClient.create(pythonBaseUrl);
            String response = client.get()
                    .uri(uriBuilder -> {
                        var b = uriBuilder.path("/api/features/{symbol}");
                        if (views != null) b = b.queryParam("views", views);
                        return b.build(sym);
                    })
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body ->
                                    new BusinessException("Features not found", HttpStatus.NOT_FOUND))
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(10));

            if (response == null) throw new BusinessException("Empty response", HttpStatus.BAD_GATEWAY);
            redisTemplate.opsForValue().set(cacheKey, response, Duration.ofMinutes(5));
            return ResponseEntity.ok(ApiResponse.ok(objectMapper.readValue(response, Object.class)));

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("Feature fetch failed [{}]: {}", sym, e.getMessage());
            throw new BusinessException("Failed to fetch features", HttpStatus.BAD_GATEWAY);
        }
    }

    // ── Stats (medium cache — 30 min) ─────────────────────────────────────────

    @GetMapping("/features/{symbol}/stats")
    public ResponseEntity<ApiResponse<Object>> getStats(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "1y") String period) {
        return proxy(
                "/api/features/" + symbol.toUpperCase() + "/stats?period=" + period,
                "stats:" + symbol.toUpperCase(),
                Duration.ofMinutes(30)
        );
    }

    // ── Validation (medium cache — 30 min) ───────────────────────────────────

    @GetMapping("/features/{symbol}/validate")
    public ResponseEntity<ApiResponse<Object>> validate(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "1y") String period) {
        return proxy(
                "/api/features/" + symbol.toUpperCase() + "/validate?period=" + period,
                "validate:" + symbol.toUpperCase(),
                Duration.ofMinutes(30)
        );
    }

    // ── Private helper ────────────────────────────────────────────────────────

    private ResponseEntity<ApiResponse<Object>> proxy(String path, String cacheKeySuffix, Duration ttl) {
        String cacheKey = CACHE_PREFIX + cacheKeySuffix;
        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            try {
                return ResponseEntity.ok(ApiResponse.ok(objectMapper.readValue(cached, Object.class)));
            } catch (Exception ignored) {
                redisTemplate.delete(cacheKey);
            }
        }
        try {
            WebClient client = WebClient.create(pythonBaseUrl);
            String response = client.get()
                    .uri(path)
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body ->
                                    new BusinessException("Feature store error: " + body, HttpStatus.BAD_GATEWAY))
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(15));

            if (response == null) throw new BusinessException("Empty response", HttpStatus.BAD_GATEWAY);
            redisTemplate.opsForValue().set(cacheKey, response, ttl);
            return ResponseEntity.ok(ApiResponse.ok(objectMapper.readValue(response, Object.class)));

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("Feature store proxy failed [{}]: {}", path, e.getMessage());
            throw new BusinessException("Feature store unavailable", HttpStatus.BAD_GATEWAY);
        }
    }
}
