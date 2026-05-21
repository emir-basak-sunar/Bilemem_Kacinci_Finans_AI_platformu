package com.finai.market.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.common.exception.BusinessException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
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
 * MLflow Controller
 *
 * Proxies MLflow model registry requests to the Python FastAPI backend.
 * Public endpoints — no auth required (read-only model metadata).
 *
 * Endpoints:
 *   POST /mlflow/import              — import trained_models/ to MLflow
 *   GET  /mlflow/summary             — all production runs summary
 *   GET  /mlflow/run/{runName}       — single run detail + SHAP
 *   GET  /mlflow/comparison          — IC/Sharpe ranking
 *   GET  /mlflow/models              — raw registry listing
 */
@RestController
@RequestMapping("/mlflow")
@RequiredArgsConstructor
@Slf4j
public class MlflowController {

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;

    @Value("${ai-service.base-url:http://localhost:8000}")
    private String pythonBaseUrl;

    private WebClient webClient;

    @PostConstruct
    public void init() {
        this.webClient = WebClient.builder()
                .baseUrl(pythonBaseUrl)
                .codecs(c -> c.defaultCodecs().maxInMemorySize(10 * 1024 * 1024))
                .build();
    }

    private static final String CACHE_PREFIX = "spring:mlflow:";

    // ── Import (no cache — always fresh) ─────────────────────────────────────

    @PostMapping("/import")
    public ResponseEntity<ApiResponse<Object>> importModels() {
        try {
            String response = webClient.post()
                    .uri("/api/mlflow/import")
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body ->
                                    new BusinessException("MLflow import failed: " + body, HttpStatus.BAD_GATEWAY))
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(120));  // import can take time

            if (response == null) throw new BusinessException("Empty response", HttpStatus.BAD_GATEWAY);

            // Invalidate summary cache after import
            redisTemplate.delete(CACHE_PREFIX + "summary");
            redisTemplate.delete(CACHE_PREFIX + "comparison");

            Object data = objectMapper.readValue(response, Object.class);
            log.info("MLflow import completed");
            return ResponseEntity.ok(ApiResponse.ok(data));

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("MLflow import failed: {}", e.getMessage());
            throw new BusinessException("MLflow import failed", HttpStatus.BAD_GATEWAY);
        }
    }

    // ── Summary (cache 5 min) ─────────────────────────────────────────────────

    @GetMapping("/summary")
    public ResponseEntity<ApiResponse<Object>> getSummary() {
        return proxy("/api/mlflow/summary", "summary", Duration.ofMinutes(5));
    }

    // ── Run detail (cache 10 min) ─────────────────────────────────────────────

    @GetMapping("/run/{runName}")
    public ResponseEntity<ApiResponse<Object>> getRunDetail(@PathVariable String runName) {
        return proxy("/api/mlflow/run/" + runName, "run:" + runName, Duration.ofMinutes(10));
    }

    // ── Comparison (cache 5 min) ──────────────────────────────────────────────

    @GetMapping("/comparison")
    public ResponseEntity<ApiResponse<Object>> getComparison() {
        return proxy("/api/mlflow/comparison", "comparison", Duration.ofMinutes(5));
    }

    // ── Models list (cache 2 min) ─────────────────────────────────────────────

    @GetMapping("/models")
    public ResponseEntity<ApiResponse<Object>> getModels() {
        return proxy("/api/mlflow/models", "models", Duration.ofMinutes(2));
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
            String response = webClient.get()
                    .uri(path)
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body ->
                                    new BusinessException("MLflow error: " + body, HttpStatus.BAD_GATEWAY))
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(30));

            if (response == null) throw new BusinessException("Empty response", HttpStatus.BAD_GATEWAY);
            redisTemplate.opsForValue().set(cacheKey, response, ttl);
            return ResponseEntity.ok(ApiResponse.ok(objectMapper.readValue(response, Object.class)));

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("MLflow proxy failed [{}]: {}", path, e.getMessage());
            throw new BusinessException("MLflow service unavailable", HttpStatus.BAD_GATEWAY);
        }
    }
}
