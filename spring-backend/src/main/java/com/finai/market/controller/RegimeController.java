package com.finai.market.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.common.exception.BusinessException;
import com.fasterxml.jackson.core.type.TypeReference;
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
 * Regime Detection Controller
 *
 * Proxies regime detection requests to the Python backend.
 * Public endpoint — no auth required (same as /market/*).
 *
 * GET /regime/{symbol}?period=3mo
 *
 * Caching: 30 min (matches Python-side TTL)
 */
@RestController
@RequestMapping("/regime")
@RequiredArgsConstructor
@Slf4j
public class RegimeController {

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;

    @Value("${ai-service.base-url:http://localhost:8000}")
    private String pythonBaseUrl;

    private WebClient webClient;

    @jakarta.annotation.PostConstruct
    public void init() {
        this.webClient = WebClient.builder()
                .baseUrl(pythonBaseUrl)
                .codecs(c -> c.defaultCodecs().maxInMemorySize(10 * 1024 * 1024))
                .build();
    }

    private static final String CACHE_PREFIX = "spring:regime:";

    @GetMapping("/{symbol}")
    public ResponseEntity<ApiResponse<Object>> getRegime(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "3mo") String period) {

        String sym = symbol.toUpperCase();
        String cacheKey = CACHE_PREFIX + sym + ":" + period;

        // L2 cache check
        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            try {
                Object data = objectMapper.readValue(cached, Object.class);
                return ResponseEntity.ok(ApiResponse.ok(data));
            } catch (Exception e) {
                redisTemplate.delete(cacheKey);
            }
        }

        try {
            String response = webClient.get()
                    .uri(uriBuilder -> uriBuilder
                            .path("/api/regime/{symbol}")
                            .queryParam("period", period)
                            .build(sym))
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body -> {
                                log.warn("Python regime error [{}]: {}", sym, body);
                                return new BusinessException("Regime detection unavailable", HttpStatus.BAD_GATEWAY);
                            })
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(30));

            if (response == null) {
                throw new BusinessException("Empty response from regime service", HttpStatus.BAD_GATEWAY);
            }

            redisTemplate.opsForValue().set(cacheKey, response, Duration.ofMinutes(30));
            log.info("Regime cached [{}/{}]", sym, period);

            Object data = objectMapper.readValue(response, Object.class);
            return ResponseEntity.ok(ApiResponse.ok(data));

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("Regime detection failed [{}/{}]: {}", sym, period, e.getMessage());
            throw new BusinessException("Failed to detect market regime", HttpStatus.BAD_GATEWAY);
        }
    }
}
