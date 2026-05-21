package com.finai.market.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.finai.common.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.Map;

/**
 * Fundamentals Service — proxies fundamental data requests to the Python backend.
 *
 * Caching strategy (Spring-side L2 cache on top of Python's L1 Redis cache):
 * - overview:    6h
 * - financials:  24h
 * - analyst:     1h
 * - holders:     6h
 * - insider:     6h
 * - earnings:    6h
 * - options:     15min
 * - full:        30min (composite — shortest TTL of its parts)
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class FundamentalsService {

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;

    @Value("${ai-service.base-url:http://localhost:8000}")
    private String pythonBaseUrl;

    private static final String CACHE_PREFIX = "spring:fundamentals:";

    public Object getFull(String symbol) {
        return fetchWithCache(symbol, "full", Duration.ofMinutes(30));
    }

    public Object getOverview(String symbol) {
        return fetchWithCache(symbol, "overview", Duration.ofHours(6));
    }

    public Object getFinancials(String symbol) {
        return fetchWithCache(symbol, "financials", Duration.ofHours(24));
    }

    public Object getAnalyst(String symbol) {
        return fetchWithCache(symbol, "analyst", Duration.ofHours(1));
    }

    public Object getHolders(String symbol) {
        return fetchWithCache(symbol, "holders", Duration.ofHours(6));
    }

    public Object getInsider(String symbol) {
        return fetchWithCache(symbol, "insider", Duration.ofHours(6));
    }

    public Object getEarnings(String symbol) {
        return fetchWithCache(symbol, "earnings", Duration.ofHours(6));
    }

    public Object getOptions(String symbol) {
        return fetchWithCache(symbol, "options", Duration.ofMinutes(15));
    }

    public Object getNews(String symbol) {
        return fetchWithCache(symbol, "news", Duration.ofMinutes(15));
    }

    public Object getEsg(String symbol) {
        return fetchWithCache(symbol, "esg", Duration.ofHours(12));
    }

    public Object getNewsContent(String articleUrl) {
        // Cache key based on URL
        String cacheKey = CACHE_PREFIX + "news:content:" + articleUrl.hashCode();
        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            try { return objectMapper.readValue(cached, Object.class); } catch (Exception ignored) {}
        }
        try {
            WebClient client = WebClient.create(pythonBaseUrl);
            String response = client.get()
                    .uri(uriBuilder -> uriBuilder
                            .path("/api/news/fetch")
                            .queryParam("url", articleUrl)
                            .build())
                    .retrieve()
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(15));
            if (response == null) throw new BusinessException("Empty response", HttpStatus.BAD_GATEWAY);
            redisTemplate.opsForValue().set(cacheKey, response, Duration.ofHours(1));
            return objectMapper.readValue(response, Object.class);
        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("News content fetch failed: {}", e.getMessage());
            throw new BusinessException("Failed to fetch article content", HttpStatus.BAD_GATEWAY);
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private Object fetchWithCache(String symbol, String section, Duration ttl) {
        String cacheKey = CACHE_PREFIX + symbol.toUpperCase() + ":" + section;

        // L2 cache check (Spring Redis)
        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            log.debug("Fundamentals cache HIT: {}", cacheKey);
            try {
                return objectMapper.readValue(cached, Object.class);
            } catch (Exception e) {
                log.warn("Failed to deserialize fundamentals cache: {}", e.getMessage());
                redisTemplate.delete(cacheKey);
            }
        }

        // Fetch from Python backend
        String path = section.equals("full")
                ? "/api/fundamentals/" + symbol.toUpperCase()
                : "/api/fundamentals/" + symbol.toUpperCase() + "/" + section;

        try {
            WebClient client = WebClient.create(pythonBaseUrl);
            String response = client.get()
                    .uri(path)
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            resp -> resp.bodyToMono(String.class).map(body -> {
                                log.warn("Python fundamentals error [{}]: {}", path, body);
                                return new BusinessException("Fundamentals data unavailable", HttpStatus.BAD_GATEWAY);
                            })
                    )
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(30));

            if (response == null) {
                throw new BusinessException("Empty response from fundamentals service", HttpStatus.BAD_GATEWAY);
            }

            // Cache the raw JSON string
            redisTemplate.opsForValue().set(cacheKey, response, ttl);
            log.info("Fundamentals cached [{}/{}] TTL={}s", symbol, section, ttl.getSeconds());

            return objectMapper.readValue(response, Object.class);

        } catch (BusinessException be) {
            throw be;
        } catch (Exception e) {
            log.error("Fundamentals fetch failed [{}/{}]: {}", symbol, section, e.getMessage());
            throw new BusinessException("Failed to fetch fundamentals data", HttpStatus.BAD_GATEWAY);
        }
    }
}
