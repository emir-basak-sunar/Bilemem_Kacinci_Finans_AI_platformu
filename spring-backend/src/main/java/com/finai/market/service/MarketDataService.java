package com.finai.market.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import com.finai.common.exception.BusinessException;

import java.time.Duration;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.ConcurrentHashMap;

@Service
@RequiredArgsConstructor
@Slf4j
public class MarketDataService {

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    private final WebClient webClient = WebClient.create("http://localhost:8000");

    private static final String CACHE_PREFIX = "market:data:";
    private static final ConcurrentHashMap<String, ReentrantLock> lockMap = new ConcurrentHashMap<>();

    public Object getMarketData(String symbol, String period) {
        String cacheKey = CACHE_PREFIX + symbol.toUpperCase() + ":" + period;

        // Try Cache
        String cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            log.debug("Cache HIT for {}", cacheKey);
            try {
                return objectMapper.readValue(cached, new TypeReference<List<Map<String, Object>>>() {});
            } catch (Exception e) {
                log.warn("Failed to deserialize cache", e);
            }
        }

        ReentrantLock lock = lockMap.computeIfAbsent(cacheKey, k -> new ReentrantLock());

        if (lock.tryLock()) {
            try {
                cached = redisTemplate.opsForValue().get(cacheKey);
                if (cached != null) {
                    return objectMapper.readValue(cached, new TypeReference<List<Map<String, Object>>>() {});
                }

                log.info("Fetching market data via Python backend for {}/{}", symbol, period);
                
                // Fetch from Python Service (Handles TA-Lib, formatting, Yahoo API correctly)
                String responseBody = webClient.get()
                        .uri(uriBuilder -> uriBuilder
                                .path("/api/market-data/{symbol}")
                                .queryParam("period", period)
                                .build(symbol))
                        .retrieve()
                        .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
                                response -> response.bodyToMono(String.class).map(body -> {
                                    log.error("Python market API error: {}", body);
                                    return new BusinessException("Data unavailable: " + response.statusCode(), HttpStatus.BAD_GATEWAY);
                                }))
                        .bodyToMono(String.class)
                        .block(Duration.ofSeconds(15));
                
                Object data = objectMapper.readValue(responseBody, new TypeReference<List<Map<String, Object>>>() {});

                Duration ttl = calculateTtl(period);
                redisTemplate.opsForValue().set(cacheKey, responseBody, ttl);

                return data;
            } catch (BusinessException be) {
                throw be;
            } catch (Exception e) {
                log.error("Error fetching market data for {}", symbol, e);
                throw new BusinessException("Failed to fetch market data", HttpStatus.BAD_GATEWAY);
            } finally {
                lock.unlock();
            }
        } else {
            try { Thread.sleep(200); } catch (Exception ignored) {}
            cached = redisTemplate.opsForValue().get(cacheKey);
            if (cached != null) {
                try {
                    return objectMapper.readValue(cached, new TypeReference<List<Map<String, Object>>>() {});
                } catch (Exception ignored) {}
            }
            throw new BusinessException("Data temporarily unavailable, please retry", HttpStatus.SERVICE_UNAVAILABLE);
        }
    }

    private Duration calculateTtl(String period) {
        boolean marketOpen = isMarketOpen();
        return switch (period) {
            case "1d" -> marketOpen ? Duration.ofSeconds(30) : Duration.ofHours(6);
            case "5d", "1mo" -> marketOpen ? Duration.ofSeconds(60) : Duration.ofHours(6);
            case "3mo", "6mo" -> Duration.ofMinutes(30);
            case "1y", "2y", "5y" -> Duration.ofHours(6);
            default -> Duration.ofMinutes(5);
        };
    }

    private boolean isMarketOpen() {
        LocalTime now = LocalTime.now(ZoneId.of("America/New_York"));
        return now.isAfter(LocalTime.of(9, 30)) && now.isBefore(LocalTime.of(16, 0));
    }
}
