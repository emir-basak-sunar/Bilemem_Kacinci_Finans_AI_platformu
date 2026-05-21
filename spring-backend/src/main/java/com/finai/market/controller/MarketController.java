package com.finai.market.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.market.service.MarketDataService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/market")
@RequiredArgsConstructor
public class MarketController {

    private final MarketDataService marketDataService;

    @GetMapping("/{symbol}")
    public ResponseEntity<ApiResponse<Object>> getMarketData(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "1mo") String period) {
        Object data = marketDataService.getMarketData(symbol, period);
        return ResponseEntity.ok(ApiResponse.ok(data));
    }
}
