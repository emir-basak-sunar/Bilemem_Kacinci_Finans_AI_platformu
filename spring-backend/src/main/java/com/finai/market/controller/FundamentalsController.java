package com.finai.market.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.market.service.FundamentalsService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * Fundamentals Controller
 *
 * All endpoints are public (no auth required) — same as /market/*.
 * Fundamental data is not user-specific and doesn't consume quota.
 *
 * Endpoints:
 *   GET /fundamentals/{symbol}            — full composite
 *   GET /fundamentals/{symbol}/overview   — company info + valuation
 *   GET /fundamentals/{symbol}/financials — income/balance/cashflow
 *   GET /fundamentals/{symbol}/analyst    — recommendations + price targets
 *   GET /fundamentals/{symbol}/holders    — institutional + major holders
 *   GET /fundamentals/{symbol}/insider    — insider transactions
 *   GET /fundamentals/{symbol}/earnings   — EPS history + next date
 *   GET /fundamentals/{symbol}/options    — options chain summary
 */
@RestController
@RequestMapping("/fundamentals")
@RequiredArgsConstructor
public class FundamentalsController {

    private final FundamentalsService fundamentalsService;

    @GetMapping("/{symbol}")
    public ResponseEntity<ApiResponse<Object>> getFull(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getFull(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/overview")
    public ResponseEntity<ApiResponse<Object>> getOverview(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getOverview(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/financials")
    public ResponseEntity<ApiResponse<Object>> getFinancials(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getFinancials(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/analyst")
    public ResponseEntity<ApiResponse<Object>> getAnalyst(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getAnalyst(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/holders")
    public ResponseEntity<ApiResponse<Object>> getHolders(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getHolders(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/insider")
    public ResponseEntity<ApiResponse<Object>> getInsider(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getInsider(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/earnings")
    public ResponseEntity<ApiResponse<Object>> getEarnings(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getEarnings(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/options")
    public ResponseEntity<ApiResponse<Object>> getOptions(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getOptions(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/news")
    public ResponseEntity<ApiResponse<Object>> getNews(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getNews(symbol.toUpperCase())));
    }

    @GetMapping("/{symbol}/esg")
    public ResponseEntity<ApiResponse<Object>> getEsg(@PathVariable String symbol) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getEsg(symbol.toUpperCase())));
    }

    /**
     * GET /fundamentals/news/content?url=...
     * Fetch and parse article content from a news URL.
     * Proxied through Python backend — no auth required.
     */
    @GetMapping("/news/content")
    public ResponseEntity<ApiResponse<Object>> getNewsContent(@RequestParam String url) {
        return ResponseEntity.ok(ApiResponse.ok(fundamentalsService.getNewsContent(url)));
    }
}
