package com.finai.watchlist.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.watchlist.dto.AddWatchlistRequest;
import com.finai.watchlist.dto.WatchlistItemResponse;
import com.finai.watchlist.service.WatchlistService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/watchlist")
@RequiredArgsConstructor
public class WatchlistController {

    private final WatchlistService watchlistService;

    /** GET /watchlist — user's full watchlist */
    @GetMapping
    public ResponseEntity<ApiResponse<List<WatchlistItemResponse>>> getWatchlist(Authentication auth) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok(watchlistService.getWatchlist(userId)));
    }

    /** POST /watchlist — add a symbol */
    @PostMapping
    public ResponseEntity<ApiResponse<WatchlistItemResponse>> addSymbol(
            Authentication auth,
            @Valid @RequestBody AddWatchlistRequest request) {
        UUID userId = (UUID) auth.getPrincipal();
        WatchlistItemResponse item = watchlistService.addSymbol(userId, request);
        return ResponseEntity.status(HttpStatus.CREATED).body(ApiResponse.ok("Added to watchlist", item));
    }

    /** DELETE /watchlist/{symbol} — remove a symbol */
    @DeleteMapping("/{symbol}")
    public ResponseEntity<ApiResponse<Void>> removeSymbol(
            Authentication auth,
            @PathVariable String symbol) {
        UUID userId = (UUID) auth.getPrincipal();
        watchlistService.removeSymbol(userId, symbol);
        return ResponseEntity.ok(ApiResponse.ok("Removed from watchlist", null));
    }
}
