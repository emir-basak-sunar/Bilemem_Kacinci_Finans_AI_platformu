package com.finai.watchlist.service;

import com.finai.common.exception.BusinessException;
import com.finai.user.entity.User;
import com.finai.user.repository.UserRepository;
import com.finai.watchlist.dto.AddWatchlistRequest;
import com.finai.watchlist.dto.WatchlistItemResponse;
import com.finai.watchlist.entity.WatchlistItem;
import com.finai.watchlist.repository.WatchlistRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class WatchlistService {

    private static final int MAX_WATCHLIST_SIZE = 30;

    private final WatchlistRepository watchlistRepository;
    private final UserRepository userRepository;

    public List<WatchlistItemResponse> getWatchlist(UUID userId) {
        return watchlistRepository.findByUserIdOrderByPositionAsc(userId)
                .stream()
                .map(this::toResponse)
                .toList();
    }

    @Transactional
    public WatchlistItemResponse addSymbol(UUID userId, AddWatchlistRequest request) {
        String symbol = request.getSymbol().toUpperCase().trim();

        if (watchlistRepository.existsByUserIdAndSymbol(userId, symbol)) {
            throw new BusinessException("Symbol already in watchlist", HttpStatus.CONFLICT);
        }

        if (watchlistRepository.countByUserId(userId) >= MAX_WATCHLIST_SIZE) {
            throw new BusinessException("Watchlist limit reached (max " + MAX_WATCHLIST_SIZE + ")", HttpStatus.BAD_REQUEST);
        }

        User user = userRepository.getReferenceById(userId);
        int nextPosition = (int) watchlistRepository.countByUserId(userId);

        WatchlistItem item = WatchlistItem.builder()
                .user(user)
                .symbol(symbol)
                .name(request.getName() != null ? request.getName().trim() : symbol)
                .position(nextPosition)
                .build();

        watchlistRepository.save(item);
        log.info("Added {} to watchlist for user {}", symbol, userId);
        return toResponse(item);
    }

    @Transactional
    public void removeSymbol(UUID userId, String symbol) {
        symbol = symbol.toUpperCase().trim();
        if (!watchlistRepository.existsByUserIdAndSymbol(userId, symbol)) {
            throw new BusinessException("Symbol not in watchlist", HttpStatus.NOT_FOUND);
        }
        watchlistRepository.deleteByUserIdAndSymbol(userId, symbol);
        log.info("Removed {} from watchlist for user {}", symbol, userId);
    }

    private WatchlistItemResponse toResponse(WatchlistItem item) {
        return WatchlistItemResponse.builder()
                .id(item.getId())
                .symbol(item.getSymbol())
                .name(item.getName())
                .position(item.getPosition())
                .build();
    }
}
