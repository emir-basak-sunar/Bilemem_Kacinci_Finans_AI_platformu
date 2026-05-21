package com.finai.watchlist.repository;

import com.finai.watchlist.entity.WatchlistItem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface WatchlistRepository extends JpaRepository<WatchlistItem, UUID> {

    List<WatchlistItem> findByUserIdOrderByPositionAsc(UUID userId);

    Optional<WatchlistItem> findByUserIdAndSymbol(UUID userId, String symbol);

    boolean existsByUserIdAndSymbol(UUID userId, String symbol);

    @Modifying
    @Query("DELETE FROM WatchlistItem w WHERE w.user.id = :userId AND w.symbol = :symbol")
    void deleteByUserIdAndSymbol(UUID userId, String symbol);

    long countByUserId(UUID userId);
}
