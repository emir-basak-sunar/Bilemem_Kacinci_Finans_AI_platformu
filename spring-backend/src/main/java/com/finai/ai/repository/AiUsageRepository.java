package com.finai.ai.repository;

import com.finai.ai.entity.AiUsage;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.util.UUID;

@Repository
public interface AiUsageRepository extends JpaRepository<AiUsage, UUID> {
    @Query("SELECT COALESCE(SUM(a.credits), 0) FROM AiUsage a WHERE a.userId = :userId AND a.createdAt >= :since")
    long sumCreditsByUserIdSince(UUID userId, Instant since);
}
