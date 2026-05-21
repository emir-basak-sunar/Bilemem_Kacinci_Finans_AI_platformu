package com.finai.subscription.repository;

import com.finai.subscription.entity.UserSubscription;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface UserSubscriptionRepository extends JpaRepository<UserSubscription, UUID> {
    @Query("SELECT us FROM UserSubscription us WHERE us.user.id = :userId AND us.status = 'ACTIVE' ORDER BY us.expiresAt DESC")
    Optional<UserSubscription> findActiveByUserId(UUID userId);
}
