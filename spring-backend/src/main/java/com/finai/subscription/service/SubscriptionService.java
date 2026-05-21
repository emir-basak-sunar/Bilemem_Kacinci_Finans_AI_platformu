package com.finai.subscription.service;

import com.finai.common.exception.BusinessException;
import com.finai.subscription.entity.SubscriptionPlan;
import com.finai.subscription.entity.UserSubscription;
import com.finai.subscription.repository.SubscriptionPlanRepository;
import com.finai.subscription.repository.UserSubscriptionRepository;
import com.finai.user.entity.User;
import com.finai.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class SubscriptionService {

    private final SubscriptionPlanRepository planRepository;
    private final UserSubscriptionRepository subscriptionRepository;
    private final UserRepository userRepository;

    public List<SubscriptionPlan> getActivePlans() {
        return planRepository.findByActiveTrue();
    }

    public Optional<UserSubscription> getActiveSubscription(UUID userId) {
        return subscriptionRepository.findActiveByUserId(userId);
    }

    @Transactional
    public UserSubscription subscribe(UUID userId, UUID planId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BusinessException("User not found", HttpStatus.NOT_FOUND));

        SubscriptionPlan plan = planRepository.findById(planId)
                .orElseThrow(() -> new BusinessException("Plan not found", HttpStatus.NOT_FOUND));

        // Cancel existing active subscription
        subscriptionRepository.findActiveByUserId(userId).ifPresent(existing -> {
            existing.setStatus("CANCELLED");
            subscriptionRepository.save(existing);
        });

        Instant now = Instant.now();
        UserSubscription subscription = UserSubscription.builder()
                .user(user)
                .plan(plan)
                .status("ACTIVE")
                .startsAt(now)
                .expiresAt(now.plus(30, ChronoUnit.DAYS))
                .build();

        log.info("User {} subscribed to plan {}", userId, plan.getName());
        return subscriptionRepository.save(subscription);
    }

    @Transactional
    public void cancel(UUID userId, UUID subscriptionId) {
        UserSubscription sub = subscriptionRepository.findById(subscriptionId)
                .orElseThrow(() -> new BusinessException("Subscription not found", HttpStatus.NOT_FOUND));

        if (!sub.getUser().getId().equals(userId)) {
            throw new BusinessException("Not your subscription", HttpStatus.FORBIDDEN);
        }

        sub.setStatus("CANCELLED");
        subscriptionRepository.save(sub);
        log.info("Subscription {} cancelled by user {}", subscriptionId, userId);
    }

    public int getMonthlyQuota(UUID userId) {
        return subscriptionRepository.findActiveByUserId(userId)
                .filter(UserSubscription::isActive)
                .map(sub -> sub.getPlan().getAiQuotaMonthly())
                .orElseGet(() ->
                    // No active subscription — fall back to FREE plan quota from DB
                    planRepository.findByName("FREE")
                        .map(SubscriptionPlan::getAiQuotaMonthly)
                        .orElse(10)  // Safe fallback: 10 credits (not unlimited)
                );
    }
}
