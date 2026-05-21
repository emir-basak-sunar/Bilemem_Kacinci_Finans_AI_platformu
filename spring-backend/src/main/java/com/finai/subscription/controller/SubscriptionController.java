package com.finai.subscription.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.subscription.entity.SubscriptionPlan;
import com.finai.subscription.entity.UserSubscription;
import com.finai.subscription.service.SubscriptionService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequiredArgsConstructor
public class SubscriptionController {

    private final SubscriptionService subscriptionService;

    @GetMapping("/plans")
    public ResponseEntity<ApiResponse<List<SubscriptionPlan>>> getPlans() {
        return ResponseEntity.ok(ApiResponse.ok(subscriptionService.getActivePlans()));
    }

    @PostMapping("/subscriptions")
    public ResponseEntity<ApiResponse<UserSubscription>> subscribe(
            Authentication auth, @RequestBody Map<String, String> body) {
        UUID userId = (UUID) auth.getPrincipal();
        UUID planId = UUID.fromString(body.get("planId"));
        UserSubscription sub = subscriptionService.subscribe(userId, planId);
        return ResponseEntity.ok(ApiResponse.ok("Subscription activated", sub));
    }

    @GetMapping("/subscriptions/me")
    public ResponseEntity<ApiResponse<UserSubscription>> getMySubscription(Authentication auth) {
        UUID userId = (UUID) auth.getPrincipal();
        return subscriptionService.getActiveSubscription(userId)
                .map(sub -> ResponseEntity.ok(ApiResponse.ok(sub)))
                .orElse(ResponseEntity.ok(ApiResponse.ok("No active subscription", null)));
    }

    @DeleteMapping("/subscriptions/{id}")
    public ResponseEntity<ApiResponse<Void>> cancel(Authentication auth, @PathVariable UUID id) {
        UUID userId = (UUID) auth.getPrincipal();
        subscriptionService.cancel(userId, id);
        return ResponseEntity.ok(ApiResponse.ok("Subscription cancelled", null));
    }
}
