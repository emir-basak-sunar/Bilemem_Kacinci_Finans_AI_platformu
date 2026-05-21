package com.finai.ai.controller;

import com.finai.ai.dto.PredictionRequest;
import com.finai.ai.service.AiGatewayService;
import com.finai.common.dto.ApiResponse;
import com.finai.subscription.service.SubscriptionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/ai")
@RequiredArgsConstructor
@Slf4j
public class AiController {

    private final AiGatewayService aiGatewayService;
    private final SubscriptionService subscriptionService;

    /**
     * POST /api/v1/ai/predict
     * Run AI prediction using pre-trained models.
     * Quota is pulled from the user's active subscription plan.
     *   - FREE: 10 predictions/month
     *   - PRO: depends on plan config
     *   - ENTERPRISE: depends on plan config
     */
    @PostMapping("/predict")
    public ResponseEntity<ApiResponse<Map<String, Object>>> predict(
            Authentication auth,
            @Valid @RequestBody PredictionRequest request) {

        UUID userId = (UUID) auth.getPrincipal();
        int quota = subscriptionService.getMonthlyQuota(userId);

        log.info("Prediction request: user={}, symbol={}, model={}, horizon={}, quota={}",
                userId, request.getSymbol(), request.getModelType(), request.getHorizon(), quota);

        Map<String, Object> result = aiGatewayService.predict(userId, request, quota);
        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    /**
     * GET /api/v1/ai/quota
     * Check current month's AI usage and remaining credits.
     * Quota limit comes from the user's subscription plan.
     */
    @GetMapping("/quota")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getQuota(Authentication auth) {
        UUID userId = (UUID) auth.getPrincipal();
        int quota = subscriptionService.getMonthlyQuota(userId);
        Map<String, Object> info = aiGatewayService.getQuotaInfo(userId, quota);
        return ResponseEntity.ok(ApiResponse.ok(info));
    }

    /**
     * GET /api/v1/ai/models
     * List available pre-trained models. No quota consumed.
     */
    @GetMapping("/models")
    public ResponseEntity<ApiResponse<Map<String, Object>>> listModels() {
        Map<String, Object> models = aiGatewayService.listModels();
        return ResponseEntity.ok(ApiResponse.ok(models));
    }
}
