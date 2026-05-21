package com.finai.ai.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Structured response from the AI prediction service.
 * Wraps the raw Python response into a typed Java object.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class PredictionResponse {

    /** Current market price */
    private Double currentPrice;

    /** Primary prediction value (ensemble or selected model) */
    private Double prediction;

    /** Individual model predictions */
    private Map<String, Double> models;

    /** Future price path for chart rendering */
    private List<FuturePoint> futurePath;

    /** Confidence intervals from TFT quantile outputs */
    private ConfidenceIntervals confidenceIntervals;

    /** Model version used */
    private Integer modelVersion;

    /** Cache status: "hit" or "miss" */
    private String cache;

    /** Response time in milliseconds */
    private Double responseTimeMs;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FuturePoint {
        private Long time;
        private Double value;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ConfidenceIntervals {
        private Double p10;
        private Double p25;
        private Double p50;
        private Double p75;
        private Double p90;
    }
}
