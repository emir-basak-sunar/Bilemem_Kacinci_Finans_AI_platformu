package com.finai.ai.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import lombok.Data;

import java.util.Set;

@Data
public class PredictionRequest {

    private static final Set<String> VALID_MODELS = Set.of(
            "ensemble", "xgboost", "lightgbm", "catboost",
            "lstm", "tcn", "tft", "arima", "sarima", "sarimax"
    );

    private static final Set<String> VALID_PERIODS = Set.of(
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"
    );

    @NotBlank(message = "Symbol is required")
    @Pattern(regexp = "^[A-Za-z0-9\\-\\.]{1,10}$", message = "Invalid symbol format")
    private String symbol;

    // Accept both snake_case (frontend) and camelCase
    @JsonProperty("model_type")
    @Pattern(
        regexp = "^(ensemble|xgboost|lightgbm|catboost|lstm|tcn|tft|arima|sarima|sarimax)$",
        message = "Invalid model type. Valid: ensemble, xgboost, lightgbm, catboost, lstm, tcn, tft, arima, sarima, sarimax"
    )
    private String modelType = "ensemble";

    @Pattern(regexp = "^(1d|5d|1mo|3mo|6mo|1y|2y|5y)$",
             message = "Invalid period")
    private String period = "1mo";

    @Min(value = 1, message = "Horizon must be at least 1")
    @Max(value = 60, message = "Horizon cannot exceed 60")
    private int horizon = 5;

    /** Normalize inputs after deserialization */
    public String getSymbol() {
        return symbol != null ? symbol.toUpperCase().trim() : null;
    }

    public String getModelType() {
        return modelType != null ? modelType.toLowerCase().trim() : "ensemble";
    }
}
