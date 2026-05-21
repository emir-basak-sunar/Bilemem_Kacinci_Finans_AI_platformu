package com.finai.transaction.dto;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.math.BigDecimal;
import java.util.UUID;

@Data
public class TransferRequest {
    @NotBlank(message = "Idempotency key is required")
    private String idempotencyKey;

    @NotNull(message = "Receiver user ID is required")
    private UUID receiverUserId;

    @NotNull(message = "Amount is required")
    @DecimalMin(value = "0.01", message = "Amount must be at least 0.01")
    private BigDecimal amount;

    private String description;
}
