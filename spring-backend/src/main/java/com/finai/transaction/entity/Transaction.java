package com.finai.transaction.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "transactions")
@Getter @Setter
@NoArgsConstructor @AllArgsConstructor
@Builder
public class Transaction {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "idempotency_key", unique = true, nullable = false, length = 64)
    private String idempotencyKey;

    @Column(name = "sender_wallet_id")
    private UUID senderWalletId;

    @Column(name = "receiver_wallet_id")
    private UUID receiverWalletId;

    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal amount;

    @Column(nullable = false, length = 20)
    private String type; // DEPOSIT, WITHDRAW, TRANSFER, AI_CHARGE

    @Column(nullable = false, length = 20)
    @Builder.Default
    private String status = "PENDING"; // PENDING, COMPLETED, FAILED

    @Column
    private String description;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;
}
