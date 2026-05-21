package com.finai.ai.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "ai_usage")
@Getter @Setter
@NoArgsConstructor @AllArgsConstructor
@Builder
public class AiUsage {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private UUID userId;

    @Column(name = "model_type", nullable = false, length = 20)
    private String modelType;

    @Column(nullable = false, length = 10)
    private String symbol;

    @Column(nullable = false)
    @Builder.Default
    private Integer credits = 1;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;
}
