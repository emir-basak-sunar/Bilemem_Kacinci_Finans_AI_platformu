package com.finai.subscription.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;

@Entity
@Table(name = "subscription_plans")
@Getter @Setter
@NoArgsConstructor @AllArgsConstructor
@Builder
public class SubscriptionPlan {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(unique = true, nullable = false, length = 50)
    private String name;

    @Column(name = "display_name", nullable = false, length = 100)
    private String displayName;

    @Column(name = "monthly_price", nullable = false, precision = 10, scale = 2)
    private BigDecimal monthlyPrice;

    @Column(name = "ai_quota_monthly", nullable = false)
    private Integer aiQuotaMonthly;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "jsonb")
    private Map<String, Object> features;

    @Column(nullable = false)
    @Builder.Default
    private Boolean active = true;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;
}
