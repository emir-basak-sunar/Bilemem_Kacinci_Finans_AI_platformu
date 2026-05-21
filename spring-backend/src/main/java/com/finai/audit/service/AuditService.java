package com.finai.audit.service;

import com.finai.audit.entity.AuditLog;
import com.finai.audit.repository.AuditLogRepository;
import com.finai.config.KafkaConfig;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuditService {

    private final AuditLogRepository auditLogRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    /**
     * Fire-and-forget audit log via Kafka.
     */
    public void logAsync(UUID userId, String action, String entityType, String entityId,
                         Map<String, Object> details, String ipAddress, String userAgent) {
        AuditLog entry = AuditLog.builder()
                .userId(userId)
                .action(action)
                .entityType(entityType)
                .entityId(entityId)
                .details(details)
                .ipAddress(ipAddress)
                .userAgent(userAgent)
                .build();

        try {
            kafkaTemplate.send(KafkaConfig.AUDIT_TOPIC, entry);
        } catch (Exception e) {
            // Fallback: save directly if Kafka is down
            log.warn("Kafka unavailable for audit, saving directly", e);
            auditLogRepository.save(entry);
        }
    }

    @KafkaListener(topics = KafkaConfig.AUDIT_TOPIC, groupId = "java-backend")
    public void handleAuditEvent(AuditLog entry) {
        auditLogRepository.save(entry);
        log.debug("Audit log saved: {} - {} by user {}", entry.getAction(), entry.getEntityType(), entry.getUserId());
    }

    // Admin queries
    public Page<AuditLog> getAll(Pageable pageable) {
        return auditLogRepository.findAllByOrderByCreatedAtDesc(pageable);
    }

    public Page<AuditLog> getByUser(UUID userId, Pageable pageable) {
        return auditLogRepository.findByUserIdOrderByCreatedAtDesc(userId, pageable);
    }

    public Page<AuditLog> getByAction(String action, Pageable pageable) {
        return auditLogRepository.findByActionOrderByCreatedAtDesc(action, pageable);
    }
}
