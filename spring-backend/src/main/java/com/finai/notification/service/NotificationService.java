package com.finai.notification.service;

import com.finai.common.exception.BusinessException;
import com.finai.config.KafkaConfig;
import com.finai.notification.dto.NotificationEvent;
import com.finai.notification.entity.Notification;
import com.finai.notification.repository.NotificationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {

    private final NotificationRepository notificationRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    /**
     * Publish notification event to Kafka (async, non-blocking).
     */
    public void sendAsync(NotificationEvent event) {
        kafkaTemplate.send(KafkaConfig.NOTIFICATION_TOPIC, event);
        log.debug("Notification event published for user {}: {}", event.getUserId(), event.getTitle());
    }

    /**
     * Kafka listener — persists notifications.
     */
    @KafkaListener(topics = KafkaConfig.NOTIFICATION_TOPIC, groupId = "java-backend")
    @Transactional
    public void handleNotification(NotificationEvent event) {
        Notification notification = Notification.builder()
                .userId(event.getUserId())
                .type(event.getType())
                .title(event.getTitle())
                .body(event.getBody())
                .build();
        notificationRepository.save(notification);
        log.info("Notification saved for user {}: {}", event.getUserId(), event.getTitle());
    }

    public Page<Notification> getNotifications(UUID userId, boolean unreadOnly, Pageable pageable) {
        if (unreadOnly) {
            return notificationRepository.findByUserIdAndReadFalseOrderByCreatedAtDesc(userId, pageable);
        }
        return notificationRepository.findByUserIdOrderByCreatedAtDesc(userId, pageable);
    }

    public long getUnreadCount(UUID userId) {
        return notificationRepository.countByUserIdAndReadFalse(userId);
    }

    @Transactional
    public void markAsRead(UUID userId, UUID notificationId) {
        Notification n = notificationRepository.findById(notificationId)
                .orElseThrow(() -> new BusinessException("Notification not found", HttpStatus.NOT_FOUND));
        if (!n.getUserId().equals(userId)) {
            throw new BusinessException("Not your notification", HttpStatus.FORBIDDEN);
        }
        n.setRead(true);
        notificationRepository.save(n);
    }
}
