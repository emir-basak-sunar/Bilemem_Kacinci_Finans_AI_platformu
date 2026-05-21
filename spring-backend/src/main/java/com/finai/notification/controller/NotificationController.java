package com.finai.notification.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.notification.entity.Notification;
import com.finai.notification.service.NotificationService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/notifications")
@RequiredArgsConstructor
public class NotificationController {

    private final NotificationService notificationService;

    @GetMapping
    public ResponseEntity<ApiResponse<Page<Notification>>> getNotifications(
            Authentication auth,
            @RequestParam(defaultValue = "false") boolean unreadOnly,
            @PageableDefault(size = 20) Pageable pageable) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok(notificationService.getNotifications(userId, unreadOnly, pageable)));
    }

    @GetMapping("/unread-count")
    public ResponseEntity<ApiResponse<Map<String, Long>>> getUnreadCount(Authentication auth) {
        UUID userId = (UUID) auth.getPrincipal();
        long count = notificationService.getUnreadCount(userId);
        return ResponseEntity.ok(ApiResponse.ok(Map.of("count", count)));
    }

    @PatchMapping("/{id}/read")
    public ResponseEntity<ApiResponse<Void>> markAsRead(Authentication auth, @PathVariable UUID id) {
        UUID userId = (UUID) auth.getPrincipal();
        notificationService.markAsRead(userId, id);
        return ResponseEntity.ok(ApiResponse.ok("Marked as read", null));
    }
}
