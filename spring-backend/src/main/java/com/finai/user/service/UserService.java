package com.finai.user.service;

import com.finai.common.exception.BusinessException;
import com.finai.user.dto.ChangePasswordRequest;
import com.finai.user.dto.UpdateProfileRequest;
import com.finai.user.dto.UserProfileResponse;
import com.finai.user.entity.User;
import com.finai.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserService {

    private final UserRepository userRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final PasswordEncoder passwordEncoder;

    private static final String PROFILE_CACHE_PREFIX = "user:profile:";

    public UserProfileResponse getProfile(UUID userId) {
        String cacheKey = PROFILE_CACHE_PREFIX + userId;
        // Skip Redis cache for profile to avoid deserialization issues
        // Profile is lightweight enough to fetch from DB every time
        
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BusinessException("User not found", HttpStatus.NOT_FOUND));

        return toResponse(user);
    }

    @Transactional
    public UserProfileResponse updateProfile(UUID userId, UpdateProfileRequest request) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BusinessException("User not found", HttpStatus.NOT_FOUND));

        if (request.getFullName() != null) {
            user.setFullName(request.getFullName().trim());
        }

        if (request.getAvatarUrl() != null) {
            // Validate base64 size (max ~2MB encoded)
            if (request.getAvatarUrl().length() > 3_000_000) {
                throw new BusinessException("Avatar image is too large (max 2MB)", HttpStatus.BAD_REQUEST);
            }
            user.setAvatarUrl(request.getAvatarUrl());
        }

        userRepository.save(user);

        log.info("Profile updated for user {}", userId);
        return toResponse(user);
    }

    @Transactional
    public void changePassword(UUID userId, ChangePasswordRequest request) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BusinessException("User not found", HttpStatus.NOT_FOUND));

        if (!passwordEncoder.matches(request.getCurrentPassword(), user.getPasswordHash())) {
            throw new BusinessException("Current password is incorrect", HttpStatus.BAD_REQUEST);
        }

        if (request.getCurrentPassword().equals(request.getNewPassword())) {
            throw new BusinessException("New password must be different from current password", HttpStatus.BAD_REQUEST);
        }

        user.setPasswordHash(passwordEncoder.encode(request.getNewPassword()));
        userRepository.save(user);
        log.info("Password changed for user {}", userId);
    }

    private UserProfileResponse toResponse(User user) {
        return UserProfileResponse.builder()
                .id(user.getId())
                .email(user.getEmail())
                .fullName(user.getFullName())
                .avatarUrl(user.getAvatarUrl())
                .role(user.getRole())
                .emailVerified(user.getEmailVerified())
                .createdAt(user.getCreatedAt())
                .build();
    }
}
