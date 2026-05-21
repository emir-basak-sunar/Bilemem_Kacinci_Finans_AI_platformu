package com.finai.auth.service;

import com.finai.auth.dto.*;
import com.finai.auth.entity.RefreshToken;
import com.finai.auth.repository.RefreshTokenRepository;
import com.finai.auth.security.JwtProvider;
import com.finai.common.exception.BusinessException;
import com.finai.user.entity.User;
import com.finai.user.repository.UserRepository;
import com.finai.wallet.entity.Wallet;
import com.finai.wallet.repository.WalletRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.HexFormat;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthService {

    private final UserRepository userRepository;
    private final WalletRepository walletRepository;
    private final RefreshTokenRepository refreshTokenRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtProvider jwtProvider;

    private static final int MAX_FAILED_ATTEMPTS = 5;
    private static final long LOCK_DURATION_MINUTES = 15;

    @Transactional
    public TokenResponse register(RegisterRequest request) {
        if (userRepository.existsByEmail(request.getEmail().toLowerCase())) {
            throw new BusinessException("Email already registered", HttpStatus.CONFLICT);
        }

        User user = User.builder()
                .email(request.getEmail().toLowerCase().trim())
                .passwordHash(passwordEncoder.encode(request.getPassword()))
                .fullName(request.getFullName().trim())
                .role("USER")
                .build();
        userRepository.save(user);

        // Create wallet for new user
        Wallet wallet = Wallet.builder()
                .user(user)
                .build();
        walletRepository.save(wallet);

        log.info("New user registered: {}", user.getEmail());
        return generateTokens(user, null);
    }

    @Transactional
    public TokenResponse login(LoginRequest request, String deviceInfo) {
        User user = userRepository.findByEmail(request.getEmail().toLowerCase())
                .orElseThrow(() -> new BusinessException("Invalid email or password", HttpStatus.UNAUTHORIZED));

        // Check account lock
        if (user.isAccountLocked()) {
            throw new BusinessException("Account is locked. Try again later.", HttpStatus.LOCKED);
        }

        // Verify password
        if (!passwordEncoder.matches(request.getPassword(), user.getPasswordHash())) {
            handleFailedLogin(user);
            throw new BusinessException("Invalid email or password", HttpStatus.UNAUTHORIZED);
        }

        // Reset failed attempts on success
        if (user.getFailedAttempts() > 0) {
            user.setFailedAttempts(0);
            user.setLocked(false);
            user.setLockExpiresAt(null);
            userRepository.save(user);
        }

        log.info("User logged in: {}", user.getEmail());
        return generateTokens(user, deviceInfo);
    }

    @Transactional
    public TokenResponse refresh(RefreshRequest request) {
        String tokenHash = hashToken(request.getRefreshToken());

        RefreshToken storedToken = refreshTokenRepository.findByTokenHash(tokenHash)
                .orElseThrow(() -> new BusinessException("Invalid refresh token", HttpStatus.UNAUTHORIZED));

        if (storedToken.isExpired()) {
            refreshTokenRepository.delete(storedToken);
            throw new BusinessException("Refresh token expired", HttpStatus.UNAUTHORIZED);
        }

        User user = storedToken.getUser();

        // Token rotation: delete old, create new
        refreshTokenRepository.delete(storedToken);

        log.info("Token refreshed for user: {}", user.getEmail());
        return generateTokens(user, storedToken.getDeviceInfo());
    }

    @Transactional
    public void logout(UUID userId) {
        refreshTokenRepository.deleteAllByUserId(userId);
        log.info("User logged out: {}", userId);
    }

    // --- Private helpers ---

    private TokenResponse generateTokens(User user, String deviceInfo) {
        String accessToken = jwtProvider.generateAccessToken(user.getId(), user.getEmail(), user.getRole());
        String refreshToken = jwtProvider.generateRefreshToken();

        // Store refresh token hash in DB
        RefreshToken entity = RefreshToken.builder()
                .user(user)
                .tokenHash(hashToken(refreshToken))
                .deviceInfo(deviceInfo)
                .expiresAt(Instant.now().plusMillis(jwtProvider.getRefreshTokenExpirationMs()))
                .build();
        refreshTokenRepository.save(entity);

        // Build user info for frontend
        double balance = 0.0;
        try {
            balance = walletRepository.findByUserId(user.getId())
                    .map(w -> w.getBalance().doubleValue())
                    .orElse(0.0);
        } catch (Exception e) {
            log.warn("Could not fetch wallet balance for user {}", user.getId());
        }

        TokenResponse.UserInfo userInfo = TokenResponse.UserInfo.builder()
                .id(user.getId())
                .email(user.getEmail())
                .fullName(user.getFullName())
                .avatar(user.getAvatarUrl())
                .role(user.getRole())
                .balance(balance)
                .emailVerified(user.getEmailVerified())
                .build();

        return TokenResponse.of(accessToken, refreshToken, 900, user.getRole(), userInfo);
    }

    private void handleFailedLogin(User user) {
        int newAttempts = user.getFailedAttempts() + 1;
        user.setFailedAttempts(newAttempts);

        if (newAttempts >= MAX_FAILED_ATTEMPTS) {
            user.setLocked(true);
            user.setLockExpiresAt(Instant.now().plusSeconds(LOCK_DURATION_MINUTES * 60));
            log.warn("Account locked due to {} failed attempts: {}", newAttempts, user.getEmail());
        }

        userRepository.save(user);
    }

    private String hashToken(String token) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(token.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }
}
