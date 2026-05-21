package com.finai.wallet.service;

import com.finai.common.exception.BusinessException;
import com.finai.wallet.dto.WalletResponse;
import com.finai.wallet.entity.Wallet;
import com.finai.wallet.repository.WalletRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Duration;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class WalletService {

    private final WalletRepository walletRepository;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final String WALLET_CACHE_PREFIX = "user:wallet:";

    @Transactional(readOnly = true)
    public WalletResponse getWallet(UUID userId) {
        // Try cache first
        String cacheKey = WALLET_CACHE_PREFIX + userId;
        Object cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached instanceof WalletResponse wr) {
            return wr;
        }

        Wallet wallet = walletRepository.findByUserId(userId)
                .orElseThrow(() -> new BusinessException("Wallet not found", HttpStatus.NOT_FOUND));

        WalletResponse response = toResponse(wallet);
        redisTemplate.opsForValue().set(cacheKey, response, Duration.ofMinutes(15));
        return response;
    }

    @Transactional
    public WalletResponse deposit(UUID userId, BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new BusinessException("Amount must be positive", HttpStatus.BAD_REQUEST);
        }

        Wallet wallet = walletRepository.findByUserIdWithLock(userId)
                .orElseThrow(() -> new BusinessException("Wallet not found", HttpStatus.NOT_FOUND));

        wallet.credit(amount);
        walletRepository.save(wallet);

        invalidateCache(userId);
        log.info("Deposit {} to wallet of user {}", amount, userId);
        return toResponse(wallet);
    }

    @Transactional
    public WalletResponse withdraw(UUID userId, BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new BusinessException("Amount must be positive", HttpStatus.BAD_REQUEST);
        }

        Wallet wallet = walletRepository.findByUserIdWithLock(userId)
                .orElseThrow(() -> new BusinessException("Wallet not found", HttpStatus.NOT_FOUND));

        if (wallet.getBalance().compareTo(amount) < 0) {
            throw new BusinessException("Insufficient balance", HttpStatus.UNPROCESSABLE_ENTITY);
        }

        wallet.debit(amount);
        walletRepository.save(wallet);

        invalidateCache(userId);
        log.info("Withdraw {} from wallet of user {}", amount, userId);
        return toResponse(wallet);
    }

    public void invalidateCache(UUID userId) {
        redisTemplate.delete(WALLET_CACHE_PREFIX + userId);
    }

    private WalletResponse toResponse(Wallet w) {
        return WalletResponse.builder()
                .id(w.getId())
                .balance(w.getBalance())
                .currency(w.getCurrency())
                .build();
    }
}
