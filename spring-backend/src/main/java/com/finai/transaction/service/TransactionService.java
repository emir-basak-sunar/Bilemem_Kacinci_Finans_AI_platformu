package com.finai.transaction.service;

import com.finai.common.exception.BusinessException;
import com.finai.transaction.dto.TransferRequest;
import com.finai.transaction.entity.Transaction;
import com.finai.transaction.repository.TransactionRepository;
import com.finai.wallet.entity.Wallet;
import com.finai.wallet.repository.WalletRepository;
import com.finai.wallet.service.WalletService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.Optional;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class TransactionService {

    private final TransactionRepository transactionRepository;
    private final WalletRepository walletRepository;
    private final WalletService walletService;

    @Transactional
    public Transaction transfer(UUID senderUserId, TransferRequest request) {
        // 1. Idempotency check
        Optional<Transaction> existing = transactionRepository.findByIdempotencyKey(request.getIdempotencyKey());
        if (existing.isPresent()) {
            log.info("Idempotent request detected: {}", request.getIdempotencyKey());
            return existing.get();
        }

        // 2. Cannot transfer to self
        if (senderUserId.equals(request.getReceiverUserId())) {
            throw new BusinessException("Cannot transfer to yourself", HttpStatus.BAD_REQUEST);
        }

        // 3. Lock wallets (always lock in consistent order to prevent deadlocks)
        Wallet senderWallet = walletRepository.findByUserIdWithLock(senderUserId)
                .orElseThrow(() -> new BusinessException("Sender wallet not found", HttpStatus.NOT_FOUND));
        Wallet receiverWallet = walletRepository.findByUserIdWithLock(request.getReceiverUserId())
                .orElseThrow(() -> new BusinessException("Receiver wallet not found", HttpStatus.NOT_FOUND));

        // 4. Check balance
        if (senderWallet.getBalance().compareTo(request.getAmount()) < 0) {
            Transaction failedTxn = createTransaction(request, senderWallet.getId(), receiverWallet.getId(), "FAILED");
            transactionRepository.save(failedTxn);
            throw new BusinessException("Insufficient balance", HttpStatus.UNPROCESSABLE_ENTITY);
        }

        // 5. Execute transfer
        senderWallet.debit(request.getAmount());
        receiverWallet.credit(request.getAmount());
        walletRepository.save(senderWallet);
        walletRepository.save(receiverWallet);

        // 6. Record transaction
        Transaction txn = createTransaction(request, senderWallet.getId(), receiverWallet.getId(), "COMPLETED");
        transactionRepository.save(txn);

        // 7. Invalidate caches
        walletService.invalidateCache(senderUserId);
        walletService.invalidateCache(request.getReceiverUserId());

        log.info("Transfer completed: {} -> {} amount={}", senderUserId, request.getReceiverUserId(), request.getAmount());
        return txn;
    }

    @Transactional
    public Transaction recordDeposit(UUID userId, BigDecimal amount, String idempotencyKey) {
        Wallet wallet = walletRepository.findByUserId(userId)
                .orElseThrow(() -> new BusinessException("Wallet not found", HttpStatus.NOT_FOUND));

        Transaction txn = Transaction.builder()
                .idempotencyKey(idempotencyKey)
                .receiverWalletId(wallet.getId())
                .amount(amount)
                .type("DEPOSIT")
                .status("COMPLETED")
                .description("Deposit to wallet")
                .build();
        return transactionRepository.save(txn);
    }

    @Transactional(readOnly = true)
    public Page<Transaction> getHistory(UUID userId, String type, Pageable pageable) {
        Wallet wallet = walletRepository.findByUserId(userId)
                .orElseThrow(() -> new BusinessException("Wallet not found", HttpStatus.NOT_FOUND));

        if (type != null && !type.isBlank()) {
            return transactionRepository.findByWalletIdAndType(wallet.getId(), type.toUpperCase(), pageable);
        }
        return transactionRepository.findByWalletId(wallet.getId(), pageable);
    }

    @Transactional(readOnly = true)
    public Transaction getById(UUID txnId) {
        return transactionRepository.findById(txnId)
                .orElseThrow(() -> new BusinessException("Transaction not found", HttpStatus.NOT_FOUND));
    }

    private Transaction createTransaction(TransferRequest req, UUID senderId, UUID receiverId, String status) {
        return Transaction.builder()
                .idempotencyKey(req.getIdempotencyKey())
                .senderWalletId(senderId)
                .receiverWalletId(receiverId)
                .amount(req.getAmount())
                .type("TRANSFER")
                .status(status)
                .description(req.getDescription())
                .build();
    }
}
