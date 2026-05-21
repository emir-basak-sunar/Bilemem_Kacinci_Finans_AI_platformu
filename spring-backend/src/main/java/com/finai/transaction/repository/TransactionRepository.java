package com.finai.transaction.repository;

import com.finai.transaction.entity.Transaction;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, UUID> {

    Optional<Transaction> findByIdempotencyKey(String idempotencyKey);

    @Query("SELECT t FROM Transaction t WHERE t.senderWalletId = :walletId OR t.receiverWalletId = :walletId ORDER BY t.createdAt DESC")
    Page<Transaction> findByWalletId(UUID walletId, Pageable pageable);

    @Query("SELECT t FROM Transaction t WHERE (t.senderWalletId = :walletId OR t.receiverWalletId = :walletId) AND t.type = :type ORDER BY t.createdAt DESC")
    Page<Transaction> findByWalletIdAndType(UUID walletId, String type, Pageable pageable);
}
