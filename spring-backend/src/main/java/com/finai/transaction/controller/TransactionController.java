package com.finai.transaction.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.transaction.dto.TransferRequest;
import com.finai.transaction.entity.Transaction;
import com.finai.transaction.service.TransactionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/transactions")
@RequiredArgsConstructor
public class TransactionController {

    private final TransactionService transactionService;

    @PostMapping("/transfer")
    public ResponseEntity<ApiResponse<Transaction>> transfer(
            Authentication auth,
            @Valid @RequestBody TransferRequest request) {
        UUID userId = (UUID) auth.getPrincipal();
        Transaction txn = transactionService.transfer(userId, request);
        return ResponseEntity.ok(ApiResponse.ok("Transfer completed", txn));
    }

    @GetMapping
    public ResponseEntity<ApiResponse<Page<Transaction>>> getHistory(
            Authentication auth,
            @RequestParam(required = false) String type,
            @PageableDefault(size = 20) Pageable pageable) {
        UUID userId = (UUID) auth.getPrincipal();
        Page<Transaction> history = transactionService.getHistory(userId, type, pageable);
        return ResponseEntity.ok(ApiResponse.ok(history));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<Transaction>> getById(@PathVariable UUID id) {
        Transaction txn = transactionService.getById(id);
        return ResponseEntity.ok(ApiResponse.ok(txn));
    }
}
