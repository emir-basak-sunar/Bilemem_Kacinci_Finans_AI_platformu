package com.finai.wallet.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.wallet.dto.AmountRequest;
import com.finai.wallet.dto.WalletResponse;
import com.finai.wallet.service.WalletService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/wallet")
@RequiredArgsConstructor
public class WalletController {

    private final WalletService walletService;

    @GetMapping
    public ResponseEntity<ApiResponse<WalletResponse>> getWallet(Authentication auth) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok(walletService.getWallet(userId)));
    }

    @PostMapping("/deposit")
    public ResponseEntity<ApiResponse<WalletResponse>> deposit(
            Authentication auth, @Valid @RequestBody AmountRequest request) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok("Deposit successful", walletService.deposit(userId, request.getAmount())));
    }

    @PostMapping("/withdraw")
    public ResponseEntity<ApiResponse<WalletResponse>> withdraw(
            Authentication auth, @Valid @RequestBody AmountRequest request) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok("Withdrawal successful", walletService.withdraw(userId, request.getAmount())));
    }
}
