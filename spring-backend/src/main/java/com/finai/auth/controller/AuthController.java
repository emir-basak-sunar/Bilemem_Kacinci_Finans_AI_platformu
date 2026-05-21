package com.finai.auth.controller;

import com.finai.auth.dto.*;
import com.finai.auth.service.AuthService;
import com.finai.common.dto.ApiResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register")
    public ResponseEntity<ApiResponse<TokenResponse>> register(@Valid @RequestBody RegisterRequest request) {
        TokenResponse tokens = authService.register(request);
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.ok("Registration successful", tokens));
    }

    @PostMapping("/login")
    public ResponseEntity<ApiResponse<TokenResponse>> login(
            @Valid @RequestBody LoginRequest request,
            HttpServletRequest httpRequest) {
        String deviceInfo = httpRequest.getHeader("User-Agent");
        TokenResponse tokens = authService.login(request, deviceInfo);
        return ResponseEntity.ok(ApiResponse.ok("Login successful", tokens));
    }

    @PostMapping("/refresh")
    public ResponseEntity<ApiResponse<TokenResponse>> refresh(@Valid @RequestBody RefreshRequest request) {
        TokenResponse tokens = authService.refresh(request);
        return ResponseEntity.ok(ApiResponse.ok("Token refreshed", tokens));
    }

    @PostMapping("/logout")
    public ResponseEntity<ApiResponse<Void>> logout(Authentication authentication) {
        UUID userId = (UUID) authentication.getPrincipal();
        authService.logout(userId);
        return ResponseEntity.ok(ApiResponse.ok("Logout successful", null));
    }
}
