package com.finai.user.controller;

import com.finai.common.dto.ApiResponse;
import com.finai.user.dto.ChangePasswordRequest;
import com.finai.user.dto.UpdateProfileRequest;
import com.finai.user.dto.UserProfileResponse;
import com.finai.user.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/me")
    public ResponseEntity<ApiResponse<UserProfileResponse>> getProfile(Authentication auth) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok(userService.getProfile(userId)));
    }

    @PutMapping("/me")
    public ResponseEntity<ApiResponse<UserProfileResponse>> updateProfile(
            Authentication auth,
            @Valid @RequestBody UpdateProfileRequest request) {
        UUID userId = (UUID) auth.getPrincipal();
        return ResponseEntity.ok(ApiResponse.ok("Profile updated", userService.updateProfile(userId, request)));
    }

    @PutMapping("/me/password")
    public ResponseEntity<ApiResponse<Void>> changePassword(
            Authentication auth,
            @Valid @RequestBody ChangePasswordRequest request) {
        UUID userId = (UUID) auth.getPrincipal();
        userService.changePassword(userId, request);
        return ResponseEntity.ok(ApiResponse.ok("Password changed successfully", null));
    }
}
