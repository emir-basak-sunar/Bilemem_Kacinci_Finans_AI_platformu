package com.finai.common.config;

import com.finai.user.entity.User;
import com.finai.user.repository.UserRepository;
import com.finai.wallet.entity.Wallet;
import com.finai.wallet.repository.WalletRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Instant;

@Component
@RequiredArgsConstructor
public class AdminSeeder implements CommandLineRunner {

    private final UserRepository userRepository;
    private final WalletRepository walletRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    @Transactional
    public void run(String... args) throws Exception {
        User admin = userRepository.findByEmail("admin@gmail.com").orElse(new User());
        boolean isNew = admin.getId() == null;
        
        admin.setFullName("Admin User");
        admin.setEmail("admin@gmail.com");
        admin.setPasswordHash(passwordEncoder.encode("password123"));
        admin.setRole("USER"); 
        
        if (isNew) {
            admin.setCreatedAt(Instant.now());
        }
        admin.setUpdatedAt(Instant.now());
        
        User savedAdmin = userRepository.save(admin);
        
        if (isNew) {
            Wallet wallet = new Wallet();
            wallet.setUser(savedAdmin);
            wallet.setBalance(new BigDecimal("100000.00"));
            wallet.setCreatedAt(Instant.now());
            wallet.setUpdatedAt(Instant.now());
            walletRepository.save(wallet);
        }
        
        System.out.println("SEEDER: Ensured admin@gmail.com / password123 exists");
    }
}
