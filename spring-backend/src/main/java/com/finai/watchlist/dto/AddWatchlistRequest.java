package com.finai.watchlist.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class AddWatchlistRequest {

    @NotBlank
    @Pattern(regexp = "^[A-Za-z0-9\\-\\.]{1,10}$", message = "Invalid symbol format")
    private String symbol;

    @Size(max = 100)
    private String name;
}
