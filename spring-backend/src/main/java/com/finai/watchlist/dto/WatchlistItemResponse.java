package com.finai.watchlist.dto;

import lombok.Builder;
import lombok.Data;

import java.util.UUID;

@Data
@Builder
public class WatchlistItemResponse {
    private UUID id;
    private String symbol;
    private String name;
    private int position;
}
