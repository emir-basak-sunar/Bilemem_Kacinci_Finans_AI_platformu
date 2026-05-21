package com.finai.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

@Configuration
public class KafkaConfig {

    // Existing topics
    public static final String NOTIFICATION_TOPIC = "finai.notifications";
    public static final String AUDIT_TOPIC = "finai.audit";

    // ML Pipeline topics (matching docker-compose kafka-init)
    public static final String RAW_PRICES_TOPIC = "raw-prices";
    public static final String MACRO_SIGNALS_TOPIC = "macro-signals";
    public static final String MODEL_SIGNALS_TOPIC = "ts-model-signals";
    public static final String LLM_DECISIONS_TOPIC = "llm-decisions";

    @Bean
    public NewTopic notificationTopic() {
        return TopicBuilder.name(NOTIFICATION_TOPIC).partitions(1).replicas(1).build();
    }

    @Bean
    public NewTopic auditTopic() {
        return TopicBuilder.name(AUDIT_TOPIC).partitions(1).replicas(1).build();
    }

    @Bean
    public NewTopic rawPricesTopic() {
        return TopicBuilder.name(RAW_PRICES_TOPIC).partitions(6).replicas(1).build();
    }

    @Bean
    public NewTopic macroSignalsTopic() {
        return TopicBuilder.name(MACRO_SIGNALS_TOPIC).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic modelSignalsTopic() {
        return TopicBuilder.name(MODEL_SIGNALS_TOPIC).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic llmDecisionsTopic() {
        return TopicBuilder.name(LLM_DECISIONS_TOPIC).partitions(3).replicas(1).build();
    }
}
