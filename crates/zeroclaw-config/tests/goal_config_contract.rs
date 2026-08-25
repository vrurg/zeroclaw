use zeroclaw_config::schema::Config;

#[test]
fn goal_mode_is_disabled_by_default_and_requires_a_resolved_verifier_when_enabled() {
    let mut config = Config::default();
    assert!(!config.goal.enabled);
    assert!(config.validate().is_ok());
    assert!(
        !toml::to_string(&config)
            .expect("default config serializes")
            .contains("[goal]"),
        "the dormant default must not alter existing serialized configuration"
    );

    config.goal.enabled = true;
    config.goal.verifier.provider = "openai.verifier".into();
    assert!(config.validate().is_err());

    let _ = config.providers.models.ensure("openai", "verifier");
    assert!(config.validate().is_ok());
}

#[test]
fn goal_limits_are_optional_but_must_be_positive_and_finite() {
    let mut config = Config::default();
    config.goal.default_token_limit = Some(0);
    assert!(config.validate().is_err());

    config.goal.default_token_limit = Some(64);
    config.goal.default_cost_limit_usd = Some(f64::INFINITY);
    assert!(config.validate().is_err());

    config.goal.default_cost_limit_usd = Some(1.25);
    let serialized = toml::to_string(&config).expect("config serializes");
    assert!(serialized.contains("[goal]"));
    let reloaded: Config = toml::from_str(&serialized).expect("config reloads");
    assert_eq!(reloaded.goal.default_token_limit, Some(64));
    assert_eq!(reloaded.goal.default_cost_limit_usd, Some(1.25));
}
