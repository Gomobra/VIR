use vir_common::ModelConfig;

#[test]
fn default_config_valid() {
    let c = ModelConfig::default();
    assert!(c.validate().is_ok());
    assert_eq!(c.levels().len(), 12);
    assert_eq!(*c.levels().last().unwrap(), 128);
}

#[test]
fn invalid_config_rejected() {
    let c = ModelConfig {
        hidden_dim: 769,
        ..ModelConfig::default()
    };
    assert!(c.validate().is_err());
    let c2 = ModelConfig {
        num_levels: 200,
        ..ModelConfig::default()
    };
    assert!(c2.validate().is_err());
    assert!(c2.validate().is_err());
}
