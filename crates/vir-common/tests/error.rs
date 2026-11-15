//! F1.1 — galat terstruktur `VirError` (Display/From) (PRD F1.1).

use vir_common::VirError;

#[test]
fn display_tiap_varian() {
    assert_eq!(VirError::Config("x".into()).to_string(), "x");
    assert!(
        VirError::Internal("boom".into())
            .to_string()
            .contains("boom")
    );
    let io = VirError::from(std::io::Error::other("gagal"));
    assert!(io.to_string().contains("gagal"), "From<io::Error>");
}

#[test]
fn from_serde_yaml() {
    let err: serde_yaml::Error =
        serde_yaml::from_str::<u32>("bukan angka").expect_err("harus gagal parse");
    let v = VirError::from(err);
    assert!(!v.to_string().is_empty(), "From<serde_yaml::Error>");
}

#[test]
fn plasmid_error_trait() {
    fn terima(_e: &dyn std::error::Error) {}
    terima(&VirError::Config("x".into()));
}
