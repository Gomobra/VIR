use proptest::prelude::*;
use vir_common::{LevelScheduleKind, level_schedule};

proptest! {
    #[test]
    fn schedule_valid_for_all_kinds(l in 1usize..=256, k in 1usize..=16) {
        // skip cases where k > l (impossible to be strictly increasing)
        prop_assume!(k <= l);
        for kind in [
            LevelScheduleKind::Linear,
            LevelScheduleKind::Quadratic,
            LevelScheduleKind::Sqrt,
            LevelScheduleKind::Log,
            LevelScheduleKind::Exp,
            LevelScheduleKind::Cosine,
        ] {
            let v = level_schedule(kind, l, k);
            prop_assert_eq!(v.len(), k);
            for w in v.windows(2) {
                prop_assert!(w[0] < w[1], "not strictly increasing: {:?}", v);
            }
            prop_assert_eq!(*v.last().unwrap(), l);
            for x in &v {
                prop_assert!(*x >= 1 && *x <= l);
            }
        }
    }
}
