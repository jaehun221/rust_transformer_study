pub fn argmax(xs: &[f32]) -> usize {
    assert!(!xs.is_empty(), "argmax input must not be empty");

    let mut best_idx = 0;
    let mut best_val = xs[0];

    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    best_idx
}