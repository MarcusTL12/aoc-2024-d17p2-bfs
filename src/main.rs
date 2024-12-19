#![feature(portable_simd)]

use std::{
    env,
    simd::{Mask, Simd, cmp::SimdPartialEq},
    time::Instant,
};

use arrayvec::ArrayVec;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const N: usize = 32;

#[inline(always)]
fn get_out(a: Simd<u64, N>) -> Simd<u64, N> {
    let mask = Simd::splat(7);

    let a8 = a & mask;

    let shifter = a8 ^ Simd::splat(2);
    let b = a8 ^ Simd::splat(1);
    let c = (a >> shifter) & mask;

    b ^ c
}

const PROGRAM: [u8; 16] = [2, 4, 1, 2, 7, 5, 1, 3, 4, 4, 5, 5, 0, 3, 3, 0];

#[inline(always)]
fn check_a(mut a: Simd<u64, N>) -> Mask<i64, N> {
    let mut i = 0;

    let mut status = Mask::splat(true);

    while i < PROGRAM.len() && status.any() && !a.simd_eq(Simd::splat(0)).all()
    {
        let o = get_out(a);
        status &= o.simd_eq(Simd::splat(PROGRAM[i] as u64));
        a >>= Simd::splat(3);

        i += 1;
    }

    status & a.simd_eq(Simd::splat(0)) & Mask::splat(i == PROGRAM.len())
}

fn check_range(lower: u64, n_vecs: u64) -> Option<u64> {
    let strider = Simd::from_array(
        (0..N as u64)
            .collect::<ArrayVec<_, N>>()
            .into_inner()
            .unwrap(),
    );

    let (a, o) = (0..n_vecs)
        .into_par_iter()
        .map(|n| Simd::splat(n * N as u64 + lower) + strider)
        .find_map_first(|a| {
            let o = check_a(a);

            o.any().then_some((a, o))
        })?;

    let &a = a.as_array();
    let o = o.to_array();

    a.into_iter().zip(o).find_map(|(a, o)| o.then_some(a))
}

fn main() {
    let mut args = env::args();

    args.next();

    let lower: u64 = args
        .next()
        .and_then(|x| x.parse().ok())
        .expect("Give lower bound as first argument");

    let upper: u64 = args
        .next()
        .and_then(|x| x.parse().ok())
        .expect("Give upper bound as first argument");

    let n_vecs = (upper - lower + 1).div_ceil(N as u64);
    let upper = lower + n_vecs * N as u64;

    println!(
        "Checking {n_vecs} vectors of width {N} in range {lower}..{upper}"
    );

    let t = Instant::now();

    let ans = check_range(lower, n_vecs);

    let t = t.elapsed();
    println!("Solving took: {t:?}");

    if let Some(x) = ans {
        println!("Found: {x}");
    } else {
        println!("Did not find answer");
    }
}
