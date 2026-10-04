//! Records the resolved versions of the Xet crates from `Cargo.lock`, so every result file names the exact engine
//! that produced it. A git dependency is recorded as `<version>+git.<short rev>`, because its version number alone
//! does not say which commit was built.

use std::fs;

const STAMPED_CRATES: [&str; 5] = ["hf-xet", "xet-client", "xet-core-structures", "xet-data", "xet-runtime"];

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let lock = fs::read_to_string("Cargo.lock").expect("Cargo.lock is committed next to Cargo.toml");
    let mut versions = Vec::new();
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        let Some(name) = line.strip_prefix("name = \"").and_then(|rest| rest.strip_suffix('"')) else {
            continue;
        };
        if !STAMPED_CRATES.contains(&name) {
            continue;
        }
        let version = lines
            .next()
            .and_then(|next| next.strip_prefix("version = \""))
            .and_then(|rest| rest.strip_suffix('"'))
            .expect("Cargo.lock lists version right after name");
        let git_rev = lines
            .next()
            .and_then(|source| source.strip_prefix("source = \"git+"))
            .and_then(|rest| rest.rsplit_once('#'))
            .map(|(_, rev)| rev.trim_end_matches('"').chars().take(7).collect::<String>());
        match git_rev {
            Some(rev) => versions.push(format!("{name}={version}+git.{rev}")),
            None => versions.push(format!("{name}={version}")),
        }
    }
    println!("cargo:rustc-env=CKPT_DEDUP_CRATE_VERSIONS={}", versions.join(","));
}
