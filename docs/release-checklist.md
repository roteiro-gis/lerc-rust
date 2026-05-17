# Release Checklist

Use this checklist for coordinated `lerc-rust` workspace releases.

## Pre-Release

1. Update `[workspace.package].version` in the root `Cargo.toml`.
2. Update all internal dependency pins to the same version.
3. Update `CHANGELOG.md` with the release date and user-facing changes.
4. Run the standard workspace checks:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Package Validation

Verify the leaf crates locally before publishing:

```sh
cargo package -p lerc-core --offline
cargo package -p lerc-band-materialize --offline
```

For `lerc-reader` and `lerc-writer`, `cargo package` / `cargo publish --dry-run`
resolve their internal dependencies from crates.io once path dependencies are
rewritten. That means full package preparation only succeeds after matching
versions of `lerc-core` and `lerc-band-materialize` have already been
published. Before that point, use `--list` for tarball contents sanity checks:

```sh
cargo package -p lerc-reader --list
cargo package -p lerc-writer --list
```

After the dependency crates are live on crates.io, rerun dry-runs for the
dependent crates:

```sh
cargo publish --dry-run -p lerc-reader
cargo publish --dry-run -p lerc-writer
```

## Publish Order

Publish in dependency order:

1. `lerc-core`
2. `lerc-band-materialize`
3. `lerc-reader`
4. `lerc-writer`
