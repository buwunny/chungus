//! chungus library.
//!
//! The hashing engine lives here so it can be tested and reused independently
//! of the binary's execution/orchestration code in `main.rs`.
//!
//! The `oci` module contains the OCI image manifest and config types, and the
//! `oci::Image` type which is the main entry point for building an image from
//! a directory of files.

pub mod hashing;
pub mod oci;
