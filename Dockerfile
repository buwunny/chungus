# A chungus node in a container. Build from the repository root:
#   docker build -t chungus .
# It runs as a public relay and bootstrap node by default; see deploy/ and the README.

FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked --bin chungus && mkdir /data

# Just the binary, glibc and CA certificates: no shell or package manager.
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/chungus /usr/local/bin/chungus
# The store, and the node's identity key inside it, live here.
COPY --from=build --chown=nonroot:nonroot /data /data
# Hardened defaults for a public node. Every flag of `chungus node` has a CHUNGUS_*
# variable, so override these (or add others) in the environment.
ENV CHUNGUS_STORE=/data/store \
    CHUNGUS_LISTEN=/ip4/0.0.0.0/tcp/4001,/ip4/0.0.0.0/udp/4001/quic-v1 \
    CHUNGUS_PUBLIC=true \
    CHUNGUS_RELAY_SERVER=true \
    CHUNGUS_MAX_UPLOAD=20 \
    CHUNGUS_MAX_CONNECTIONS=400 \
    CHUNGUS_RELAY_MAX_CIRCUITS=32 \
    CHUNGUS_RELAY_CIRCUIT_MB=1024 \
    CHUNGUS_RELAY_CIRCUIT_SECS=600
VOLUME /data
EXPOSE 4001/tcp 4001/udp
ENTRYPOINT ["chungus", "node"]
