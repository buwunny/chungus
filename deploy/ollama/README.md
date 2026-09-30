# chungus in front of Ollama

`chungus ollama` listens on Ollama's default port (11434) and forwards every call to the
real Ollama, moved to 11433. Pulls are filled from this machine and LAN peers first, then
from registry.ollama.ai. Peers reach each other on port 7447/tcp (open it to the LAN) and
find each other over mDNS.

## Linux service (the official install script)

The service runs as the `ollama` user and keeps models in
`/usr/share/ollama/.ollama/models`. chungus runs as the same user so it can write there.

```sh
sudo install -m 755 chungus /usr/local/bin/chungus
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo cp ollama.service.d-chungus.conf /etc/systemd/system/ollama.service.d/chungus.conf
sudo cp chungus-ollama.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl enable --now chungus-ollama

# Let this machine seed the models it already has (reads each once, copies nothing)
sudo -u ollama chungus ollama import \
  --models /usr/share/ollama/.ollama/models --store /var/lib/chungus-ollama/store
```

`ollama pull`, `ollama run` and other clients need no change. Logs:
`journalctl -u chungus-ollama`.

## Docker (Linux hosts)

```sh
docker compose up -d
docker compose exec ollama ollama pull llama3.2
```

## On a laptop or desktop

```sh
OLLAMA_HOST=127.0.0.1:11433 ollama serve     # or quit the Ollama app and run this
chungus ollama
```

## Without the shim

`chungus ollama pull llama3.2` writes the blobs and the manifest straight into Ollama's
models directory; Ollama sees the model on its next `ollama list`. Useful in scripts and
image builds.
