#!/usr/bin/env bash
# Day-1 node probe (docs/design.md section 14, risk 1): GPUs, Docker, compose v2, Harbor egress-control kernel support, capacity.
# Exits non-zero if anything training depends on is missing.
set -uo pipefail
FAIL=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; }

echo "== GPUs"
if command -v nvidia-smi >/dev/null; then
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
  (( n >= 4 )) && ok "$n GPUs (two concurrent runs: train GPU + vLLM GPU each)" || warn "$n GPUs: one run needs 2 (server layout) or 1 (configs/colocate_1gpu.yaml)"
  nvidia-smi topo -m 2>/dev/null | head -6 | sed 's/^/  /'
else
  bad "nvidia-smi not found"
fi

echo "== Docker"
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  ok "docker daemon reachable ($(docker version --format '{{.Server.Version}}'))"
  ok "storage driver: $(docker info --format '{{.Driver}}'), root: $(docker info --format '{{.DockerRootDir}}')"
else
  bad "docker daemon not reachable (container-based pods usually lack it: rent a VM / bare metal)"
fi
if docker compose version >/dev/null 2>&1; then ok "$(docker compose version)"; else bad "docker compose v2 plugin missing"; fi

echo "== Harbor egress control (no-network / allowlist policies)"
# Exactly Harbor 0.24.0's probe (unchanged since 0.23.0) (harbor/environments/docker/docker.py::_egress_control_kernel_support). If it fails,
# Harbor silently disables the sidecar and any non-public network policy raises ValueError at boot.
PROBE_IMG="alpine:3.23.4@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11"
PROBE_SH="if [ ! -f /proc/config.gz ]; then exit 0; fi; zcat /proc/config.gz 2>/dev/null | grep -qE '^CONFIG_NFT_FIB_INET=[ym]'"
if timeout 60 docker container run --rm "$PROBE_IMG" sh -c "$PROBE_SH" >/dev/null 2>&1; then
  ok "Harbor kernel probe passed"
else
  bad "Harbor kernel probe failed: CONFIG_NFT_FIB_INET missing -> network policies cannot be enforced"
fi
# Harbor's probe is optimistic when /proc/config.gz is absent (the Ubuntu default), so check the host config too.
KCFG="/boot/config-$(uname -r)"
if [[ -f /proc/config.gz ]]; then
  :
elif [[ -f "$KCFG" ]]; then
  if grep -qE '^CONFIG_NFT_FIB_INET=[ym]' "$KCFG"; then ok "$KCFG: CONFIG_NFT_FIB_INET enabled"
  else bad "$KCFG: CONFIG_NFT_FIB_INET not enabled (Harbor's probe would wrongly pass)"; fi
else
  warn "no kernel config found; relying on the functional test below"
fi
# Functional test: the rule the sidecar actually installs (`fib daddr type local`), in a NET_ADMIN container.
NFT_TEST='apk add --no-cache nftables >/dev/null 2>&1 && nft add table inet probe && nft add chain inet probe out "{ type filter hook output priority 0; }" && nft add rule inet probe out fib daddr type local accept'
if timeout 120 docker container run --rm --cap-add NET_ADMIN "$PROBE_IMG" sh -c "$NFT_TEST" >/dev/null 2>&1; then
  ok "nftables 'fib daddr type local' rule installs in a NET_ADMIN container"
else
  bad "could not install an nftables fib rule (or apk could not reach the network)"
fi

echo "== Capacity (each task container: ~1 CPU, 2 GB; up to 64 agent + 64 grading containers)"
cpus=$(nproc); mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
(( cpus >= 96 )) && ok "$cpus CPUs" || warn "$cpus CPUs: lower env.max_concurrent_boots / batch size"
(( mem_gb >= 256 )) && ok "${mem_gb} GB RAM" || warn "${mem_gb} GB RAM"
root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
df -h "$root" 2>/dev/null | tail -1 | awk '{print "  disk at docker root: " $4 " free of " $2}'
df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | tail -1 | awk '{print "  disk at HF cache: " $4 " free"}'
nofile=$(ulimit -n)
(( nofile >= 65536 )) && ok "ulimit -n $nofile" || warn "ulimit -n $nofile (raise to >=65536: many compose subprocesses + sockets)"
nproc_lim=$(ulimit -u); ok "ulimit -u $nproc_lim"
ip_fwd=$(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null || echo ?); ok "net.ipv4.ip_forward=$ip_fwd"

echo
if (( FAIL )); then echo "PROBE FAILED"; exit 1; fi
echo "PROBE OK"
