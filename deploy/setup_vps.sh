#!/bin/bash
# Run as root on fresh Ubuntu 22.04
# Usage: cd /path/to/hermes && sudo bash deploy/setup_vps.sh
set -e

# ── Resolve project root (directory containing this script's parent) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="/opt/hermes"

echo "=== Hermes VPS Setup ==="
echo "Project source: $PROJECT_DIR"
echo "Install target: $INSTALL_DIR"

# Verify requirements.txt exists in source before doing anything
if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
    echo ""
    echo "ERROR: requirements.txt not found in $PROJECT_DIR"
    echo "Run this script from inside the hermes project folder:"
    echo "  cd /path/to/hermes && sudo bash deploy/setup_vps.sh"
    exit 1
fi

# ── System deps ───────────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y python3 python3-venv python3-pip \
    build-essential python3-dev libssl-dev libffi-dev
# build-essential/python3-dev/libssl-dev/libffi-dev: py-clob-client pulls in
# web3/eth-account, which commonly depend on coincurve (libsecp256k1 crypto
# bindings). Pre-built wheels cover most x86_64 setups, but ARM-based VPS
# instances or less common Python versions can miss them, forcing pip to
# compile from source — which fails with a cryptic C-toolchain error if
# these aren't already present. Installing them preemptively costs a few
# extra seconds on a one-time VPS setup and avoids that failure entirely.

# ── Create hermes user and install dir ───────────────────────────────────────
echo "[2/6] Creating hermes user and $INSTALL_DIR..."
useradd -r -s /bin/bash -d "$INSTALL_DIR" hermes 2>/dev/null || true
mkdir -p "$INSTALL_DIR"

# ── Copy project files ────────────────────────────────────────────────────────
echo "[3/6] Copying project files from $PROJECT_DIR..."
cp -r "$PROJECT_DIR"/. "$INSTALL_DIR/"
chown -R hermes:hermes "$INSTALL_DIR"

# Remove clutter that has no business in a production deploy — a blind
# cp -r copies EVERYTHING in the source tree, including dev artifacts.
find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$INSTALL_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf "$INSTALL_DIR/.git" 2>/dev/null || true

# Warn — don't silently decide — if local dev state got carried over.
# A local hermes.db could contain test/paper trade history that would
# corrupt trailing_bias() and win-rate stats if mistaken for real
# production history. A local .env could hold test or wrong-wallet
# credentials that silently pre-empt the "create fresh from .env.example"
# step below, since that step only fires when no .env exists yet.
if [ -f "$INSTALL_DIR/hermes.db" ]; then
    echo ""
    echo "⚠️  WARNING: hermes.db was present in the source directory and has"
    echo "   been copied to $INSTALL_DIR/hermes.db. If this is local test/dev"
    echo "   data, delete it now so the bot starts with a clean ledger:"
    echo "     rm $INSTALL_DIR/hermes.db"
    echo ""
fi
if [ -f "$INSTALL_DIR/.env" ]; then
    echo ""
    echo "⚠️  WARNING: a .env file was present in the source directory and has"
    echo "   been copied to $INSTALL_DIR/.env — it will NOT be overwritten by"
    echo "   the fresh-template step below. Verify it has the correct"
    echo "   credentials for THIS deployment before starting the bot:"
    echo "     cat $INSTALL_DIR/.env"
    echo ""
fi

# Verify copy succeeded
if [ ! -f "$INSTALL_DIR/requirements.txt" ]; then
    echo "ERROR: Copy failed — requirements.txt missing in $INSTALL_DIR"
    exit 1
fi
echo "      Files verified: requirements.txt present"

# ── Python venv + deps ────────────────────────────────────────────────────────
echo "[4/6] Creating virtualenv and installing dependencies..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
chown -R hermes:hermes "$INSTALL_DIR/venv"

# ── .env config ───────────────────────────────────────────────────────────────
echo "[5/6] Configuring environment..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    chown hermes:hermes "$INSTALL_DIR/.env"
    echo "      Created $INSTALL_DIR/.env from example — fill in credentials before starting"
else
    echo "      $INSTALL_DIR/.env already exists — skipping"
fi

# ── systemd service ───────────────────────────────────────────────────────────
echo "[6/6] Installing systemd services..."
cp "$INSTALL_DIR/deploy/hermes.service" /etc/systemd/system/hermes.service
cp "$INSTALL_DIR/deploy/hermes-dashboard.service" /etc/systemd/system/hermes-dashboard.service
systemctl daemon-reload
systemctl enable hermes hermes-dashboard

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Fill in credentials:      nano $INSTALL_DIR/.env"
echo "  2. Generate CLOB API creds:  cd $INSTALL_DIR && venv/bin/python generate_creds.py"
echo "  3. Start the trading bot:    systemctl start hermes"
echo "  4. Start the dashboard:      systemctl start hermes-dashboard"
echo "  5. Watch bot logs:           journalctl -u hermes -f"
echo "  6. Watch dashboard logs:     journalctl -u hermes-dashboard -f"
echo "  7. Open the dashboard:       http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "To stop:    systemctl stop hermes hermes-dashboard"
echo "To restart: systemctl restart hermes hermes-dashboard"
