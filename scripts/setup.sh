#!/usr/bin/env bash
# SciKnow setup script — safe to run multiple times.
# Assumes Ubuntu/Debian and an RTX 3090 with CUDA 12.x drivers installed.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Helpers ────────────────────────────────────────────────────────────────

ok()   { echo "  ✓ $*"; }
skip() { echo "  · $* (already done)"; }
info() { echo "  → $*"; }
warn() { echo "  ! $*"; }

# Is a system service active?
svc_active()      { systemctl is-active --quiet "$1" 2>/dev/null; }
# Is a user-level systemd service active?
user_svc_active() { systemctl --user is-active --quiet "$1" 2>/dev/null; }
# Is a user-level service enabled (even if not yet started)?
user_svc_exists() { [ -f "$HOME/.config/systemd/user/$1.service" ]; }
# Is an Ollama model already downloaded?
ollama_has_model() { ollama list 2>/dev/null | grep -q "^$1[[:space:]]"; }

# ── Banner ─────────────────────────────────────────────────────────────────

echo "========================================="
echo " SciKnow Setup"
echo "========================================="

# ── 1. PostgreSQL ──────────────────────────────────────────────────────────

echo ""
echo "[1/6] PostgreSQL"

if ! command -v psql &>/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y postgresql postgresql-client
    ok "Installed PostgreSQL"
else
    skip "PostgreSQL binary present"
fi

# Ensure the service is running before we try to talk to it
if ! svc_active postgresql; then
    sudo systemctl enable --now postgresql
    ok "PostgreSQL service started"
else
    skip "PostgreSQL service already running"
fi

# Create role (idempotent)
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='sciknow'" | grep -q 1; then
    sudo -u postgres psql -c "CREATE USER sciknow WITH PASSWORD 'sciknow';"
    ok "Role 'sciknow' created"
else
    skip "Role 'sciknow' already exists"
fi

# Create database (idempotent)
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='sciknow'" | grep -q 1; then
    sudo -u postgres psql -c "CREATE DATABASE sciknow OWNER sciknow;"
    ok "Database 'sciknow' created"
else
    skip "Database 'sciknow' already exists"
fi

# ── 2. Qdrant ──────────────────────────────────────────────────────────────

echo ""
echo "[2/6] Qdrant"

QDRANT_DIR="$HOME/.local/qdrant"
QDRANT_BIN="$QDRANT_DIR/qdrant"

# Download binary if missing
if [ ! -f "$QDRANT_BIN" ]; then
    mkdir -p "$QDRANT_DIR"
    QDRANT_URL=$(curl -s https://api.github.com/repos/qdrant/qdrant/releases/latest \
        | grep "browser_download_url.*x86_64-unknown-linux-gnu.tar.gz" \
        | cut -d '"' -f 4)
    if [ -z "$QDRANT_URL" ]; then
        warn "Could not fetch Qdrant release URL from GitHub API. Check your internet connection."
        exit 1
    fi
    curl -L "$QDRANT_URL" | tar -xz -C "$QDRANT_DIR"
    chmod +x "$QDRANT_BIN"
    ok "Qdrant binary downloaded to $QDRANT_BIN"
else
    skip "Qdrant binary already at $QDRANT_BIN"
fi

# Create systemd user service if missing
if ! user_svc_exists qdrant; then
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/qdrant.service" <<EOF
[Unit]
Description=Qdrant vector database
After=network.target

[Service]
ExecStart=$QDRANT_BIN
WorkingDirectory=$QDRANT_DIR
Restart=on-failure
Environment=QDRANT__SERVICE__HTTP_PORT=6333

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    ok "Qdrant systemd service created"
else
    skip "Qdrant systemd service already exists"
fi

# Enable and start the service if not already running
if ! user_svc_active qdrant; then
    systemctl --user enable --now qdrant
    ok "Qdrant service started"
else
    skip "Qdrant service already running"
fi

# ── 3. Ollama ──────────────────────────────────────────────────────────────

echo ""
echo "[3/6] Ollama"

if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    skip "Ollama already installed"
fi

# Ensure the Ollama service is running before pulling models
if ! svc_active ollama && ! pgrep -x ollama &>/dev/null; then
    warn "Ollama service does not appear to be running. Attempting to start..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Pull models only if not already present
FAST_MODEL="mistral:7b-instruct-q4_K_M"
MAIN_MODEL="qwen2.5:32b-instruct-q4_K_M"

if ollama_has_model "$FAST_MODEL"; then
    skip "Model $FAST_MODEL already downloaded"
else
    info "Pulling $FAST_MODEL (this may take a while)..."
    ollama pull "$FAST_MODEL"
    ok "Model $FAST_MODEL ready"
fi

# Main LLM — uncomment when ready (~19 GB VRAM required)
# if ollama_has_model "$MAIN_MODEL"; then
#     skip "Model $MAIN_MODEL already downloaded"
# else
#     info "Pulling $MAIN_MODEL (large download, ~20 GB)..."
#     ollama pull "$MAIN_MODEL"
#     ok "Model $MAIN_MODEL ready"
# fi

# ── 4. Python environment (uv) ─────────────────────────────────────────────

echo ""
echo "[4/6] Python environment"

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this shell session without reloading .bashrc
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed"
else
    skip "uv already installed"
fi

# uv sync is always safe to re-run — it only installs what's missing
uv sync
ok "Python dependencies up to date"

# ── 5. Marker (marker-pdf) ─────────────────────────────────────────────────

echo ""
echo "[5/6] Marker"

if uv run python -c "import marker" &>/dev/null 2>&1; then
    skip "marker-pdf already installed"
else
    uv pip install marker-pdf
    ok "marker-pdf installed"
fi
info "Marker models (Surya OCR, layout) download automatically on first use from HuggingFace"
info "GPU is used automatically when CUDA is available"

# ── 6. .env file ───────────────────────────────────────────────────────────

echo ""
echo "[6/6] Configuration"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    ok ".env created from .env.example"
    info "Edit .env and set CROSSREF_EMAIL to your email address"
else
    skip ".env already exists"
fi

# ── Final summary ──────────────────────────────────────────────────────────

echo ""
echo "========================================="
echo " Setup complete!"
echo ""
echo " Service status:"
svc_active postgresql  && echo "   ✓ PostgreSQL running" || echo "   ✗ PostgreSQL NOT running"
user_svc_active qdrant && echo "   ✓ Qdrant running"     || echo "   ✗ Qdrant NOT running"
svc_active ollama || pgrep -x ollama &>/dev/null \
                   && echo "   ✓ Ollama running"      || echo "   ✗ Ollama NOT running"
echo ""
echo " Next steps:"
echo "   cd $PROJECT_DIR"
echo "   nano .env                  # set CROSSREF_EMAIL"
echo "   sciknow db init            # create schema + Qdrant collections"
echo "   sciknow db stats           # verify everything is running"
echo "   sciknow ingest file paper.pdf"
echo "========================================="
