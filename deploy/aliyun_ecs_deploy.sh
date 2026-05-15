#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/cat_color_project}"
REPO_URL="${REPO_URL:-https://github.com/xinggaowu20-a11y/cat_color_project.git}"
BRANCH="${BRANCH:-main}"
HOST_PORT="${HOST_PORT:-80}"
CONTAINER_PORT="${CONTAINER_PORT:-7860}"
IMAGE_NAME="${IMAGE_NAME:-cat-color-project:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-cat-color-project}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run this script as root or with sudo."
  exit 1
fi

install_package() {
  local package="$1"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "$package"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y "$package"
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y "$package"
  else
    echo "Unsupported Linux package manager. Please install $package manually."
    exit 1
  fi
}

if ! command -v git >/dev/null 2>&1; then
  install_package git
fi

if ! command -v docker >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc || true
    chmod a+r /etc/apt/keyrings/docker.asc || true
    . /etc/os-release
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME:-jammy} stable" > /etc/apt/sources.list.d/docker.list
    apt-get update || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io || DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io
  else
    install_package docker
  fi
fi

systemctl enable docker >/dev/null 2>&1 || true
systemctl start docker >/dev/null 2>&1 || service docker start >/dev/null 2>&1 || true

if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
  mkdir -p "$(dirname "$APP_DIR")"
  rm -rf "$APP_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

docker build -t "$IMAGE_NAME" .
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -e PORT="$CONTAINER_PORT" \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  "$IMAGE_NAME"

echo "Deployment complete."
echo "Local health check:"
curl -fsS "http://127.0.0.1:${HOST_PORT}/health" || true
echo
echo "Open: http://<your-ecs-public-ip>/"
