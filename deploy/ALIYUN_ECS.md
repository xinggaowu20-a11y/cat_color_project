# Aliyun ECS Deployment

This project can run on Alibaba Cloud ECS with Docker.

## ECS Requirements

- Linux ECS, recommended Ubuntu 22.04 / Alibaba Cloud Linux 3
- At least 2 vCPU and 4 GB RAM
- Public IPv4 address or EIP
- Security group inbound rules:
  - TCP 22: SSH login, preferably restricted to your own IP
  - TCP 80: public web access

Alibaba Cloud ECS security groups work like a virtual firewall. For production, use least-privilege inbound rules and avoid opening management ports to `0.0.0.0/0` when possible.

## One-Command Deploy On ECS

SSH into your ECS, then run:

```bash
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/xinggaowu20-a11y/cat_color_project/main/deploy/aliyun_ecs_deploy.sh)"
```

After it finishes, open:

```text
http://<your-ecs-public-ip>/
```

## Manual Docker Deploy

```bash
sudo yum install -y git docker || sudo apt-get update && sudo apt-get install -y git docker.io
sudo systemctl enable --now docker
sudo git clone https://github.com/xinggaowu20-a11y/cat_color_project.git /opt/cat_color_project
cd /opt/cat_color_project
sudo docker build -t cat-color-project:latest .
sudo docker run -d --name cat-color-project --restart unless-stopped -p 80:7860 -e PORT=7860 cat-color-project:latest
```

Health check:

```bash
curl http://127.0.0.1/health
```

## Update Existing Deployment

```bash
cd /opt/cat_color_project
sudo git pull
sudo docker build -t cat-color-project:latest .
sudo docker rm -f cat-color-project
sudo docker run -d --name cat-color-project --restart unless-stopped -p 80:7860 -e PORT=7860 cat-color-project:latest
```
