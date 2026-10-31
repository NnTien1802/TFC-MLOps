# Docker TF-C HAR → Gesture (CPU)

## 1. Chuẩn bị
Giải nén các file trong ZIP vào thư mục gốc `C:\AI_projects\TFC-MLOps`. File ZIP này KHÔNG chứa checkpoint; hãy sao chép hai checkpoint vào `models/` và đặt tên `tfc_har_gesture_model.pt` và `tfc_har_gesture_classifier.pt`. Giữ nguyên `code/TFC/model.py` và `code/config_files/HAR_Configs.py` của repo.

## 2. Build
Mở Docker Desktop (Linux containers), mở CMD:

```bat
cd /d C:\AI_projects\TFC-MLOps
docker info --format "{{.OSType}}"
docker build -t tfc-mlops:v2 .
```

## 3. Chạy

```bat
docker run -d --name tfc-ai -p 8001:8000 tfc-mlops:v2
docker logs tfc-ai
```

Kiểm tra `http://127.0.0.1:8001/health`, `http://127.0.0.1:8001/docs`, `http://127.0.0.1:8001/model-info`. `/predict` nhận JSON `{"samples": [206 giá trị số]}` và trả `class_id` (0..7), xác suất của 8 lớp. KHÔNG sử dụng input ngẫu nhiên để đánh giá độ chính xác; phải tiền xử lý Gesture đúng như giai đoạn train.

## 4. Docker Hub

```bat
docker login
docker tag tfc-mlops:v2 TEN_DOCKERHUB/tfc-mlops:v2
docker push TEN_DOCKERHUB/tfc-mlops:v2
```

Image public sẽ chứa cả hai checkpoint, cần đảm bảo có quyền chia sẻ. `.dockerignore` không chặn `models/*.pt`; `.gitignore` vẫn nên chặn checkpoint khỏi git.
