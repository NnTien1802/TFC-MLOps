# TF-C MLOps — CI/CD với GitHub Actions và Render

Repository thực hành triển khai **CI/CD cho mô hình TF-C (Time-Frequency Consistency)** trên dữ liệu chuỗi thời gian. Dự án giữ kiến trúc TF-C gốc với hai nhánh **Time Encoder** và **Frequency Encoder**; phần bổ sung phục vụ bài thực hành gồm kiểm thử tự động, API kiểm tra trạng thái và cấu hình triển khai lên Render.

- **Repository:** [NnTien1802/TFC-MLOps](https://github.com/NnTien1802/TFC-MLOps)
- **Bài toán thí nghiệm:** Pre-training trên **HAR**, fine-tuning/testing trên **Gesture**.
- **CI:** GitHub Actions kiểm tra cú pháp, forward/backward của TF-C và FastAPI.
- **CD (cách A):** Render deploy trực tiếp từ GitHub khi **CI checks pass**.
- **Giới hạn hiện tại:** API chỉ phục vụ kiểm tra triển khai, **chưa cung cấp dự đoán Gesture**. Docker Hub là yêu cầu thực hành riêng, chưa nằm trong luồng này.

## 1. Project Overview

### Mục tiêu

Đưa một dự án nghiên cứu AI từ môi trường chạy cục bộ sang quy trình kiểm thử và triển khai có thể lặp lại:

1. Quản lý mã nguồn TF-C trên GitHub.
2. Tự động chạy kiểm thử mỗi khi thay đổi nhánh `main` hoặc tạo Pull Request.
3. Chỉ cho phép Render tự triển khai phiên bản mới sau khi CI thành công.
4. Cung cấp endpoint công khai để xác minh dịch vụ đang chạy.

**Phạm vi:** Bài thực hành kiểm thử và triển khai mã nguồn, **không** thực hiện pre-training/fine-tuning lại trên GitHub Actions hoặc Render sau mỗi commit.

## 2. System Overview

### Luồng CI/CD

```text
Developer (commit / push)
          |
          v
GitHub Repository (main)
          |
          v
GitHub Actions (CI)
  |-- Kiểm tra cú pháp Python
  |-- Kiểm thử TF-C: time/frequency + classifier
  |-- Kiểm thử FastAPI
  `-- Đóng gói source artifact (push lên main)
          |
          | Chỉ khi CI checks thành công
          v
Render Web Service (CD từ GitHub)
          |
          v
GET /health  ->  {"status":"ok"}
```

### Cấu trúc thư mục cần thiết

```text
TFC-MLOps/
├── .github/workflows/tfc-ci.yml  # GitHub Actions
├── app/main.py                    # FastAPI - deployment status
├── code/
│   ├── TFC/                       # Model, trainer, dataloader, loss...
│   └── config_files/             # HAR_Configs.py và cấu hình khác (nếu có)
├── tests/
│   ├── test_tfc_baseline.py       # Smoke test TF-C gốc
│   └── test_render_api.py         # Kiểm thử API
├── requirements-ci.txt
├── requirements-render.txt
├── requirements-render-test.txt
├── .gitignore
├── LICENSE
└── README.md
```

**Không có cơ sở dữ liệu hoặc frontend riêng.** Render chạy một dịch vụ FastAPI tối giản; việc huấn luyện TF-C vẫn được thực hiện bằng mã trong `code/TFC/` trên máy phù hợp.

## 3. Environment

| Thành phần | Phiên bản / yêu cầu | Vai trò |
| --- | --- | --- |
| Hệ điều hành local | Windows (đã thử nghiệm) | Chạy TF-C và kiểm thử API |
| Python local | **3.8.20**, Conda `tfc` | Môi trường TF-C sẵn có, không cần tạo môi trường mới |
| PyTorch | Theo môi trường `tfc` hiện có | Mô hình TF-C |
| FastAPI | **0.115.12** | Web API |
| Uvicorn local | **0.30.6** | Chạy API, tương thích Python 3.8 |
| pytest local | **8.3.5** | Kiểm thử |
| GitHub Actions | **Python 3.11**; PyTorch **2.5.1 CPU** | Chạy CI tự động |
| Render | Web Service Python; chọn runtime tương thích với requirements | Triển khai API sau CI |
| Database | Không sử dụng | — |

> **Lưu ý phiên bản:** Các số phiên bản local là cấu hình đã thử trên máy thực hành. Trong GitHub Actions, các phiên bản được quy định tại `.github/workflows/tfc-ci.yml` và các file `requirements-*.txt`. Cần giữ chúng tương thích khi thay đổi runtime Render.

## 4. Setup

**Vị trí chạy lệnh:** CMD hoặc Anaconda Prompt trên Windows. Dùng môi trường Conda `tfc` đã có.

### Bước 1. Lấy mã nguồn

```bat
git clone https://github.com/NnTien1802/TFC-MLOps.git
cd TFC-MLOps
```

**Kết quả mong đợi:** Có các thư mục `code/TFC`, `app`, `tests` và `.github/workflows`.

### Bước 2. Kích hoạt Python và cài API

```bat
conda activate tfc
python --version
python -m pip install -r requirements-render-test.txt
```

**Kết quả mong đợi:** Python `3.8.20` trên máy thực hành; cài thành công FastAPI, Uvicorn, httpx và pytest. Các dependency TF-C (bao gồm PyTorch) được quản lý trong môi trường `tfc` sẵn có.

### Bước 3. Chuẩn bị dữ liệu nếu muốn chạy thí nghiệm TF-C

Dữ liệu **không được lưu trong GitHub Repository**. Để chạy lại HAR → Gesture trên máy cá nhân, đặt dữ liệu đã tiền xử lý theo cấu trúc mà `code/TFC/main.py` và `dataloader.py` đang sử dụng:

```text
datasets/
├── HAR/
│   ├── train.pt
│   └── test.pt
└── Gesture/
    ├── train.pt
    └── test.pt
```

Các tệp `.pt` cần có cấu trúc dữ liệu phù hợp với `dataloader.py` (đặc biệt là `samples` và `labels`). Các tệp chia tập khác, nếu pipeline yêu cầu, cũng phải được đặt đúng tên. **Bước này không cần thiết để chạy CI smoke test hoặc API `/health`.**

## 5. Run & Verify

### 5.1. Kiểm thử FastAPI

**Vị trí chạy:** thư mục gốc `TFC-MLOps/`.

```bat
python -m pytest -v tests/test_render_api.py
```

**Kết quả mong đợi:**

```text
tests/test_render_api.py::test_health PASSED
tests/test_render_api.py::test_info_makes_no_inference_claim PASSED
2 passed
```

### 5.2. Kiểm thử kiến trúc TF-C

**Vị trí chạy:** thư mục gốc `TFC-MLOps/`. Cần có PyTorch trong môi trường hiện tại.

```bat
python -m pytest -v tests/test_tfc_baseline.py
```

**Kết quả mong đợi:** 2 bài smoke test thành công: kiểm tra shape của hai encoder/projector và gradient của classifier. Đây **không phải** kết quả accuracy trên Gesture.

### 5.3. Chạy API local

**Vị trí chạy:** thư mục gốc `TFC-MLOps/`.

```bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt tại **http://127.0.0.1:8000/health** (không dùng `0.0.0.0` làm URL trình duyệt).

**Expected response:**

```json
{"status":"ok"}
```

### 5.4. Kiểm tra CI và CD

1. Push một commit lên nhánh `main`.
2. Mở tab **Actions** tại GitHub và kiểm tra workflow **TF-C Original - CI**.
3. Xác nhận job **Validate TF-C original on CPU** và **Package verified source** hoàn thành thành công.
4. Tại Render Web Service kết nối cùng repository/nhánh `main`, chọn **Auto-Deploy: After CI Checks Pass**.
5. Sau khi Render báo deploy thành công, truy cập `https://<your-render-service>.onrender.com/health` và xác minh phản hồi JSON như trên.

**Chú ý:** GitHub Actions và Render chỉ được xem là hoạt động thành công khi giao diện của từng nền tảng xác nhận kết quả. README không thay thế bằng chứng chạy thực tế.

## 6. Usage / API

FastAPI hiện có các endpoint sau:

| Method | Endpoint | Mục đích |
| --- | --- | --- |
| `GET` | `/` | Thông tin về dự án và giới hạn API |
| `GET` | `/health` | Kiểm tra trạng thái dịch vụ |
| `GET` | `/model-info` | Kiểm tra source TF-C và checkpoint trong repository |
| `GET` | `/docs` | Trang Swagger UI tự động |

**Ví dụ (CMD / Anaconda Prompt, khi API đang chạy):**

```bat
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

**Expected response `/health`:**

```json
{"status":"ok"}
```

**Ví dụ phản hồi `/model-info` khi repository có source TF-C nhưng không chứa checkpoint:**

```json
{
  "model_source_present": true,
  "checkpoint_in_repo": false,
  "status": "CI/CD deployment demonstration; inference not enabled"
}
```

> API hiện **không có** endpoint `/predict`. Các trạng thái và thông tin trả về không được hiểu là kết quả phân loại Gesture hoặc độ chính xác của mô hình.

## 7. AI Model

| Thuộc tính | Thông tin |
| --- | --- |
| Model | **TF-C (Time-Frequency Consistency)** — kiến trúc gốc sử dụng Transformer Encoder |
| Bài toán | Học biểu diễn chuỗi thời gian, sau đó fine-tune để phân loại |
| Dữ liệu thí nghiệm | HAR (nguồn, pre-training) → Gesture (đích, fine-tuning/testing) |
| Time Encoder | `self.transformer_encoder_t` trong `code/TFC/model.py` |
| Frequency Encoder | `self.transformer_encoder_f` trong `code/TFC/model.py` |
| Đầu vào encoder trong smoke test | Hai tensor `x_t`, `x_f` có dạng `[B, 1, 206]`; nhánh tần số nhận biên độ FFT |
| Đầu ra projector | `z_time`, `z_freq` — mỗi vector 128 chiều |
| Đầu ra classifier | Logits `[B, num_classes_target]`; cấu hình HAR đang sử dụng `num_classes_target = 2` |
| File trọng số | `ckp_last.pt` khi pre-training; trọng số classifier fine-tune cần được xác định riêng |
| Đánh giá thực nghiệm | Chạy trong môi trường TF-C với dataset thực, **không** phải trong API status |

**Vị trí code chính:** [`code/TFC/model.py`](code/TFC/model.py), [`code/TFC/main.py`](code/TFC/main.py), [`code/TFC/trainer.py`](code/TFC/trainer.py) và [`code/config_files/HAR_Configs.py`](code/config_files/HAR_Configs.py).

**Giới hạn hiện tại:**

- CI dùng dữ liệu giả lập để kiểm tra tính đúng đắn cơ bản của quá trình tính toán; không tính accuracy/F1 trên Gesture.
- Log fine-tuning hiện có cho thấy đã chạy 40 epochs, nhưng không nên suy ra accuracy nếu chưa đối chiếu kết quả đánh giá thực tế.
- Checkpoint encoder riêng không đủ để khẳng định đã có hệ thống dự đoán Gesture hoàn chỉnh; cần checkpoint classifier, mapping nhãn và quy trình tiền xử lý thống nhất.
- Cấu hình số lớp mục tiêu phải khớp nhãn dữ liệu thực tế trước khi đánh giá hoặc bật inference.

## 8. Troubleshooting

| Hiện tượng | Nguyên nhân có thể | Cách xử lý |
| --- | --- | --- |
| `No matching distribution found for uvicorn==0.34.2` | Python local 3.8 không tương thích | Dùng `uvicorn==0.30.6` trong requirements local hoặc một runtime tương thích |
| `ERR_ADDRESS_INVALID` khi mở `0.0.0.0:8000` | `0.0.0.0` là địa chỉ bind server, không phải URL truy cập | Mở `http://127.0.0.1:8000/health` |
| `ModuleNotFoundError` | Thiếu package trong môi trường đang dùng | Kiểm tra `conda activate tfc`; cài requirements thích hợp |
| `Address already in use` | Cổng 8000 đang bị chiếm | Dừng server cũ hoặc dùng cổng khác |
| GitHub Actions không chạy | Workflow không có trên `main` hoặc trigger không phù hợp | Kiểm tra `.github/workflows/tfc-ci.yml` và tên nhánh |
| Render không tự deploy | Chưa chọn `After CI Checks Pass`, CI thất bại hoặc kết nối repository sai | Kiểm tra mục Settings / Deploys của Render và GitHub Actions |
| Không tìm thấy dữ liệu `.pt` | Dữ liệu gốc không được commit vào repo | Đặt dữ liệu vào đúng thư mục `datasets/` khi chạy thử nghiệm local |
| Không tìm thấy checkpoint classifier | Chỉ có checkpoint pre-training của encoder | Bổ sung checkpoint classifier đã fine-tune trước khi xây dựng API dự đoán |
| Cảnh báo `typing-extensions` với TensorFlow 2.13 | Dependency của TensorFlow xung đột phiên bản | Không sửa bừa PyTorch/TF-C; kiểm tra `python -m pip check` và chỉ điều chỉnh khi cần dùng TensorFlow |

### Nguồn tham khảo và giấy phép

- **Bài báo gốc:** [Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency (NeurIPS 2022)](https://arxiv.org/abs/2206.08496).
- **Source code tham khảo:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining).
- **Giấy phép:** Xem file [`LICENSE`](LICENSE) được giữ lại từ nguồn mã ban đầu. Repository này là bản thực hành MLOps độc lập; không phải bản phát hành chính thức của nhóm nghiên cứu.

