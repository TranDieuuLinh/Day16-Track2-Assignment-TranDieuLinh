## Báo cáo ngắn: CPU (LightGBM) thay GPU (vLLM)

Do tài khoản AWS mới bị **hạn chế quota GPU** (dòng G/VT mặc định = 0 vCPU), không thể triển khai `g4dn.xlarge` (NVIDIA T4) để chạy LLM với vLLM. Em chuyển sang **phương án dự phòng**: train LightGBM trên CPU với dataset Credit Card Fraud (284.807 dòng, 30 features).

**Kết quả benchmark** (`benchmark_result.json`, instance `t3.micro`):

| Metric | Kết quả |
|---|---|
| Training time | **2,20 giây** (early stopping tại iteration 1) |
| AUC-ROC | **0,939** |
| Inference latency (1 row) | **0,44 ms** |
| Inference throughput (1000 rows) | **~1,2 triệu rows/giây** |

**So sánh với phương án GPU gốc:** Phương án GPU (Gemma + vLLM) không có bước training/AUC theo nghĩa ML truyền thống; cold start kéo dài 15–20 phút (tải Docker image + model weights vài GB). Inference LLM thường **hàng trăm ms đến vài giây mỗi request**, chậm hơn nhiều so với LightGBM (~0,44 ms/dòng). Ngược lại, GPU phù hợp cho **deep learning / LLM**, còn LightGBM (gradient boosting trên bảng) **tối ưu cho CPU**, không cần CUDA.

**Lý do dùng CPU:** (1) Quota GPU chưa được duyệt trên tài khoản mới/Free Tier; (2) LightGBM không yêu cầu GPU và vẫn đạt AUC cao (~0,94) với thời gian train rất ngắn; (3) Instance CPU như `r5.2xlarge` hoặc thậm chí `t3.micro` đều có sẵn ngay, không cần Deep Learning AMI hay driver NVIDIA; (4) Chi phí tương đương GPU nhưng phù hợp hơn với workload tabular ML.

**Kết luận:** Với bài toán phát hiện gian lận thẻ tín dụng, CPU + LightGBM cho **training nhanh, AUC tốt, inference cực nhanh** — đủ để hoàn thành lab (Terraform → Cloud → Training → Inference → Billing) mà không phụ thuộc quota GPU.