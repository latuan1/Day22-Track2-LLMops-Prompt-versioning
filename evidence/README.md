# Ghi Chú Bằng Chứng

## Phân Tích RAGAS: V1 So Với V2

Lần chạy RAGAS đã đánh giá 50 cặp hỏi đáp với `gpt-4o`, embedding model `text-embedding-3-small`, kích thước chunk `500`, overlap `50`, và truy xuất top-`3`.

Prompt V1 có kết quả tổng thể tốt hơn. V1 đạt `0.9498` ở chỉ số faithfulness và `0.9341` ở answer relevancy, vượt mục tiêu của lab là faithfulness >= `0.8`. Prompt ngắn gọn có khả năng giúp mô hình bám sát ngữ cảnh truy xuất tốt hơn và hạn chế thêm các chi tiết không được hỗ trợ bởi tài liệu.

Prompt V2 có faithfulness thấp hơn, ở mức `0.7002`, dù context recall nhỉnh hơn một chút (`0.9800` so với `0.9600`). Điều này cho thấy hướng dẫn theo kiểu gia sư có cấu trúc có thể khiến câu trả lời dài hơn hoặc diễn giải nhiều hơn, từ đó tăng nguy cơ sinh ra các khẳng định không hoàn toàn được hỗ trợ bởi context.

Context precision gần như bằng nhau cho cả hai prompt, khoảng `0.9626`, nên chất lượng truy xuất ổn định giữa hai phiên bản. Khác biệt chính đến từ cách mô hình sinh câu trả lời, không phải từ retriever.

## Tóm Tắt Điểm Số

| Chỉ số | Prompt V1 | Prompt V2 | Phiên bản tốt hơn |
|---|---:|---:|---|
| Faithfulness | 0.9498 | 0.7002 | V1 |
| Answer relevancy | 0.9341 | 0.9246 | V1 |
| Context recall | 0.9600 | 0.9800 | V2 |
| Context precision | 0.9626 | 0.9626 | Hòa |

Nhìn chung, V1 là lựa chọn phù hợp hơn cho môi trường production với knowledge base này vì tạo câu trả lời bám sát nguồn hơn, đồng thời vẫn giữ được độ liên quan và chất lượng truy xuất tốt.
