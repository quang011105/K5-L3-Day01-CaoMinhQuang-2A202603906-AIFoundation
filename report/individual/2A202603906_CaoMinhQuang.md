# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603906
- **Họ và tên:** Cao Minh Quang
- **SEED** (in ở ô đầu notebook):3969

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✓☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✓☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✓☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.974   0.792      0.874        0.788   
2. BoW + MLP (deep learning)         0.978   0.917      0.946        0.919   
3a. Encoder (MiniLM) + kNN           0.826   0.990      0.900        0.976   
3b. Encoder (MiniLM) + LogReg        0.949   0.979      0.964        1.000   
3c. Encoder (MiniLM) + MLP head      0.950   1.000      0.975        1.000   
4a. LLM nhỏ zero-shot                0.864   0.950      0.905        0.905   
4c. LLM nhỏ few-shot                 0.905   0.950      0.927        0.927   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.588      14194.466       0.077   
2. BoW + MLP (deep learning)        10.621      23676.905       0.304   
3a. Encoder (MiniLM) + kNN         107.414         21.836      45.087   
3b. Encoder (MiniLM) + LogReg      107.924         31.494      26.320   
3c. Encoder (MiniLM) + MLP head    108.398         30.340      26.464   
4a. LLM nhỏ zero-shot                0.000          1.712     522.293   
4c. LLM nhỏ few-shot                 0.000          1.608     550.083   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.286         0.000  
2. BoW + MLP (deep learning)          0.384         0.000  
3a. Encoder (MiniLM) + kNN           70.629         0.000  
3b. Encoder (MiniLM) + LogReg        36.060         0.000  
3c. Encoder (MiniLM) + MLP head      36.740         0.000  
4a. LLM nhỏ zero-shot               878.087         0.022  
4c. LLM nhỏ few-shot                966.403         0.060  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. Dự đoán F1 của mô hình Machine Learning
- Dự đoán: F1 khoảng 0.9.
- Kết quả thực tế: F1 = 0.874 với mô hình Bag-of-Words + Logistic Regression.
- Vì sao lệch: Mình đã dự đoán hơi cao vì nghĩ Logistic Regression có thể phân loại spam tốt. Tuy nhiên, mô hình bỏ sót một số tin spam, khiến Recall chỉ đạt 0.792 và kéo F1 xuống
2. Dự đoán mô hình không sử dụng ReLU
- Dự đoán: Bỏ ReLU sẽ làm mất khả năng học đặc tính phi tuyến, khiến mô hình giống Linear Regression và F1 giảm.
- Kết quả thực tế: Mô hình không ReLU đạt TEST F1 = 0.941.
- Vì sao lệch: Dự đoán về việc F1 giảm là đúng hướng, nhưng mình chưa đánh giá được mức độ giảm. Mô hình vẫn đạt F1 khá cao vì cấu trúc mạng và dữ liệu vẫn cho phép học các đặc trưng hữu ích. 
3. Dự đoán mô hình nào phân loại tin nhắn ngụy trang tốt hơn
- Dự đoán: Encoder + Logistic Regression (3b) sẽ phân loại đúng nhiều tin hơn.
- Kết quả thực tế: Encoder + Logistic Regression đạt F1 = 0.964, cao hơn Bag-of-Words + Logistic Regression (F1 = 0.874).
- Vì sao lệch: Dự đoán này không lệch nhiều về kết quả tổng thể. Tuy nhiên, mình chưa dự đoán được mức chênh lệch lớn như thực tế. Embedding giúp mô hình hiểu ngữ nghĩa và ngữ cảnh tốt hơn, thay vì chỉ dựa vào sự xuất hiện của từ khóa.

## 4. Phân tích 50 case khó

Trong quá trình thực nghiệm trên 50 case khó, mình gặp vấn đề về thời gian phản hồi khi sử dụng Gemini API. Do các yêu cầu phân loại phải gửi đến API bên ngoài và thời gian xử lý kéo dài, quá trình thực nghiệm không thể hoàn thành trong thời gian dự kiến. Vì vậy, mình chủ động dừng chạy để tránh tiêu tốn thêm thời gian và tài nguyên API.

Do chưa thu được đầy đủ kết quả trên 50 case, mình chưa thể đưa ra kết luận định lượng chính xác về model tụt hiệu năng nhiều nhất hoặc nhóm case có tỷ lệ phân loại sai cao nhất.

Tuy nhiên, dựa trên đặc điểm của bài toán phân loại spam, các nhóm case có thể gây khó khăn cho mô hình bao gồm tin nhắn spam ngụy trang, tin nhắn chứa từ khóa spam nhưng thực chất không phải spam, và các trường hợp spam sử dụng ngôn ngữ hoặc ngữ cảnh khác biệt so với dữ liệu huấn luyện.

Đây là những nhận định mang tính phân tích, chưa phải kết quả được xác nhận từ thực nghiệm 50 case.
## 5. Thử thách cá nhân

Mình đã thực hiện phần thiết lập SEED cá nhân ngay từ đầu notebook. SEED được tạo dựa trên mã sinh viên thông qua hàm băm MD5, sau đó sử dụng để cố định tính ngẫu nhiên trong Python và NumPy.

Việc sử dụng SEED giúp các kết quả thực nghiệm có tính nhất quán và thuận tiện hơn khi kiểm tra, so sánh giữa các lần chạy. Tuy nhiên, mình chưa hoàn thành đầy đủ phần thử thách cá nhân theo yêu cầu, đặc biệt là việc tổng hợp riêng cách thực hiện, kết quả thực nghiệm và phân tích sự khác biệt do SEED tạo ra.

Nguyên nhân là mình tập trung vào các thí nghiệm so sánh mô hình và gặp vấn đề về thời gian thực thi khi sử dụng Gemini API. Vì vậy, phần thử thách cá nhân chưa được kiểm chứng và trình bày đầy đủ như dự kiến.

## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý 10 triệu SMS/ngày và p95 latency dưới 50 ms/tin, mình lựa chọn TF-IDF + Logistic Regression làm phương án triển khai chính.

Cơ sở lựa chọn:

Trong thực nghiệm, Bag-of-Words + Logistic Regression đạt F1 = 0.874.

Encoder + Logistic Regression đạt F1 = 0.964, cho thấy biểu diễn ngữ nghĩa giúp cải thiện chất lượng phân loại.

Tuy nhiên, yêu cầu p95 < 50 ms/tin khiến tốc độ suy luận và khả năng mở rộng trở thành tiêu chí quan trọng, không chỉ riêng F1.

LLM API không phải lựa chọn ưu tiên cho luồng xử lý chính do độ trễ phụ thuộc vào mạng và thời gian phản hồi từ dịch vụ bên ngoài.

Đánh đổi chấp nhận:

Mình chấp nhận giảm một phần khả năng phát hiện spam phức tạp để đổi lấy tốc độ xử lý, chi phí thấp và khả năng triển khai ổn định. Các tin nhắn khó phân loại có thể được chuyển sang mô hình mạnh hơn hoặc kiểm tra bổ sung.

Đây là lựa chọn dựa trên sự cân bằng giữa hiệu năng phân loại và yêu cầu vận hành. Tuy nhiên, F1 và latency thực tế của TF-IDF + Logistic Regression cần được benchmark trên môi trường triển khai để xác nhận SLA p95 < 50 ms.

## 7. AI log và điều đã học

### 7.1. Nhật ký sử dụng AI

| Nội dung hỏi AI | Kết quả nhận được | Phần mình tự sửa / điều chỉnh |
|---|---|---|
| SEED là gì và có tác dụng gì trong notebook? | AI giải thích SEED dùng để kiểm soát yếu tố ngẫu nhiên, giúp kết quả thực nghiệm có thể tái lập. | Mình liên hệ với đoạn code tạo SEED từ mã sinh viên và hiểu rõ hơn vai trò của `random.seed()` và `np.random.seed()`. |
| Một số cú pháp Python/NumPy mình quên trong quá trình làm notebook, chẳng hạn cách sử dụng `np.random.seed()`, `random.seed()` và các thao tác xử lý mảng. | AI giải thích cú pháp, ý nghĩa tham số và đưa ra ví dụ minh họa. | Mình đối chiếu với code trong notebook, tự điều chỉnh cú pháp cho phù hợp với biến và dữ liệu đang sử dụng, thay vì sao chép trực tiếp. |
| Hãy hỗ trợ phân tích 50 case khó trong bài toán phân loại SMS spam. Với mỗi model, hãy xác định nhóm case dễ sai, giải thích cơ chế dự đoán sai và đề xuất cách trình bày kết quả. | AI đề xuất cách phân tích theo nhóm lỗi, phân biệt false positive và false negative, đồng thời gợi ý các nguyên nhân như ngữ cảnh, từ khóa gây nhiễu và khả năng nhận diện spam ngụy trang. | Mình điều chỉnh nội dung theo cấu trúc bài tập và kết quả thực tế. Do Gemini API chạy quá lâu nên mình đã dừng thực nghiệm, không sử dụng các nhận định do AI đưa ra như kết quả đã được kiểm chứng. |

### 7.2. Điều mình hiểu ra hôm nay

Trước đây, mình thường chỉ sử dụng AI để tìm câu trả lời hoặc giải thích những phần chưa hiểu, nhưng hôm nay mình nhận ra AI còn có thể hỗ trợ xây dựng hướng phân tích và cách trình bày một thí nghiệm. Tuy nhiên, câu trả lời của AI chỉ nên được xem là giả thuyết hoặc gợi ý, không thể thay thế kết quả thực nghiệm. Mình cũng hiểu rằng cần tự kiểm tra cú pháp, đối chiếu kết quả và chịu trách nhiệm với những nội dung cuối cùng đưa vào notebook.
