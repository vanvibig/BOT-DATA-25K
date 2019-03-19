

- Train mô hình: chỉ đến thư mục chứa hàm main -> Chạy hàm main đồng thời nhập các đối số như batch-size (bs), số lần huấn luyện epoch (n), learning rate (lr)
VD: python main.py -bs 128 -n 50 -lr 0.001

- Chat với Bot: chỉ đến thư mục chứa hàm main -> Chạy hàm main và nhập đối số mode inference (inf)
VD: python main.py -inf

- Trong thư mục data có hàm data.py là hàm dùng để xử lý để xây dựng tập từ điển, hàm này chưa tích hợp các bước tiền xử lý vào. Chỉ cần chạy hàm là sẽ ra tập từ điển