1. **Trích rút đặc trưng** 
directory_path = r'D:\thầy Hóa\dataset' là đường dẫn folder video dạng sau:

dataset

├── animals

│ └── animals

│ └── v1_animal.mp4

├── flowers

│ └── flowers

│ └── v1_flower.mp4

├── foods

│ └── foods

│ └── v1_food.mp4

├── humans

│ └── humans

│ └── v1_human.mp4

└── natures

└── natures

└── v1_nature.mp4

read_files_in_directory(directory_path)

2. **Tìm kiếm video**
query_image là ảnh đầu vào

directory_feature = r'D:\thầy Hóa\features'  là đường dẫn folder video dạng sau:

features

├── animals

│ └── v1_animal.mp4.npy

├── flowers

│ └── v1_flower.mp4.npy

├── foods

│ └── v1_food.mp4.npy

├── humans

│ └── v1_human.mp4.npy

└── natures

└── v1_fnature.mp4.npy

similar_videos_indices = find_similar_videos(query_image, directory_feature, k = 3)