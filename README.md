# Building traditional method for video recognition

Subject at PTIT: **Hệ cơ sở dữ liệu đa phương tiện (Thầy Hóa)"**

Method: [SIFT_create](https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html)

Pipline consists of feature extraction and finding similar videos

# Extract Feature

Directory_path:

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

        │ └─ natures

            │ └── v1_nature.mp4

read_files_in_directory(directory_path)

# Search video

query_image is input image

Directory_feature:

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
