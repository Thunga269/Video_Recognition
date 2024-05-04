import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_surf_features(frame, max_kpt):
    # Khởi tạo bộ trích xuất đặc trưng SURF
    sirf = cv2.SIFT_create()

    # Tìm key points và descriptors của ảnh
    keypoints, descriptors = sirf.detectAndCompute(frame, None)
    #Nếu đặc trưng là None
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
        
    # Giảm số lượng keypoints nếu vượt quá ngưỡng
    if len(keypoints) > max_kpt:
        # Sắp xếp các keypoints theo độ quan trọng (response)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_kpt]

        # Trích xuất descriptors của các keypoints đã chọn
        selected_descriptors = []
        for keypoint in keypoints:
            selected_descriptors.append(descriptors[keypoints.index(keypoint)])

        descriptors = np.array(selected_descriptors) #[the number of keypoints, 128]
    
    # Nếu số lượng keypoints ít hơn max_kpt, thêm các số 0 vào các descriptors
    if len(keypoints) < max_kpt:
        pad_width = ((0, max_kpt - len(keypoints)), (0, 0))
        descriptors = np.pad(descriptors, pad_width, mode='constant')

    return keypoints, descriptors

def read_video(name_video, video_path, max_kpt = 1000, frame_skip = 24):
    
    # if os.path.exists(f'features/{name_video}.npy'):
    #     return
    # Mở video
    cap = cv2.VideoCapture(video_path)
    # Đọc chiều dài và chiều rộng của video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f'Chiều cao: {height}, chiều rộng {width} ')
    
    all_descriptors = []
    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip == 0:
            # Trích xuất đặc trưng từ mỗi frame
            keypoints, descriptors = extract_surf_features(frame, max_kpt)
            # print(descriptors.shape)
            # Hiển thị số lượng key points tìm thấy trong mỗi frame
            # print(f"Số lượng key points trong frame: {len(keypoints)}")
            all_descriptors.append(descriptors)
            # Hiển thị frame với key points
            frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)
            cv2.imshow('Frame', frame_with_keypoints)

        # Đợi 25ms, bấm phím 'q' để thoát
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(frame_count)
   
    np.save(f'features/{name_video}.npy', np.array(all_descriptors))


def read_files_in_directory(directory):
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(directory):
        print("Thư mục không tồn tại.")
        return
    
    # Kiểm tra xem 'directory' là một thư mục không
    if not os.path.isdir(directory):
        print(f"{directory} không phải là một thư mục.")
        return
    
    # Lặp qua tất cả các tệp trong thư mục và in ra tên của chúng
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        filepath2 = os.path.join(filepath, filename)
        for video_file in os.listdir(filepath2):
            video_path = os.path.join(filepath2, video_file) # đường dẫn file
            print(video_path) 
            read_video(filename+'/'+video_file, rf"{video_path}")


# directory_path = r'D:\Data_2023\kỳ 2 năm 4\thầy Hóa\dataset'
# read_files_in_directory(directory_path) #trích xuất đặc trưng
# read_video('v22_animal.mp4',r'dataset\flowers\flowers\v22_flower.mp4', max_kpt = 1000, frame_skip = 24)

#################### STAGE 2 ############################

def resize_image(image, width, height):
    # Resize ảnh về kích thước mới
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def cosine_similarity(A, B): 
    '''
    Input: 2 vector A, B có cùng chiều
    output: độ tương đồng của 2 vector
    '''
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def find_similar_videos(query_image, directory, k=3):
    top_videos = []
    #tiền xử lý video, cắt video theo chiều dài/ rộng cố định
    img_cropped = resize_image(query_image, 1920, 1080)
    #trích xuất đặc trưng của ảnh đầu vào
    kpt, feature_img = extract_surf_features(img_cropped, max_kpt=1000)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        for video_file in os.listdir(filepath):
            video_feature_path = os.path.join(filepath, video_file)
            print(video_feature_path)
            video_features = np.load(rf"{video_feature_path}")
            # Tính toán độ tương đồng giữa ảnh đầu vào và mỗi frame của 1 video
            similarities = []
            for video_feature in video_features:
                similarity = cosine_similarity(feature_img.flatten(), video_feature.flatten())
                similarities.append(similarity)

            # lấy ra độ tương đồng lớn nhất trong 1 video
            max_similarity_each_video = max(similarities)
            # Thêm vào danh sách cặp (điểm số, tên tệp video)
            top_videos.append((max_similarity_each_video, video_file[:-4]))
    # Sắp xếp danh sách theo điểm số và lấy ra 3 cặp đầu tiên
    top_videos.sort(reverse=True)
    top_3_videos = top_videos[:3]
    return top_3_videos

# Đường dẫn đến các video và ảnh đầu vào
directory_feature = r'D:\Data_2023\kỳ 2 năm 4\thầy Hóa\features'
query_image = cv2.imread(r'img test\1.jpg')

# Hiển thị ảnh sau khi cắt
# plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # Tắt trục
# plt.show()

# Tìm ra 3 video có độ tương đồng lớn nhất với ảnh đầu vào
similar_videos_indices = find_similar_videos(query_image, directory_feature, k = 3)
print("video tương đồng lớn nhất:")
print(similar_videos_indices)