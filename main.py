import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_surf_features(frame, max_kpt):
    # Khởi tạo bộ trích xuất đặc trưng SURF
    sirf = cv2.SIFT_create(nfeatures=max_kpt)
    # Tìm key points và descriptors của ảnh
    keypoints, descriptors = sirf.detectAndCompute(frame, None)
    print(descriptors.shape)
    #Nếu đặc trưng là None
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    # Nếu số lượng keypoint nhiều hơn 1000, thì lấy 1000 kpt đầu tiên
    if len(keypoints) > max_kpt:
        keypoints = keypoints[:max_kpt]
        descriptors = descriptors[:max_kpt, :]
    
    # Nếu số lượng keypoints ít hơn max_kpt, thêm các số 0 vào các descriptors
    if len(keypoints) < max_kpt:
        pad_width = ((0, max_kpt - len(keypoints)), (0, 0))
        descriptors = np.pad(descriptors, pad_width, mode='constant')

    return keypoints, descriptors

def read_video(name_video, video_path, max_kpt = 1000, frame_skip = 24):
    
    if os.path.exists(f'feature_old/{name_video}.npy'):
        return
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
   
    # np.save(f'feature_old/{name_video}.npy', np.array(all_descriptors))


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
        # filepath2 = os.path.join(filepath, filename)
        # for video_file in os.listdir(filepath):
        # video_path = os.path.join(filepath, video_file) # đường dẫn file
        print(filepath) 
        read_video(filename, rf"{filepath}")


directory_path = r'D:\Data_2023\kỳ 2 năm 4\thầy Hóa\cats'
# read_files_in_directory(directory_path) #trích xuất đặc trưng
# read_video('v1_animal.mp4',r'cats\v110_cat.mp4', max_kpt = 1000, frame_skip = 24)

#################### STAGE 2 ############################

# def resize_image(image, width, height):
#     # Resize ảnh về kích thước mới
#     resized_image = cv2.resize(image, (width, height))
#     return resized_image

def resize_image(image, width, height):
    # Lấy kích thước hiện tại của ảnh
    print(image)
    h, w = image.shape[:2]

    # Tính toán tỉ lệ và kích thước mới
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize ảnh
    resized_image = cv2.resize(image, (new_w, new_h))

    # Tạo ảnh nền với kích thước mới và màu đen
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Tính toán vị trí để đặt ảnh đã resize vào ảnh nền
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2

    # Đặt ảnh đã resize vào giữa ảnh nền
    result_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return result_image

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

def save_img_with_kpt(query_image):
    #tiền xử lý ảnh video theo chiều dài/ rộng cố định
    img_cropped = resize_image(query_image, 1280, 720)
    #trích xuất đặc trưng của ảnh đầu vào
    kpt, feature_img = extract_surf_features(img_cropped, max_kpt=1000)
    # Vẽ các keypoints lên ảnh
    frame_with_keypoints = cv2.drawKeypoints(img_cropped, kpt, None)
    # Hiển thị ảnh đã được vẽ keypoints
    cv2.imshow('Ảnh đã được vẽ keypoint', frame_with_keypoints)
    # Lưu ảnh đã vẽ keypoints ra file
    cv2.imwrite('img test/anh_da_ve_keypoints.jpg', frame_with_keypoints)

def find_similar_videos(query_image, directory, k=3):
    top_videos = []
    #tiền xử lý ảnh video theo chiều dài/ rộng cố định
    img_cropped = resize_image(query_image, 1280, 720)
    #trích xuất đặc trưng của ảnh đầu vào
    kpt, feature_img = extract_surf_features(img_cropped, max_kpt=1000)

    # print(feature_img)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        for video_file in os.listdir(filepath):
            video_feature_path = os.path.join(filepath, video_file)
            print(video_feature_path)
            video_features = np.load(rf"{video_feature_path}")
            # Tính toán độ tương đồng giữa ảnh đầu vào và mỗi frame của 1 video
            similarities = []
            for video_feature in video_features:
                # video_feature = min_max_normalize(video_feature)
                similarity = cosine_similarity(feature_img.flatten(), video_feature.flatten())
                similarities.append(similarity)

            # lấy ra độ tương đồng lớn nhất trong 1 video
            max_similarity_each_video = np.max(similarities)
            # Thêm vào danh sách cặp (điểm số, tên tệp video)
            top_videos.append((max_similarity_each_video, video_file[:-4]))
    # Sắp xếp danh sách theo điểm số và lấy ra 3 cặp đầu tiên
    top_videos.sort(reverse=True)
    top_3_videos = top_videos[:3]
    return top_3_videos

# Đường dẫn đến các video và ảnh đầu vào
directory_feature = r'D:\Data_2023\kỳ 2 năm 4\thầy Hóa\feature_new'
query_image = cv2.imread(r"img test\animals\meo.jpg")

# Hiển thị ảnh sau khi cắt
# plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # Tắt trục
# plt.show()

#save ảnh sau khi vẽ các keypoints
save_img_with_kpt(query_image)

# Tìm ra 3 video có độ tương đồng lớn nhất với ảnh đầu vào
# similar_videos_indices = find_similar_videos(query_image, directory_feature, k = 3)
# print("video tương đồng lớn nhất:")
# print(similar_videos_indices)

