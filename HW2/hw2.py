import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(img):
    # 建立大小為256的直方圖陣列
    histogram = np.zeros(256, dtype = int)

    height, width = img.shape

    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            histogram[pixel_value] +=1
        
    return histogram

def Global_HE(img):

    # Step 1: 計算像素強度的出現次數（直方圖）
    histogram = calculate_histogram(img)

    # Step 2: 正規化直方圖，使其總和為影像中像素數量
    height, width = img.shape
    num_pixels = height * width
    cdf = np.zeros(256, dtype = float) #累積分部函數(CDF)
    cdf[0] = histogram[0] / num_pixels

    #計算累積分部函數(CDF)
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i] / num_pixels

    #Step 3: 建立對應的像素值轉換表
    equalized_lut = np.round(cdf * 255).astype(np.uint8) #將CDF映射到[0, 255]

    #Step 4: 根據轉換表對原始影像進行像素強度調整
    equalized_img = np.zeros_like(img, dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            equalized_img[i, j] = equalized_lut[img[i, j]]

    plot_combined_histogram(img, equalized_img, "Global Histogram Equalization")
    Global_Histogram_img_HPSNR = HPSNR(img, equalized_img)
    print(f"The HPSNR value of Global_Histogram_img is: {Global_Histogram_img_HPSNR}")
    return equalized_img

def Local_HE(img, window_size = 3):

    # Step 1: 初始化輸出影像與尺寸資訊
    height, width = img.shape
    locals_equlized_img = np.zeros_like(img, dtype = np.uint8)
    half_window = window_size // 2
    
    # Step 2: 對每個像素進行局部直方圖等化
    for i in range(height):
        for j in range(width):
            #定義局部範圍(考慮邊界)
            row_min = max(0 , i - half_window)
            row_max = min(height , i + half_window + 1) #要 +1 後面切片範圍才會是[i - 1, i + 1] , j同理
            col_min = max(0 , j - half_window)
            col_max = min(width, j + half_window + 1)

            #提取局部區域
            local_region = img[row_min:row_max, col_min:col_max]

            #計算局部區域的直方圖
            local_hist = calculate_histogram(local_region)

            #計算局部區域的 CDF
            num_pixels = local_region.size #.size回傳pixels數
            local_cdf = np.zeros(256 , dtype = float)
            local_cdf[0] = local_hist[0] / num_pixels
            for k in range(1, 256):
                local_cdf[k] = min(1, local_cdf[k-1] + local_hist[k] / num_pixels)

            #使用局部 CDF 建立轉換表
            local_lut = np.round(local_cdf * 255).astype(np.uint8)

            # 根據轉換表更新像素值
            locals_equlized_img[i, j] = local_lut[img[i, j]]

    plot_combined_histogram(img, locals_equlized_img, "Local Histogram Equalization")
    Local_Histogram_img_HPSNR = HPSNR(img, locals_equlized_img)
    print(f"The HPSNR value of Local_Histogram_img is: {Local_Histogram_img_HPSNR}")
    return locals_equlized_img 

def plot_combined_histogram(img, processed_img, title):
    # 將原始影像和處理後影像的直方圖疊加在同一張圖上
    plt.figure(figsize=(8, 5))

    # 原始影像的直方圖（藍色）
    plt.hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.5, label='Original')

    # 處理後影像的直方圖（橙色）
    plt.hist(processed_img.ravel(), 256, [0, 256], color='orange', alpha=0.5, label='Processed')

    plt.title(f'{title} - Combined Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')  # 顯示圖例

    plt.show()
    
def HPSNR(original_img, processed_img):
    
    if(original_img.shape != processed_img.shape):
        raise ValueError("The original and halftone images must have the same dimensions.")

    L = 255 # 最大像數值
    M, N = original_img.shape # 取得影像的高度和寬度

    # 計算MSE
    mse = np.mean((original_img - processed_img) ** 2) 
    print(f"MSE = {mse}")
    if mse == 0:
        return float('inf')
    
    hspnr = 10 * np.log10 ((L - 1) ** 2 / mse) # 計算hspnr

    return hspnr

def adaptive_histogram_equalization(img, window_size=7): #Bonus 用動差進行 Histogram Equalization
    # Step 1: 初始化輸出影像
    height, width = img.shape
    enhanced_img = np.zeros_like(img, dtype=np.uint8)
    half_window = window_size // 2

    # Step 2: 對每個像素位置進行處理
    for i in range(height):
        for j in range(width):
            # 定義局部區域的範圍，考慮邊界
            row_min = max(0, i - half_window)
            row_max = min(height, i + half_window + 1)
            col_min = max(0, j - half_window)
            col_max = min(width, j + half_window + 1)

            # 提取局部區域
            local_region = img[row_min:row_max, col_min:col_max]

            # 計算局部平均值和變異數
            local_mean = np.mean(local_region)
            local_std = np.sqrt(np.var(local_region))  # 標準差

            # Step 3: 根據局部統計量調整像素值(做標準化)
            pixel_value = img[i, j]
            if local_std > 0:  # 避免除以 0
                normalized_value = (pixel_value - local_mean) / local_std
                enhanced_value = 128 + 64 * normalized_value  # 將值映射到 [0, 255] 範圍內
            else:
                enhanced_value = local_mean

            # 將結果裁切到 [0, 255] 範圍
            enhanced_img[i, j] = np.clip(enhanced_value, 0, 255)

    plot_combined_histogram(img, enhanced_img, "Adaptive Histogram Equalization")
    Adaptive_Histogram_img_HPSNR = HPSNR(img, enhanced_img)
    print(f"The HPSNR value of Adaptive_Histogram_img is: {Adaptive_Histogram_img_HPSNR}")

    return enhanced_img


if __name__ == '__main__':

    img = cv.imread("./images/Lena.png", cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not open or find the image.")
        sys.exit(1)
    #img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

    GLHE_img = Global_HE(img) # Global Histogram Equalization
    LOHE_img = Local_HE(img) # Local Histogram Equalization
    AdaptiveHE_img = adaptive_histogram_equalization(img) #Bonus 用動差進行 Histogram Equalization

    cv.imwrite("GLHE_img.png", GLHE_img)
    cv.imwrite("LOHE_img.png", LOHE_img)
    cv.imwrite("AdaptiveHE_img.png", AdaptiveHE_img)

    cv.imshow("origin_img", img)
    cv.imshow("GLHE_img", GLHE_img)
    cv.imshow("LOHE_img", LOHE_img)
    cv.imshow("AdaptiveHE_img", AdaptiveHE_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    