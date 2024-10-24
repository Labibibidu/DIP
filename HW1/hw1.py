import sys
import cv2 as cv
import numpy as np
import matplotlib as plt

def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0, 2], [3, 1]])
    else:
        smaller_matrix = generate_bayer_matrix(n - 1)
        size = 2 ** n
        new_matrix = np.zeros((size, size), dtype=int)
        for i in range(2 ** (n - 1)):
            for j in range(2 ** (n - 1)):
                base_value = 4 * smaller_matrix[i, j]
                new_matrix[i, j] = base_value
                new_matrix[i, j + 2 ** (n - 1)] = base_value + 2
                new_matrix[i + 2 ** (n - 1), j] = base_value + 3
                new_matrix[i + 2 ** (n - 1), j + 2 ** (n - 1)] = base_value + 1
        
        return new_matrix

def generate_thresholds_matrix(bayer_matrix):
    N = bayer_matrix.shape[0] #(bayer_matrix.shape (4,4) , N is the size of the dither matrix)
    thresholds_matrix = np.zeros_like(bayer_matrix, int)
    # TODO:Calculate each bayer matrix element threshold
    
    for i in range(N):
        for j in range(N):
            thresholds_matrix[i, j] = (255 * (bayer_matrix[i, j] + 0.5 )) / N ** 2

    #print(thresholds_matrix)
    return thresholds_matrix

def Ordered_Dithering(img, thresholds_matrix):
    # TODO:Implementing the ordered dithering algorithm
    N = thresholds_matrix.shape[0] # N=4
    Ordered_Dithering_img = np.zeros_like(img , np.uint8) # 建立與輸入影像大小相同的空白陣列
    
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            threhold = thresholds_matrix[i % N][j % N]  # 找到對應的 Bayer matrix 位置
    
            #比較pixels value與threshold
            if img[i, j] > threhold:
                Ordered_Dithering_img[i, j] = 255
            else:
                Ordered_Dithering_img[i, j] = 0


    return Ordered_Dithering_img

def Error_Diffusion(img):
    # TODO:Implementing the error diffusion algorithm
    img_copy = img.astype(np.int32).copy() #複製原影像，避免更改到原始影像 **要用深拷貝 若用img_copy = img仍會改到同一張影像

    Error_Diffusion_img = np.zeros_like(img_copy, np.uint8) #建立與輸入影像大小相同的空白陣列
    diffusion_kernal = np.array([
        [0, 0, 7/16],
        [3/16, 5/16, 1/16]
        ], dtype = float) #定義 Floyd-Steinberg 的誤差擴散權重

    height, width = img_copy.shape
    
    for i in range(height):
        for j in range(width):
            #將像數二值化 (threshold為128)
            old_pixel = img_copy[i, j]
            if old_pixel > 128:  
                new_pixel = 255
            else:
                new_pixel = 0
            Error_Diffusion_img[i, j] = new_pixel
        
            error = old_pixel - new_pixel #計算誤差

            #擴散誤差給鄰近像數
            for di in range(2):
                for dj in range(3):
                    if i + di < height and j + dj -1 < width and j + dj -1 >= 0: #檢查下邊界、右邊界、左邊界
                        img_copy[i + di, j + dj -1] += error * diffusion_kernal[di , dj]

    return Error_Diffusion_img

def calculate_mse(original_img, halftone_img):

    return np.mean((original_img - halftone_img) ** 2)  #計算兩張影像間的 MSE (均方誤差)

def toggle_pixel(halftone_img, x, y):

    halftone_img[x, y] = 255 if halftone_img[x, y] == 0 else 0 # 翻轉 (toggle) 某個像素的值 (0 <-> 255)

def DBS(img, iteration = 10):

    height, width = img.shape

    DBS_img = np.random.choice([0,255], size = (height, width)).astype(np.uint8) # 初始化二值影像 (隨機初始化或全 0 / 全 255)

    neighborhood_size = 3  # 使用 3x3 的區域進行 MSE 計算

    for it in range(iteration):
        #print(f"DBS Iteration {it +1}")
        
        #儲存原始影像的MSE
        for i in range(height):
            for j in range(width):
                #檢查邊界
                origin_patch = DBS_img[
                    max(0 , i - 1) : min(height, i + 2), 
                    max(0, j - 1) : min(width, j + 2),
                ]
                original_mse = calculate_mse(img[max(0, i - 1) : min(height, i + 2), 
                                                max(0, j - 1) : min(width, j + 2)],
                                             origin_patch)

                #翻轉當前像素的值，並重新計算 MSE
                toggle_pixel(DBS_img, i, j)
                new_patch = DBS_img[
                    max(0, i - 1) : min(height, i + 2),
                    max(0, j - 1) : min(width, j + 2),
                ]
                new_mse = calculate_mse(img[max(0, i - 1) : min(height, i + 2), 
                                            max(0, j - 1) : min(width, j + 2)],
                                         new_patch)

                # 如果新的 MSE 較差，就還原翻轉
                if new_mse >= original_mse:
                    toggle_pixel(DBS_img, i, j)

    return DBS_img

def generate_class_matrix(size=8):

    #生成 Knuth 的 Class Matrix (8x8)
    return np.array([
        [34, 48, 40, 32, 29, 15, 23, 31],
        [42, 58, 56, 53, 21, 5, 7, 10],
        [50, 62, 61, 45, 13, 1, 2, 18],
        [38, 46, 54, 37, 25, 17, 9, 26],
        [28, 14, 22, 30, 35, 19, 51, 33],
        [20,  4,  6, 11, 43, 49, 47, 39],
        [24, 16,  8, 3, 59, 55, 57, 52],
        [44, 36, 41, 27, 63, 60, 50, 12]
    ])

def process_block(img, output_img, start_i, start_j, class_matrix, diffusion_kernel):

    #處理 8x8 區塊，根據 Class Matrix 進行像素二值化與擴散
    block_size = class_matrix.shape[0]

    # 對每個 class 值進行處理
    for class_value in range(block_size ** 2):
        for i in range(block_size):
            for j in range(block_size):
                # 確認該像素是否符合當前的 class value
                if class_matrix[i, j] == class_value:
                    # 取得當前像素位置
                    pixel_i = start_i + i
                    pixel_j = start_j + j

                    # 檢查是否超出影像邊界
                    if pixel_i >= img.shape[0] or pixel_j >= img.shape[1]:
                        continue

                    # 二值化處理 (threshold 為 128)
                    old_pixel = img[pixel_i, pixel_j]
                    new_pixel = 255 if old_pixel > 128 else 0
                    output_img[pixel_i, pixel_j] = new_pixel

                    # 計算誤差
                    error = old_pixel - new_pixel

                    # 擴散誤差給鄰近像素
                    for di in range(2):
                        for dj in range(3):
                            ni, nj = pixel_i + di, pixel_j + dj - 1
                            if ni < img.shape[0] and 0 <= nj < img.shape[1]:
                                img[ni, nj] += error * diffusion_kernel[di, dj]


def dot_diffusion(img):
    img_copy = img.astype(np.int32)  # **將影像從 uint8 轉為 int32 型態以負數計算溢位
    #Dot Diffusion 二值化與誤差擴散演算法
    height, width = img.shape

    # 生成 Class Matrix 和擴散權重矩陣
    class_matrix = generate_class_matrix()
    diffusion_kernel = np.array([
        [0, 0, 7/16],
        [3/16, 5/16, 1/16]
    ], dtype=float)

    dot_diffusion_img = np.zeros_like(img, np.uint8)  # 初始化輸出影像

    # 將影像分塊，使用 8x8 的 Class Matrix
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            process_block(img_copy, dot_diffusion_img, i, j, class_matrix, diffusion_kernel)
    return dot_diffusion_img

def HPSNR(original_img, halftone_img):
    
    if(original_img.shape != halftone_img.shape):
        raise ValueError("The original and halftone images must have the same dimensions.")

    L = 255 # 最大像數值
    M, N = original_img.shape # 取得影像的高度和寬度

    # 計算MSE
    mse = np.mean((original_img - halftone_img) ** 2) 
    print(f"MSE = {mse}")
    if mse == 0:
        return float('inf')
    
    hspnr = 10 * np.log10 ((L - 1) ** 2 / mse) # 計算hspnr

    return hspnr


if __name__ == '__main__':
    
    img = cv.imread("Baboon-image.png", cv.IMREAD_GRAYSCALE) # 要轉灰階
    # img = cv.imread(sys.argv[1])
    
    if img is None:
        print("Error: Could not open or find the image.")
        sys.exit(1)

    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    #print(bayer_matrix)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)

    Ordered_Dithering_img = Ordered_Dithering(img, thresholds_matrix)
    Ordered_Dithering_img_HPSNR = HPSNR(img, Ordered_Dithering_img)
    print(f"The HPSNR value of Ordered_Dithering_img is: {Ordered_Dithering_img_HPSNR}")

    Error_Diffusion_img = Error_Diffusion(img)
    Error_Diffusion_img_HPSNR = HPSNR(img, Error_Diffusion_img)
    print(f"The HPSNR value of Error_Diffusion_img is: {Error_Diffusion_img_HPSNR}")    

    DBS_img = DBS(img , iteration = 5)
    DBS_img_HPSNR = HPSNR(img, DBS_img)
    print(f"The HPSNR value of DBS_img_ is: {DBS_img_HPSNR}")  

    dot_diffusion_img = dot_diffusion(img)
    dot_diffusion_img_HPSNR = HPSNR(img, dot_diffusion_img)
    print(f"The HPSNR value of dot_diffusion_img is: {dot_diffusion_img_HPSNR}")  

    #存取檔案
    #cv.imwrite(f"Ordered_Dithering_n={n}_img.png",Ordered_Dithering_img)
    #cv.imwrite("Error_Diffusion_img.png",Error_Diffusion_img)
    #cv.imwrite("DBS_img.png" , DBS_img)
    #cv.imwrite("dot_diffusion_img.png" , dot_diffusion_img)
    #print(f"succeessfully save")

    # TODO:Show your picture
    #顯示原始照片
    cv.imshow(f"origin_img",img)

    #顯示經過Ordered_Dithering的照片
    cv.imshow(f"Ordered_Dithering_n={n}img", Ordered_Dithering_img)

    #顯示經過error_Diffusion的照片
    cv.imshow("Error_Diffusion_img", Error_Diffusion_img)

    #顯示經過DBS的照片
    cv.imshow("DBS_img", DBS_img)

    #顯示經過dot_diffusion的照片
    cv.imshow("dot_diffusion_img", dot_diffusion_img)

    cv.waitKey(0)
    cv.destroyAllWindows

