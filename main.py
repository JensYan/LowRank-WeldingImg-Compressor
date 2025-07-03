import numpy as np
import matplotlib.pyplot as plt
import cv2

from algorithms import lora_decomposition, gradient_descent_lora




def test_optimization_sample():
    """
    案例1：
        测试使用梯度下降法进行低秩分解的案例。
    """
    # img
    img_path = "imgs/cropped_img1.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(f"img: {img.shape[0]} x {img.shape[1]}")

    # HyParam
    rank = np.linalg.matrix_rank(img)
    print(f"Rank of img: {rank}")
    ratio = 0.05
    rank = np.int(rank * ratio)
    print(f"prune rank: {rank}")

    # 使用梯度下降进行分解
    P, Q, losses = gradient_descent_lora(
        img,
        rank=rank,
        learning_rate=0.001,
        max_iter=2000,
        tol=1e-32,
        verbose=True
    )

    output = P @ Q
    output = output.astype(np.uint8)

    print("矩阵 P:")
    print(P)
    print("\n矩阵 Q:")
    print(Q)
    print("\n重构矩阵 P×Q:")
    print(output)

    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Log loss')
    plt.grid(True)
    plt.show()

    cv2.imshow("sample 1: input", img)
    cv2.imshow("sample 1: output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output/opt_output.jpg", output)

def test_svd_sample():
    """
    测试案例2：
        测试使用SVD实现低秩分解的案例。
    """
    # img
    img_path = "imgs/cropped_img1.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(f"img: {img.shape[0]} x {img.shape[1]}")

    # HyParam
    rank = np.linalg.matrix_rank(img)
    print(f"Rank of img: {rank}")
    ratio = 0.05
    rank = np.int(rank * ratio)
    print(f"prune rank: {rank}")

    # 使用梯度下降进行分解
    P, Q = lora_decomposition(img, rank=rank)

    output = P @ Q
    output = output.astype(np.uint8)

    print("矩阵 P:")
    print(P)
    print("\n矩阵 Q:")
    print(Q)
    print("\n重构矩阵 P×Q:")
    print(output)

    cv2.imshow("sample 2: input", img)
    cv2.imshow("sample 2: output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output/svd_output.jpg", output)

def print_log(text):
    print("=" * 40)
    print(" " * 6, text)
    print("=" * 40)


# 示例用法
if __name__ == "__main__":
    print_log("测试最优化实现低秩分解")
    test_optimization_sample()
    print_log("测试SVD实现低秩分解")
    test_svd_sample()