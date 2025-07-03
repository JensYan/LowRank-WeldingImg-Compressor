import numpy as np


def lora_decomposition(A, rank=1):
    """
    对矩阵A进行低秩分解，返回矩阵P和Q

    参数:
    A (numpy.ndarray): 输入矩阵，形状为(h, w)
    rank (int): 分解的秩（必须满足 1 <= rank <= min(h, w)）

    返回:
    P (numpy.ndarray): 形状为(h, rank)
    Q (numpy.ndarray): 形状为(rank, w)
    """
    # 验证输入矩阵
    assert len(A.shape) == 2, "输入必须是二维矩阵"
    h, w = A.shape
    assert 1 <= rank <= min(h, w), "无效的秩值"

    # 执行奇异值分解（SVD）
    U, s, Vh = np.linalg.svd(A, full_matrices=False)

    # 截断到指定秩
    U_trunc = U[:, :rank]  # (h, rank)
    s_trunc = s[:rank]  # (rank,)
    Vh_trunc = Vh[:rank, :]  # (rank, w)

    # 构造分解矩阵
    P = U_trunc * np.sqrt(s_trunc)  # 广播乘法
    Q = np.diag(np.sqrt(s_trunc)) @ Vh_trunc

    return P, Q

def gradient_descent_lora(A, rank=1, learning_rate=0.01, max_iter=1000, tol=1e-6, verbose=False):
    """
    使用梯度下降法进行低秩分解

    参数:
    A (numpy.ndarray): 输入矩阵 (h, w)
    rank (int): 分解的秩
    learning_rate (float): 学习率
    max_iter (int): 最大迭代次数
    tol (float): 收敛阈值（损失变化小于此值时停止）
    verbose (bool): 是否打印训练过程

    返回:
    P (numpy.ndarray): (h, rank)
    Q (numpy.ndarray): (rank, w)
    losses (list): 训练过程中的损失值记录
    """
    h, w = A.shape
    np.random.seed(42)

    # SVD初始化
    P, Q = lora_decomposition(A, rank=rank)

    losses = []

    for i in range(max_iter):
        # 计算重构矩阵和误差
        R = P @ Q
        error = A - R

        # 计算损失 (Frobenius范数平方)
        loss = np.sum(error ** 2)
        losses.append(loss)

        # 检查收敛条件
        if i > 0 and abs(losses[-1] - losses[-2]) < tol:
            if verbose:
                print(f"在迭代 {i} 收敛")
            break

        # 计算梯度
        grad_P = -2 * error @ Q.T
        grad_Q = -2 * P.T @ error

        # 更新参数
        P -= learning_rate * grad_P
        Q -= learning_rate * grad_Q

        if verbose and (i % 100 == 0 or i == max_iter - 1):
            print(f"Iter {i}: Loss = {loss:.6f}")

    return P, Q, losses