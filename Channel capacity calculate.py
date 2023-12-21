import numpy as np
import math

epsilon = 0.001  # 误差精度
bsc = np.array([[0.3,0.7], [0.7,0.3]])      #二元对称信道
bec = np.array([[0.5,0.5,0],[0,0.5,0.5]])   #二元擦除信道
llc = np.array([[0.5,0.5,0,0,0],[0,0,0.5,0.5,0],[0,0,0,0,1]])     #无损信道
nfc = np.array([[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])   #无噪信道
lnfc = np.array([[1,0,0],[0,1,0],[0,0,1]])    #无损无噪信道
sc = np.array([[0.4,0.4,0.1,0.1],[0.1,0.1,0.4,0.4]])      #对称信道
uc = np.array([[0.4,0.2,0.2,0.2],[0.2,0.4,0.2,0.2],[0.2,0.2,0.4,0.2],[0.2,0.2,0.2,0.4]])     #均匀信道

def iter_func(matrix):                      #迭代计算函数
    r, c = matrix.shape                     #得到转移矩阵的行数和列数，行数代表输入的符号数（X），列数代表输出的符号数（Y）
    p_x = np.full((1,r), 1.0/r, dtype=float)               #根据转移矩阵的维度初始化信源，默认信源等概，r个符号则初值为1/r         P.S.  np.full第一个参数是维度，第二个参数是数值
    print("原始信源分布：",p_x)
    p_x2y = matrix
    p_y = np.dot(p_x,p_x2y)                 #根据信源概率p_x和转移矩阵matrix求得输出分布p_y
    I_l = 0         
    I_u = 10                                #对IL和IU进行初始化
    p_y2x = np.empty([r,c], dtype=float)                 #初始化后验概率矩阵
    while abs(I_u-I_l)>epsilon:          #当差值不属于误差内时开始更新迭代
        for i in range(r):
            for j in range(c):
                p_y2x[i][j] = float(p_x2y[i][j] * np.log(p_x2y[i][j] / p_y[0][j] +1e-10))        #1e-10作为参数解决矩阵中元素为0问题
        β = np.exp(p_y2x.sum(axis=1))           #求和后以e为底的指数运算求β
        l = np.sum(np.vdot(β,p_x))              #求ΣP(ai)*β(P)
        I_l = math.log2(l)
        u = max(β)                              #求I_l和I_u
        I_u = math.log2(u)
        for i in range(r):
            p_x[0][i] = p_x[0][i] * β[i] / l       #P(a)=P(a)*(β/Σ(P(a)*β(P)))
        p_y = np.dot(p_x,p_x2y)             #更新输出分布p_y
        if abs(I_u-I_l)<epsilon:
            return I_l

print("二元对称信道的转移矩阵为：",bsc)
print("信道容量为：", iter_func(bsc))

print("二元擦除信道的转移矩阵为：",bec)
print("信道容量为：", iter_func(bec))

print("无损信道的转移矩阵为：",llc)
print("信道容量为：", iter_func(llc))

print("无噪信道的转移矩阵为：",nfc)
print("信道容量为：", iter_func(nfc))

print("无损无噪信道的转移矩阵为：",lnfc)
print("信道容量为：", iter_func(lnfc))

print("对称信道的转移矩阵为：",sc)
print("信道容量为：", iter_func(sc))

print("均匀信道的转移矩阵为：",uc)
print("信道容量为：", iter_func(uc))
    