import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# dxはsigmoidの順伝播の結果
def sigmoid_derivative(dx):
    return dx * (1 - dx)

weight_1 = np.random.randn(784, 128)
weight_2 = np.random.randn(128, 1)

inputs = 0
label = 0

x_layer_1 = inputs # バッチ * 784

# 第1層目の順伝播
y_layer_1 = x_layer_1 * weight_1 # バッチ * 784 と 784 * 128 の行列積 -> バッチ * 128
y_sigmoid_layer_1 = sigmoid(y_layer_1) # バッチ * 128
x_layer_2 = y_sigmoid_layer_1

# 第2層目の順伝播
y_layer_2 = x_layer_2 * weight_2 # バッチ * 128 と 128 * 1 の行列積 -> バッチ * 1
y_sigmoid_layer_2 = sigmoid(y_layer_2) # バッチ * 1
output = y_sigmoid_layer_2

# 誤差の計算
d_error = label - output # バッチ * 1

# 第2層目の逆伝播
yd_sigmoid_layer_2 = d_error * sigmoid_derivative(y_sigmoid_layer_2) # バッチ * 1
xd_layer_2 = yd_sigmoid_layer_2.dot(weight_2.T) # バッチ * 1 と 1 * 128 の行列積 -> バッチ * 128
yd_sigmoid_layer_1 = xd_layer_2

# 第1層目の逆伝播
yd_sigmoid_layer_1 = yd_sigmoid_layer_1 * sigmoid_derivative(y_sigmoid_layer_1) # バッチ * 128
xd_layer_1 = yd_sigmoid_layer_1.dot(weight_2.T) # バッチ * 128 と 128 * 784の行列積 -> バッチ * 784

weight_2 += x_layer_2.T.dot(yd_sigmoid_layer_2) # 128 * バッチ と バッチ * 1 の行列積 -> 128 * 1
weight_1 += x_layer_1.T.dot(yd_sigmoid_layer_1) # 784 * バッチ と バッチ * 128 の行列積 -> 784 * 128