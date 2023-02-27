import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# seed 고정
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# print(x_train.shape)  #torch.Size([3, 1])
# print(y_train.shape)  #torch.Size([3, 1])

# 가중치 W: 0으로 초기화 & 학습을 통해 값 변경되는 변수
W = torch.zeros(1, requires_grad=True)
#print(W)  #tensor([0.], requies_grad=True)
# 편향 b: 초기화 & 값 변경되는 변수
b = torch.zeros(1, requires_grad=True)
#print(b)  #tensor([0.], requies_grad=True)

# 경사 하강법 SGD
optimizer = optim.SGD([W, b], lr=0.01) #lr: 학습률(learning rate)

nb_epochs = 1999 #원하는만큼 경사하강법 반복
for epoch in range(nb_epochs + 1):

    # 가설 세우기 H(x) = Wx + b
    hypothesis = x_train * W + b

    # 비용 함수(Cost func): 오차 제곱의 평균(MSE)
    cost = torch.mean((hypothesis - y_train) ** 2)

    #cost로 H(x) 개선
    optimizer.zero_grad()  # gradient(기울기) 0으로 초기화
    cost.backward()  # 비용함수 미분하여 gradient 계산
    # W, b 업데이트: 인수 W, b에서 return 되는 변수들의 gradient에 lr 곱하여 뺌
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{}: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs,
                                                    W.item(), b.item(), cost.item()))

