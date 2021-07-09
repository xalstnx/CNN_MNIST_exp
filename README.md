# CNN_MNIST_exp

---

## 일반적인 CNN_MNIST 모델에서 아래의 예시와 같이 다양한 이미지에서도 동작하도록 확장
![image](https://user-images.githubusercontent.com/22045179/125031183-38251480-e0c7-11eb-8f73-12052c8dc32f.png)

---

## train에 사용한 데이터는 SVHN과 cv2로 기존의 MNIST 이미지를 변경하여 만들어낸 데이터를 사용

- SVHN이미지는 기본 사이즈가 32X32이므로 transforms.Resize(28)로 28X28로 변경

![image](https://user-images.githubusercontent.com/22045179/125031813-124c3f80-e0c8-11eb-84f5-7c49ed67d20e.png)

![image](https://user-images.githubusercontent.com/22045179/125031944-36a81c00-e0c8-11eb-9847-307b00c5eee0.png)
![image](https://user-images.githubusercontent.com/22045179/125032009-4a538280-e0c8-11eb-95c9-b4aeae257da3.png)

- 변경한 MNIST 이미지가 28X28X3의 형태이므로 3X28X28로 변경

![image](https://user-images.githubusercontent.com/22045179/125032138-7c64e480-e0c8-11eb-99a5-9ce8b88146e6.png)

---

## CNN_MODEL
- 기존의 CNN_MNIST 모델에 batch_normalization을 추가

![image](https://user-images.githubusercontent.com/22045179/125032358-d1085f80-e0c8-11eb-9e4e-b80f9518881a.png)


![image](https://user-images.githubusercontent.com/22045179/125032601-2f354280-e0c9-11eb-90fc-63ba700e529c.png)

---

## 성능 향상을 위한 분석 및 결과 비교
### 1. epoch 비교
가설 : 반복 학습 횟수(epoch)가 많아질수록 더 높은 정확도가 산출될 것이다.
동일 조건 : batch normalization, learning rate(1e-3), keep probability(0.8)
비교 조건 : 반복 학습 횟수(epoch)
new_model => batch norm, lr=1e-3, keep=0.8, epoch=5		Accuracy = 0.8767762184143066

new_model1 => batch norm, lr=1e-3, keep=0.8, epoch=15	Accuracy = 0.8892373442649841

new_model2 => batch norm, lr=1e-3, keep=0.8, epoch=20	Accuracy = 0.8909025192260742

new_model3 => batch norm, lr=1e-3, keep=0.8, epoch=30	Accuracy = 0.8894871473312378

new_model4 => batch norm, lr=1e-3, keep=0.8, epoch=50	Accuracy = 0.8935945630073547

new_model5 => batch norm, lr=1e-3, keep=0.8, epoch=70	Accuracy = 0.9048068523406982

new_model6 => batch norm, lr=1e-3, keep=0.8, epoch=90	Accuracy = 0.9036689400672913

new_model7 => batch norm, lr=1e-3, keep=0.8, epoch=110	Accuracy = 0.9016429781913757

![image](https://user-images.githubusercontent.com/22045179/125044640-2186b980-e0d7-11eb-8de6-2f80287c19f9.png)

![image](https://user-images.githubusercontent.com/22045179/125044646-23507d00-e0d7-11eb-9349-19431fece965.png)

결과 : epoch가 70까지는 유의미하게 테스트 Accuracy가 증가하였지만, 90부터 오히려 Accuracy가 감소함.


### 2. keep probability 비교(=dropout)
가설 : keep probability가 높을수록 더 높은 train Accuracy가 산출되지만, 너무 높을 경우 오버피팅이 발생하여 test Accuracy가 떨어질 수 있다.
동일 조건 : batch normalization, learning rate(1e-3), epoch(15)
비교 조건 : keep probability(keep)
new_model11 => batch norm, lr=1e-3, keep=0.2, epoch=15	Accuracy = 0.711839497089386

new_model10 => batch norm, lr=1e-3, keep=0.5, epoch=15	Accuracy = 0.8582371473312378

new_model1 => batch norm, lr=1e-3, keep=0.8, epoch=15	Accuracy = 0.8892373442649841

new_model12 => batch norm, lr=1e-3, keep=1.0, epoch=15	Accuracy = 0.8888210654258728

![image](https://user-images.githubusercontent.com/22045179/125044709-32372f80-e0d7-11eb-9547-7b39f6e8a553.png)

![image](https://user-images.githubusercontent.com/22045179/125044714-33685c80-e0d7-11eb-893f-d207191a1d1f.png)

결과 : keep probability가 증가할수록 test Accuracy도 따라서 증가하다가, keep probability가 너무 높을 때(0.8이상) test Accuracy가 오히려 감소함.


### 3. batch normalization 비교
가설 : model에서 Conv2d와 Relu사이에 batch normalization을 하면 더 향상된 Accuracy가 나올 것이다.
동일 조건 : learning rate(1e-3), keep probability(0.8), epoch(15)
비교 조건 : batch normalization
new_model1 => batch norm, lr=1e-3, keep=0.8, epoch=15	Accuracy = 0.8892373442649841

new_model20 => no-batch norm, lr=1e-3, keep=0.8, epoch=15	Accuracy = 0.876526415348053

![image](https://user-images.githubusercontent.com/22045179/125044764-40854b80-e0d7-11eb-9965-3f887aea8fc9.png)

![image](https://user-images.githubusercontent.com/22045179/125044768-41b67880-e0d7-11eb-8419-6b4d0d9cd2db.png)

결과 : batch normalization을 사용하였을 때가 사용하지 않았을 때보다 더 높은 test Accuracy가 나옴.


### 4. learning rate 비교
가설 : learning rate의 경우 값이 높거나 낮다고 좋은 것이 아닌 가장 적합한 값을 찾아야함.
동일 조건 : batch normalization, keep probability(0.8), epoch(5)
비교 조건 : learning rate
new_model30 => batch norm, lr=1e-1, keep=0.8, epoch=5	Accuracy = 0.1730128824710846

new_model31 => batch norm, lr=1e-2, keep=0.8, epoch=5	Accuracy = 0.7737289071083069

new_model => batch norm, lr=1e-3, keep=0.8, epoch=5		Accuracy = 0.8767762184143066

new_model33 => batch norm, lr=1e-4, keep=0.8, epoch=5	Accuracy = 0.7808059453964233

new_model34 => batch norm, lr=1e-5, keep=0.8, epoch=5	Accuracy = 0.3470248579978943

![image](https://user-images.githubusercontent.com/22045179/125044807-4b3fe080-e0d7-11eb-9eff-a9bc714401dd.png)

![image](https://user-images.githubusercontent.com/22045179/125044816-4d09a400-e0d7-11eb-97a1-907a8624fc4c.png)

결과 : training 결과 learning rate가 1e-3일 때 가장 높은 test Accuracy가 산출됨.


### 5. best model
다차례의 학습 결과, batch normalization을 사용하고, learning rate가 1e-3이고, keep probability가 0.8이고(dropout=0.2), epoch가 70인 모델이 가장 높은 test Accuracy를 산출함. 
new_model5 => batch norm, lr=1e-3, keep=0.8, epoch=70	Accuracy = 0.9048068523406982

![image](https://user-images.githubusercontent.com/22045179/125044902-67438200-e0d7-11eb-959c-8ed27d576b5d.png)
