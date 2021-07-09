# CNN_MNIST_exp

## 일반적인 CNN_MNIST 모델에서 아래의 예시와 같이 다양한 이미지에서도 동작하도록 확장
![image](https://user-images.githubusercontent.com/22045179/125031183-38251480-e0c7-11eb-8f73-12052c8dc32f.png)

## train에 사용한 데이터는 SVHN과 cv2로 기존의 MNIST 이미지를 변경하여 만들어낸 데이터를 사용

- SVHN이미지는 기본 사이즈가 32X32이므로 transforms.Resize(28)로 28X28로 변경
![image](https://user-images.githubusercontent.com/22045179/125031813-124c3f80-e0c8-11eb-84f5-7c49ed67d20e.png)

![image](https://user-images.githubusercontent.com/22045179/125031944-36a81c00-e0c8-11eb-9847-307b00c5eee0.png)
![image](https://user-images.githubusercontent.com/22045179/125032009-4a538280-e0c8-11eb-95c9-b4aeae257da3.png)

- 변경한 MNIST 이미지가 28X28X3의 형태이므로 3X28X28로 변경
![image](https://user-images.githubusercontent.com/22045179/125032138-7c64e480-e0c8-11eb-99a5-9ce8b88146e6.png)

## CNN_MODEL
- 기존의 CNN_MNIST 모델에 batch_normalization을 추가
![image](https://user-images.githubusercontent.com/22045179/125032358-d1085f80-e0c8-11eb-9e4e-b80f9518881a.png)


![image](https://user-images.githubusercontent.com/22045179/125032601-2f354280-e0c9-11eb-90fc-63ba700e529c.png)
