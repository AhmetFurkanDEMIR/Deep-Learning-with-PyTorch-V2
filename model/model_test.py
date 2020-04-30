import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import cv2
from torchvision import datasets
import matplotlib.pyplot as plt

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # ilk katman CNN (girdi,çıktı, filitre boyutu)
        self.conv2 = nn.Conv2d(20, 50, 5, 1) # ikini katman
        self.fc1 = nn.Linear(4*4*50, 500)
        # nn.Linear = Gelen verilere doğrusal bir dönüşüm uygular: y = x * W ^ T + b
        # in_features - her bir giriş örneğinin boyutu (yani x'in boyutu)
        # out_features - her çıktı örneğinin boyutu (yani y'nin boyutu)
        # bias (Yanlış) - False (Yanlış) olarak ayarlanırsa, katman ek bir sapma öğrenmez. Varsayılan: Doğru
        self.fc2 = nn.Linear(500, 10) # 10 adet sınıf için.

        
    def forward(self, x):
       
        x = F.relu(self.conv1(x)) # aktivasyon fok. ve ilk katman
        x = F.max_pool2d(x, 2, 2) # enbüykleri biriktirme
        x = F.relu(self.conv2(x))
        # Yığın normalleştirme
        # modelin daha iyi geneleştirme yapmasını sağlar.
        # eğitim süresince  verinin ortalaması ve standart sapmasının değişimlerine bakarak veriyi normalize eder. 
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x)) # son aktivasyon fonksiyonu
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) # çoklu sınıflandırma için gerekli aktivasyon fonksiyonu


device = torch.device("cpu:0")

_model = Net()

path2weights="agirlik.pt"

weights=torch.load(path2weights)

_model.load_state_dict(weights) 

device = torch.device("cpu:0")

path2data = "./data"

train_data = datasets.MNIST(path2data, train=True, download=True) # veri setini indirme islemi

x_train = train_data.data # sayılarımız


while True:

    try:

        a = int(input("60000 veriden rastgele bir sayı çek = "))

    except:

        quit()

    if a > 59999 or a < 0:

        quit()

    x = x_train[a]

    x1 = x_train[a]


    x= x.unsqueeze(0)
    x= x.unsqueeze(0)

    x=x.type(torch.float)

    x=x.to(device)


    output=_model(x)

    pred = output.argmax(dim=1, keepdim=True)
    z = pred.item()

    plt.imshow(x1,cmap="gray")
    plt.title("Modelimizin tahmini = {}".format(z))
    plt.show()