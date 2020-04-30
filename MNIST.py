from torchvision import datasets # PyTorch içindeki tüm verisetlerine erişim sağlamak
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torchvision import transforms

from torch.utils.data import DataLoader

from torch import nn

import torch.nn.functional as F

import os

from torchsummary import summary

from torch import optim

import torch

path2data = "./data" # veri setinin indirileceği konum

# eğitim seti
train_data = datasets.MNIST(path2data, train=True, download=True) # veri setini indirme islemi

x_train = train_data.data # sayılarımız

y_train = train_data.targets # etiketleri

print(x_train.shape) # 60000 adet sayı 28*28 boyutunda

print(y_train.shape) # 60000 etiket

# test seti
val_data = datasets.MNIST(path2data, train=False, download=True)

x_val = val_data.data # sayılar

y_val = val_data.targets # etiket

print(x_val.shape) # 10000 adet 28*28 veri

print(y_val.shape) # 10000 adet etiket

# verilere kanal değeri ekliyoruz.
if len(x_train.shape)==3:
    x_train=x_train.unsqueeze(1)

print("Eğitim verisetinin son hali",x_train.shape)

if len(x_val.shape)==3:
    x_val=x_val.unsqueeze(1)

print("etiketlerin son hali",x_val.shape)

# veri görselleştirme
def show(img):

    npimg = img.numpy()

    npimg_tr=np.transpose(npimg, (1,2,0))
    
    plt.imshow(npimg_tr,interpolation='nearest')
    
    plt.show()
    
# 40 resim
x_grid=utils.make_grid(x_train[:40], nrow=8, padding=2)
print(x_grid.shape)
# verileri görselleştirme
show(x_grid)

# bazı görüntü dönüşümleri uygulamak için sınıf tanımladık
# veri kümemizi büyülttük.
# Yatay ve dikey döndürme olasılığı, döndürmeyi zorlamak için p = 1 olarak ayarlandı
data_transform = transforms.Compose([
transforms.RandomHorizontalFlip(p=1),
transforms.RandomVerticalFlip(p=1),
transforms.ToTensor(),
])

# Dönüşümleri MNIST veri kümesindeki bir görüntüye uygulayalım

# örnek resim
img = train_data[0][0]


# dönüşümü uyguluyoruz 
img_tr=data_transform(img)

# tensörü numpy dizisine dönüştür
img_tr_np=img_tr.numpy()

print(img_tr.shape)

# orjinal ve dönüşüm uygulanmış resim
plt.subplot(1,2,1)
plt.imshow(img,cmap="gray")
plt.title("original")
plt.subplot(1,2,2)
plt.imshow(img_tr_np[0],cmap="gray");
plt.title("transformed")
plt.show()


#  tensörleri veri kümesine aktarmak
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
for x,y in train_ds:
    print(x.shape,y.item())
    break

"""
Egzersiz sırasında verileri kolayca tekrarlamak için aşağıdakileri kullanarak bir veri yükleyici oluşturabiliriz:
DataLoader sınıfı aşağıdaki gibidir:
"""

# 8 lik yığınlarla verileri oluşturduk
# böylece belleği fazla zorlamayız.
train_dl = DataLoader(train_ds, batch_size=8)
val_dl = DataLoader(val_ds, batch_size=8)

os.system("clear")

# model oluşturma

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

"""

Model =  cpu  da eğitilecek
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 20, 24, 24]             520
            Conv2d-2             [-1, 50, 8, 8]          25,050
            Linear-3                  [-1, 500]         400,500
            Linear-4                   [-1, 10]           5,010
================================================================
Total params: 431,080
Trainable params: 431,080
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.12
Params size (MB): 1.64
Estimated Total Size (MB): 1.76


"""

model = Net()

print(model) # katmanlar

device = torch.device("cpu:0")

print("Model = ",next(model.parameters()).device," da eğitilecek")

summary(model, input_size=(1,28,28)) # modelin parametreleri

# kayıp fonksiyonu
# ağımızın veri seti üzerinde kendi performansını ölçmesinin sağlar.

loss_func = nn.NLLLoss(reduction="sum")

# eniyileme:
# ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde
# bulundurarak kendisini güncelleme mekanizması
opt = optim.Adam(model.parameters(), lr=1e-4)

"""
Ardından, mini parti başına doğruluğu hesaplamak için bir yardımcı işlev tanımlayacağız
"""
def metrics_batch(target, output):
    # çıktı sınıfını al
    pred = output.argmax(dim=1, keepdim=True)
    
    # çıktı sınıfını hedef sınıfla karşılaştır
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

"""
Mini seri başına kayıp değerini hesaplamak için bir yardımcı fonksiyon geliştirelim, 8 lik yığın
"""
def loss_batch(loss_func, xb, yb,yb_h, opt=None):
    
    # kayıp elde etmek
    loss = loss_func(yb_h, yb)
    
    # performans ölçümü elde etmek
    metric_b = metrics_batch(yb,yb_h)
    
    if opt is not None:
        loss.backward() # model parametrelerine göre yeniden hesaplama
        opt.step() # # model parametrelerini güncelleme
        opt.zero_grad() # degradeleri sıfıra ayarlama

    return loss.item(), metric_b

"""
Ardından, bir kayıp ve metrik değerlerini hesaplamak için bir yardımcı işlev tanımlayacağız
"""
def loss_epoch(model,loss_func,dataset_dl,opt=None):
    
    loss=0.0
    metric=0.0
    len_data=len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb=xb.type(torch.float).to(device)
        yb=yb.to(device)
        
        # model çıktısı elde et
        yb_h=model(xb)

        loss_b,metric_b=loss_batch(loss_func, xb, yb,yb_h, opt)
        loss+=loss_b
        if metric_b is not None:
            metric+=metric_b
    loss/=len_data # bir döngüdeki ortalama hata
    metric/=len_data
    return loss, metric


# eğitim döngüsü
def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    
    for epoch in range(epochs):
        # eğitime başlandığını bildirir
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)
        
        # doğrulama verileri üzerindeki eğitimi bildirir.    
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)
        
        accuracy=100*val_metric

        print("epoch: %d, train loss: %.6f, val loss: %.6f, accuracy: %.2f" %(epoch, train_loss,val_loss,accuracy))


# fonksiyonu çağırma işlemi
num_epochs=5
train_val(num_epochs, model, loss_func, opt, train_dl, val_dl)

# modelin kaydedileceği konum ve adı
path2weights="./model/agirlik.pt"
# modeli kaydetmek.
torch.save(model.state_dict(), path2weights)
