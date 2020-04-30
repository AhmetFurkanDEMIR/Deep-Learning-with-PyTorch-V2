![demir](https://img.shields.io/badge/PyTorch-V1.2.0-red)
![licence](https://img.shields.io/badge/demir-ai-blueviolet)
![licence](https://img.shields.io/badge/Ahmet%20Furkan-DEM%C4%B0R-blue)

# Deep Learning with PyTorch V2

* Bu projemizde MNIST veri setini kullanarak 0 dan 9 a kadar olan sayıları sınıflandıracağız. (Çoklu sınıflandırma)
* Modelimizde Evrişimli Sinir Ağlarını kullandık (ilk projemizde tamamen bağlı katmanları kullanmıştık.)
* Ekstra olarak modelimizde max pooling katmanlar bulunmaktadır.
                      
      Yığın normalleştirme
      modelin daha iyi geneleştirme yapmasını sağlar.
      eğitim süresince  verinin ortalaması ve standart sapmasının değişimlerine bakarak veriyi normalize eder. 
* Tek etiketli çoklu sınıflandırma yaptığımız için son katmanın aktivasyon fonksiyonunu softmax olarak seçtik.
