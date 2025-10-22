🦉🔍RAG Temelli Türkiye Yaban Hayatı Araştırma Asistanı
Generative AI 101 Bootcamp için hazırlanmış Türkçe RAG (Retrieval-Augmented Generation) tabanlı chatbot projesi.
📋1.Proje Hakkında
Bu proje,Türkiye'deki nesli tehlike altındaki hayvan türleri ve kritik ekosistemler hakkında sorular sorabileceğiniz bir yapay zeka asistanı oluşturur. Özel olarak toplanmış zooloji ve koruma raporlarından oluşan bilgi kümesini kullanarak, kullanıcıların sorularına ilgili metinlerden bilgi çekerek doğru ve detaylı yanıtlar verir.
2. VERİ SETİ HAZIRLAMA
Bu Python dosyası, RAG Temelli Türkiye Yaban Hayatı Araştırma Asistanı'nın ana
kodunu içerir.
Konu Alanı: Türkiye'deki nesli tehlike altındaki ve mevcut önemli hayvan türleri (Karaca, Çizgili Sırtlan, Kızıl Geyik vb.) ile ekosistem raporları.
İçerik: Veri seti, T.C. Tarım ve Orman Bakanlığı, TÜBİTAK ve akademik kurumlara ait, türlerin habitatları, ekolojik rolleri ve popülasyon dinamikleri hakkındaki detaylı metinlerden oluşturulmuştur.
Metodoloji: Veri setimiz, halka açık resmi ve akademik kaynakların (örneğin eylem planı PDF'leri) taranmasıyla **kürasyon (özel derleme)** yöntemiyle oluşturulmuş, metin içeriği kullanılarak RAG sistemine beslenmiştir. Bu sayede, devasa veri setlerini ayıklama zorunluluğu ortadan kalkmıştır.
# Gerekli kütüphaneler
import os
from langchain_community.document_loaders import PyPDFLoader
3. Çalışma Kılavuzu
## 3. Çalışma Kılavuzu (Nasıl Başlatılır?)
Bu kılavuz, projenin kaynak kodunu kendi yerel ortamınızda çalıştırmak için gerekli adımları listeler.
### Ön Gereksinimler
* Python (3.8 veya üzeri sürüm)
* Git (GitHub reposunu klonlamak için)
### Adımlar
1.  **Projeyi Klonlama:** GitHub reposunu yerel bilgisayarınıza indirin.
    ```bash
    git clone [SİZİN REPO ADRESİNİZ]
    cd rag-yabanhayati-asistani
    ```
2.  **Sanal Ortam Kurulumu:** Proje bağımlılıklarını izole etmek için bir sanal ortam oluşturun ve etkinleştirin:
    ```bash
    python -m venv venv
    # Linux/Mac için
    source venv/bin/activate
    # Windows için
    # venv\Scripts\activate
    ```
3.  **Gerekli Kütüphaneleri Yükleme:** Projenin tüm bağımlılıklarını `requirements.txt` dosyasından yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarını Ayarlama:** Gemini LLM'e erişim için API anahtarınızı (Gemini API Key) bir ortam değişkeni olarak ayarlayın. Anahtarınızı Google AI Studio'dan alabilirsiniz.
    ```bash
    # Linux/Mac için
    export GEMINI_API_KEY="ANAHTARINIZ_BURAYA_GELECEK"
    # Windows için
    # set GEMINI_API_KEY="ANAHTARINIZ_BURAYA_GELECEK"
    ```
5.  **Veri Seti Konumlandırma:** İndirilen zooloji raporlarını (PDF, TXT vb.) `docs/` klasörünün içine yerleştirdiğinizden emin olun. (Bu adım RAG sisteminin beynini oluşturur.)
6.  **Projeyi Başlatma:** Projenin Streamlit arayüzünü başlatmak için aşağıdaki komutu kullanın:
    ```bash
    streamlit run rag_chatbot_app.py
    # Not: rag_chatbot_app.py, Streamlit arayüzünüzün olduğu dosya olmalıdır.
    ```
## 4. Çözüm Mimariniz
### A. Problemin Tanımı
Projemiz, genel amaçlı Büyük Dil Modellerinin (LLM) yeterli bilgiye sahip olmadığı **Türkiye'deki yaban hayatı, koruma eylem planları ve ekosistem raporları** gibi yerel ve uzmanlık gerektiren konularda bilgiye erişim problemini çözmektedir. "RAG Temelli Türkiye Yaban Hayatı Araştırma Asistanı", araştırmacılara ve meraklılara **sadece kendi bilgi tabanımızdaki kanıtlara dayanan**, güncel ve doğru yanıtlar sunarak bu bilgi boşluğunu doldurur.
### B. Kullanılan Teknolojiler
Projemizin RAG mimarisi, aşağıdaki temel teknolojiler üzerine kurulmuştur:
1.  **Büyük Dil Modeli (LLM):** Google **Gemini API** (veya belirtilen bir Gemini modeli). Nihai cevabı üreten, soruyu anlayan akıllı motor olarak görev yapar.
2.  **RAG Çerçevesi (Framework):** **LangChain** veya **Haystack**. (Projenizin kodunda hangisini kullanacaksanız onu belirtin.) RAG akışındaki tüm adımları (yükleme, parçalama, sorgulama, cevaplama) birbirine bağlayan temel araç setidir.
3.  **Vektör Veri Tabanı (Vector Database):** **ChromaDB** veya **FAISS**. Zooloji raporlarından gelen metin parçalarını sayısal vektörler olarak depolayan ve hızlı, anlamsal arama yapılmasını sağlayan özel hafıza birimidir.
### C. RAG Çalışma Akışı (Mimarinin İşleyişi)
Sistemimiz iki ana aşamada çalışır:
| Aşama | Adım | İşlevi |
| :--- | :--- | :--- |
| **I. İndeksleme (Offline)** | **Veri Yükleme ve Vektörize Etme** | `docs/` klasöründeki zooloji raporları (PDF'ler), küçük parçalara (chunks) ayrılır. Bu parçalar bir **Embedding Modeli** ile sayı dizilerine (vektörlere) dönüştürülür ve **ChromaDB** veri tabanına kaydedilerek **bilgi bankası** oluşturulur. |
| **II. Çalıştırma (Online)** | **Geri Alım (Retrieval)** | Kullanıcı bir soru sorduğunda, bu soru da sayısal vektöre çevrilir. ChromaDB'de bu vektöre en çok benzeyen (en alakalı) metin parçaları (kanıtlar) hızla geri çekilir. |
| | **Üretim (Generation)** | Çekilen metin kanıtları, kullanıcının orijinal sorusuyla birlikte **Gemini LLM**'e gönderilir. Model, *sadece bu kanıtlara dayanarak* akıcı, özetlenmiş ve doğru nihai cevabı üretir. |
5. Web Arayüzü & Product Kılavuzu
### A. Dağıtım (Deployment) Bilgileri
Projemiz, Python tabanlı Streamlit/Gradio gibi hızlı bir arayüz çerçevesi kullanılarak geliştirilmiş ve Hugging Face Spaces gibi bir platformda yayınlanmıştır.
* **Canlı Demo Linki:** [PROJENİZİ YAYINLADIĞINIZ WEB LİNKİ BURAYA GELECEK]
    *(Not: Bu linkin, README.md dosyasının en sonunda mutlaka paylaşılması gerekmektedir.)*
### B. Çalışma Akışı ve Kullanım Kılavuzu
Kullanıcı arayüze girdiğinde, robotun temel çalışma prensibi (RAG) aşağıdaki adımları izler:
1.  **Soru Girişi:** Kullanıcı, arayüzdeki metin giriş kutusuna Türkiye yaban hayatı ve koruma alanları hakkında bir soru yazar (Örn: "Karaca popülasyonunu etkileyen temel faktörler nelerdir?").
2.  **RAG İşlemi:** Sistem, soruyu anında sayısal bir vektöre çevirir, **ChromaDB'de** depolanan zooloji raporlarından bu soruya en alakalı olan **3-5 adet metin parçasını (kanıtı)** çeker.
3.  **Cevap Üretimi:** Çekilen kanıtlar, Gemini LLM'e gönderilerek kanıtlara dayalı bir cevap oluşturması istenir.
4.  **Sonuç:** Cevap, arayüzde kullanıcıya sunulur. *(İdeal olarak, cevapla birlikte kanıt olarak kullanılan metin parçalarının kaynakları da gösterilmelidir.)*
### C. Örnek Test Senaryoları
Robotumuzun, sadece basit kelime eşleştirmesi yapmak yerine **sentez ve analiz** yeteneğini test etmek için aşağıdaki karmaşık soruları kullanabilirsiniz:
1.  **Biyolojik Rol:** "Türkiye'de görülen çizgili sırtlanın habitat gereksinimleri nelerdir ve ekosistemdeki görevi (rolü) hakkında ne gibi bilgiler mevcuttur?"
2.  **Karşılaştırma ve Sentez:** "Yabani at türü olan Yılkı Atları ile Kızıl Geyik arasındaki beslenme ve habitat kullanımı farkları nelerdir? Bu iki türün aynı alanda yaşaması ekosistemi nasıl etkiler?"
3.  **Popülasyon Dinamikleri:** "Karaca türünün üreme mevsimi davranışları ve popülasyon yoğunluğunu etkileyen en önemli 3 faktör, hazırlanan raporlara göre nelerdir?"
🛠️ Kullanılan Teknolojiler
Haystack: RAG pipeline framework
Streamlit: Web arayüzü
Sentence Transformers: Türkçe embedding modeli (trmteb/turkish-embedding-model)
Google Gemini: Text generation modeli
InMemory Document Store: Vektör veritabanı
Hugging Face Datasets: Veri seti yönetimi
🚀 Kurulum
1. Gerekli Paketleri Yükleyin
2. API Anahtarlarını Ayarlayın
3. Uygulamayı Çalıştırın
📁 Proje Yapısı
💡 Nasıl Çalışır?
Veri Yükleme: Hugging Face'ten Türkçe akademik tez veri seti indirilir
Belge İşleme: Tezler küçük parçalara bölünür
Embedding: Her parça Türkçe embedding modeli ile vektöre dönüştürülür
Vektör Veritabanı: Vektörler InMemory document store'da saklanır
Sorgulama: Kullanıcı sorusu embedding'e dönüştürülür ve en ilgili belgeler bulunur
Yanıt Üretimi: Gemini modeli, bulunan belgelerden yararlanarak yanıt oluşturur
🎯 Örnek Sorular
Türkiye'de yaşayan çizgili sırtlanın (Hyaena hyaena) habitat gereksinimleri nelerdir ve ekosistemdeki görevi (rolü) hakkında ne gibi bilgiler mevcuttur?
Karaca (Capreolus capreolus) türünün üreme mevsimi davranışları ve popülasyon yoğunluğunu etkileyen en önemli 3 faktör, hazırlanan raporlara göre nelerdir?
Türkiye'deki yabani at türü olan Yılkı Atları ile, büyük otobur (herbivor) sınıfındaki bir geyik türü (örneğin Kızıl Geyik) arasındaki beslenme ve habitat kullanımı farkları nelerdir? Bu iki türün aynı alanda yaşaması ekosistemi nasıl etkiler?
