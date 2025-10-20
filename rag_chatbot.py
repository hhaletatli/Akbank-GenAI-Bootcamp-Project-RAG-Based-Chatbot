import os
from google import genai

# RAG ve Veri İşleme Kütüphaneleri
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- 0. BAŞLANGIÇ AYARLARI ---

# Gemini API Anahtarını Ortam Değişkeninden Al
try:
    # Ortam değişkeni (export GEMINI_API_KEY="...") ayarlanmış olmalı
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("HATA: GEMINI_API_KEY ortam değişkeni ayarlanmadı. Lütfen ayarlayınız.")
except Exception as e:
    print(e)
    exit() # Anahtar olmadan program çalışmaz

# --- AŞAMA I: İNDEKSLEME (BİLGİ BANKASI OLUŞTURMA) ---

def create_vector_db():
    """docs/ klasöründeki PDF'leri yükler, parçalar ve ChromaDB'ye kaydeder."""
    
    # 1. Veri Yükleme (Data Loading)
    print("1. Zooloji raporları yükleniyor...")
    loader = DirectoryLoader(
        './docs',
        glob="**/*.pdf", # docs/ klasöründeki tüm PDF'leri yükle
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Dosya Yükleme Kontrolü
    if not documents:
        print("UYARI: docs/ klasöründe PDF dosyası bulunamadı. Lütfen kontrol edin.")
        return None

    print(f"-> Başarıyla yüklenen toplam belge sayısı: {len(documents)}")
    
    # 2. Parçalama (Chunking)
    print("2. Metinler parçalara ayrılıyor...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Vektörize Etme ve Kaydetme (Embedding & Storage)
    print("3. Vektörizasyon yapılıyor ve ChromaDB oluşturuluyor...")
    
    # Gemini'ın önerdiği embedding modelini kullan
    embedding_model = GoogleGenerativeAIEmbeddings(model="text-embedding-004") 

    # Veri tabanını oluştur ve kalıcı olarak kaydet
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory="./chroma_db" 
    )
    print("İndeksleme Tamamlandı. Bilgi Bankası Hazır.")
    return vector_db

# --- AŞAMA II: ÇALIŞTIRMA (RAG SORGULAMA) ---

def setup_rag_chain(vector_db):
    """LLM ve RAG zincirini (RetrievalQA) kurar."""
    
    if vector_db is None:
        return None
        
    # Gemini LLM'i tanımlama
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3 # Daha deterministik (kesin) cevaplar için düşük sıcaklık
    )
    
    # Retriever (Geri Alım Mekanizması)
    # ChromaDB'den en alakalı 3 kanıtı çekmek için ayarla
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 

    # RAG Zincirini Kurma
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Çekilen kanıtları direkt LLM'e gönderme yöntemi
        retriever=retriever,
        return_source_documents=True # Cevabın kanıtlarını da göstermek için
    )
    return qa_chain

def ask_assistant(qa_chain, query):
    """Asistanı sorgular ve sonucu ekrana yazdırır."""
    if qa_chain is None:
        print("RAG zinciri kurulamadığı için sorgu yapılamıyor.")
        return

    print(f"\n=======================================================")
    print(f"-> SORGU: {query}")
    print(f"=======================================================")
    
    result = qa_chain.invoke(query)
    
    print("\n--- CEVAP (RAG Asistanı) ---")
    print(result['result'])
    print("-----------------------------------")
    print("--- KANITLAR (Veri Setinden Çekilen) ---")
    
    # Cevabın dayandığı kaynakları göster
    for i, doc in enumerate(result['source_documents']):
        source_name = doc.metadata.get('source', 'Bilinmiyor').split('/')[-1]
        page_num = doc.metadata.get('page', 'Bilinmiyor')
        print(f"Kanıt {i+1}: Kaynak Dosya: {source_name}, Sayfa: {page_num}")
    print("=======================================================\n")

# --- ANA PROGRAM ÇALIŞTIRMA BÖLÜMÜ ---
if __name__ == "__main__":
    
    # 1. Vektör Veri Tabanını Oluştur
    vector_db = create_vector_db()
    
    # 2. RAG Zincirini Kur
    rag_chain = setup_rag_chain(vector_db)
    
    # 3. Örnek Test Senaryolarını Çalıştır
    if rag_chain:
        print("\n--- RAG SORGULAMA BAŞLATILIYOR ---")
        
        # Test Senaryosu 1: Biyolojik Rol
        ask_assistant(rag_chain, "Türkiye'de görülen çizgili sırtlanın habitat gereksinimleri nelerdir ve ekosistemdeki görevi (rolü) hakkında ne gibi bilgiler mevcuttur?")
        
        # Test Senaryosu 2: Karşılaştırma ve Sentez
        ask_assistant(rag_chain, "Yabani at türü olan Yılkı Atları ile Kızıl Geyik arasındaki beslenme ve habitat kullanımı farkları nelerdir? Bu iki türün aynı alanda yaşaması ekosistemi nasıl etkiler?")
        
        # Test Senaryosu 3: Popülasyon Dinamikleri
        ask_assistant(rag_chain, "Karaca türünün üreme mevsimi davranışları ve popülasyon yoğunluğunu etkileyen en önemli 3 faktör, hazırlanan raporlara göre nelerdir?")
