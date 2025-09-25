import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import io

# Título da aplicação
st.title("Ferramenta de Aprimoramento de Áudio")

# Descrição
st.markdown("Faça o upload de um arquivo de áudio para remover ruído de fundo.")
st.markdown("---")

def enhance_audio(audio_data):
    """
    Processa os dados de áudio para reduzir o ruído e retorna o áudio aprimorado.
    
    Esta é uma demonstração simplificada de aprimoramento de áudio. Para uma
    abordagem mais robusta (como a do Adobe Podcast), um modelo de IA treinado
    seria necessário.
    
    Args:
        audio_data (io.BytesIO): O objeto de arquivo de áudio carregado.
    
    Returns:
        np.ndarray: O array NumPy com o áudio aprimorado.
        int: A taxa de amostragem (sample rate) do áudio.
    """
    try:
        # Carrega o áudio a partir dos dados do arquivo
        y, sr = librosa.load(audio_data, sr=None)

        # Gera o espectrograma do áudio
        S = librosa.stft(y)
        S_magnitude, S_phase = librosa.magphase(S)

        # Aplica uma filtragem simples baseada na magnitude para demonstrar a redução de ruído.
        # Filtra frequências de baixa magnitude (que geralmente representam ruído).
        S_magnitude_filtered = np.where(S_magnitude > 0.02, S_magnitude, 0)
        
        # Reconstrói o sinal de áudio a partir do espectrograma filtrado
        y_enhanced = librosa.istft(S_magnitude_filtered * S_phase)
        
        return y_enhanced, sr
    
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o áudio: {e}")
        return None, None

# Widget para upload de arquivo
uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["wav", "mp3"])

if uploaded_file is not None:
    # Mostra um "spinner" enquanto o arquivo está sendo processado
    with st.spinner("Processando áudio... Isso pode levar alguns segundos."):
        # Processa o áudio carregado
        enhanced_audio, sr = enhance_audio(uploaded_file)

    if enhanced_audio is not None:
        st.success("Áudio processado com sucesso!")
        
        # Converte o áudio aprimorado para o formato WAV
        buffer = io.BytesIO()
        sf.write(buffer, enhanced_audio, sr, format='WAV')
        
        # Adiciona o player de áudio para pré-visualização
        st.subheader("Pré-visualização do Áudio Aprimorado")
        st.audio(buffer.getvalue(), format='audio/wav')
        
        # Adiciona um botão de download para o arquivo aprimorado
        st.download_button(
            label="Baixar Áudio Aprimorado",
            data=buffer.getvalue(),
            file_name="audio_aprimorado.wav",
            mime="audio/wav"
        )

# Instruções de uso
st.markdown("---")
st.subheader("Como usar este aplicativo:")
st.markdown("1. Salve este código como `app.py`.")
st.markdown("2. Certifique-se de que o **Streamlit** e as bibliotecas necessárias estão instalados: `pip install streamlit librosa soundfile numpy`.")
st.markdown("3. Execute o aplicativo a partir do seu terminal: `streamlit run app.py`.")
st.markdown("4. Carregue um arquivo `.wav` ou `.mp3` na interface do navegador.")
