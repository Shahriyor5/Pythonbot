import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

TOKEN = '5018620219:AAG_DD2sXRyyVvJvLZ4UHqVlCzG3pIxz2Lk'  # O'z tokeningni bu yerga qo'y

def start(update: Update, context: CallbackContext):
    update.message.reply_text('Salom! Menga audio fayl yubor, men uning spectrogramini chizib beraman.')

def get_peaks(S_db, threshold_db=-30):
    neighborhood_size = (20, 20)
    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db
    peaks = (S_db > threshold_db) & local_max
    return np.argwhere(peaks)

def handle_audio(update: Update, context: CallbackContext):
    audio_file = update.message.audio or update.message.voice or update.message.document

    if not audio_file:
        update.message.reply_text("Iltimos, audio fayl yuboring.")
        return

    file = context.bot.getFile(audio_file.file_id)
    file_path = 'input_audio.ogg'
    file.download(file_path)

    # Yuklash va ishlov berish
    y, sr = librosa.load(file_path, sr=None)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    peaks = get_peaks(S_db)

    # Rasmga olish
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.scatter(peaks[:, 1], peaks[:, 0], marker='x', color='red')
    plt.title('Spectrogram with Peaks')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('spectrogram.png')
    plt.close()

    # Yuborish
    with open('spectrogram.png', 'rb') as photo:
        update.message.reply_photo(photo)

    # Tozalash
    os.remove(file_path)
    os.remove('spectrogram.png')

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.audio | Filters.voice | Filters.document.audio, handle_audio))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
