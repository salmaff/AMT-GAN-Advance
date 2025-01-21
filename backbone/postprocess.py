#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image

#Memproses hasil akhir gambar setelah melewati jaringan atau model neural network.
class PostProcess:
    def __init__(self, config):
        #Konfigurasi ini biasanya diberikan oleh pengguna saat program dijalankan, 
        #dan menyimpan informasi tentang apakah gambar perlu dihaluskan (denoise) serta ukuran gambar.
        self.denoise = config.POSTPROCESS.WILL_DENOISE
        self.img_size = config.DATA.IMG_SIZE

    def __call__(self, source: Image, result: Image):
        # TODO: Refract -> name, resize
        # Mengubah gambar input (source) dan hasil (result) menjadi array NumPy. 
        # Ini diperlukan karena operasi seperti resize dan perhitungan perbedaan citra dilakukan dengan NumPy dan OpenCV, 
        # yang bekerja dengan array numerik.
        source = np.array(source)
        result = np.array(result)

        #Mengambil dimensi asli gambar source, yaitu tinggi dan lebar.
        height, width = source.shape[:2]
        #Mengubah ukuran gambar source menjadi ukuran yang lebih kecil sesuai dengan konfigurasi.
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        #Gambar yang telah di-resize (small_source) diubah ukurannya kembali ke ukuran asli (width, height).
        #Perbedaan laplacian dihitung antara gambar asli (source) dan gambar yang telah di-resize kembali.
        #Ini dilakukan untuk memulihkan detail halus pada gambar hasil setelah proses neural network, 
        #yang mungkin hilang akibat perubahan resolusi atau pemrosesan lainnya.
        laplacian_diff = source.astype(
            np.float64) - cv2.resize(small_source, (width, height)).astype(np.float64)
        result = (cv2.resize(result, (width, height)) +
                  laplacian_diff).round().clip(0, 255).astype(np.uint8)
        #mengurangi noise pada gambar hasil tanpa merusak detailnya.
        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        result = Image.fromarray(result).convert('RGB')
        return result
