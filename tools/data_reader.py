import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import os

class DataReader:
    image_dir_name = "images"
    seg_dir_name = "segs"
    landmark_dir_name = "landmarks"

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir.joinpath(self.image_dir_name)
        self.seg_dir = self.data_dir.joinpath(self.seg_dir_name)
        self.lms_dir = self.data_dir.joinpath(self.landmark_dir_name)
        
        # Membaca langsung dari folder makeup dan non-makeup
        self.makeup_names = [f for f in os.listdir(self.image_dir.joinpath('makeup')) 
                           if f.endswith('.png') or f.endswith('.jpg')]
        self.non_makeup_names = [f for f in os.listdir(self.image_dir.joinpath('non-makeup')) 
                                if f.endswith('.png') or f.endswith('.jpg')]

        self.random = None

    def read_file(self, name, is_makeup=True):
        # Menentukan subfolder berdasarkan tipe gambar
        subfolder = 'makeup' if is_makeup else 'non-makeup'
        
        # Membaca image
        image = Image.open(
            self.image_dir.joinpath(subfolder, name).as_posix()
        ).convert("RGB")
        
        # Membaca segmentation
        seg = np.asarray(
            Image.open(
                self.seg_dir.joinpath(subfolder, name).as_posix()
            )
        )
        
        # Membaca landmark (mengubah ekstensi ke .pkl)
        landmark_name = name.replace('.png', '.pkl').replace('.jpg', '.pkl')
        lm = pickle.load(self.lms_dir.joinpath(subfolder, landmark_name).open("rb"))

        return image, seg, lm

    def __getitem__(self, index):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        if isinstance(index, tuple):
            assert len(index) == 2
            index_non_makeup = index[1]
            index = index[0]
        else:
            assert isinstance(index, int)
            index_non_makeup = index

        return self.read_file(self.non_makeup_names[index_non_makeup], False),\
            self.read_file(self.makeup_names[index], True)

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def pick(self):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return self[a_index, another_index]

    def check_files(self):
        """Fungsi untuk mengecek keberadaan semua file"""
        print("Checking files...")
        
        # Cek file makeup
        for name in self.makeup_names:
            img_path = self.image_dir.joinpath('makeup', name)
            seg_path = self.seg_dir.joinpath('makeup', name)
            lm_path = self.lms_dir.joinpath('makeup', name.replace('.png', '.pkl').replace('.jpg', '.pkl'))
            
            if not img_path.exists():
                print(f"Missing makeup image: {img_path}")
            if not seg_path.exists():
                print(f"Missing makeup seg: {seg_path}")
            if not lm_path.exists():
                print(f"Missing makeup landmark: {lm_path}")
        
        # Cek file non-makeup
        for name in self.non_makeup_names:
            img_path = self.image_dir.joinpath('non-makeup', name)
            seg_path = self.seg_dir.joinpath('non-makeup', name)
            lm_path = self.lms_dir.joinpath('non-makeup', name.replace('.png', '.pkl').replace('.jpg', '.pkl'))
            
            if not img_path.exists():
                print(f"Missing non-makeup image: {img_path}")
            if not seg_path.exists():
                print(f"Missing non-makeup seg: {seg_path}")
            if not lm_path.exists():
                print(f"Missing non-makeup landmark: {lm_path}")