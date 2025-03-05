# Deep Learning-based RGBA Image Compression with Masked Window-based Attention
This repository contains the official implementation of the paper "Deep Learning-based RGBA Image Compression with Masked Window-based Attention."

---

### Directory Structure
Organize the repository with the following folder hierarchy:
RGBACompression
```
RGBACompression
├── Deep-Learning-based-RGBA-Image-Compression-with-Masked-Window-based-Attention
└── Kodak (evaluation datasets)
    ├── ImageSets
    ├── MaskImages
    └── PNGImages
```

---

## Usage

### Evaluating RGBA Images

To evaluate RGBA images, execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python trainRGB.py --config examples/example/config4096RGB.json -n test -pm checkpoints/JournalMask/1024/iter_600000.pth.tar -p checkpoints/JournalRGB/4096/iter_1500000.pth.tar --test
```

### Training RGBA Images

To train RGBA images, execute:

```bash
CUDA_VISIBLE_DEVICES=0 python trainRGB.py --config examples/example/config4096RGB.json -n test -pm checkpoints/JournalMask/1024/iter_600000.pth.tar
```

### Evaluating Alpha Images

To evaluate alpha images, execute:

```bash
CUDA_VISIBLE_DEVICES=0 python trainmask.py --config examples/example/config4096.json -n test -p checkpoints/JournalMask/4096/iter_600000.pth.tar --test
```

### Training Alpha Images

To train alpha images, execute:

```bash
CUDA_VISIBLE_DEVICES=0 python trainmask.py --config examples/example/config4096.json -n test
```

---

### Trained Weights

Download the trained weights from the [Google Drive](https://drive.google.com/drive/folders/11kJ1T3uwsdGtS_dW5pQybVP1SgNtCezO?usp=sharing).

