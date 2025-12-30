# model/preprocess.py
import cv2
import numpy as np
import torch
from skimage import filters, morphology, measure
from skimage.filters import frangi
from skimage.feature import local_binary_pattern

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_nclahe(img):
    clahe = cv2.createCLAHE(3.0, (8,8))
    return clahe.apply(img)

def bone_suppress(img):
    base = cv2.bilateralFilter(img, 7, 30, 30)
    detail = cv2.subtract(img, base)
    return cv2.add(base, (0.18 * detail).astype(np.uint8))

def safe_lung_mask(img):
    blur = cv2.GaussianBlur(img, (11,11), 0)
    thr = filters.threshold_otsu(255 - blur)
    mask = (255 - blur) > thr
    mask = morphology.remove_small_objects(mask, 200)
    mask = morphology.binary_closing(mask, morphology.disk(7))
    return mask.astype(np.float32)

def frequency_enhance(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    r1, r2 = 15, 60
    Y, X = np.ogrid[:rows, :cols]
    D = np.sqrt((Y-crow)**2 + (X-ccol)**2)
    mask = np.logical_and(D > r1, D < r2)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f*mask)))
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def vessel_enhance(img):
    v = frangi(img.astype(float), scale_range=(1,8))
    return cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def texture_lbp(img):
    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    return cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def make_4ch_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lung = safe_lung_mask(img)
    base = bone_suppress(apply_nclahe(img))
    roi = (base * lung).astype(np.uint8)

    freq = frequency_enhance(roi)
    vessel = vessel_enhance(roi)
    lbp = texture_lbp(roi)

    def norm(x):
        x = cv2.resize(x, (299,299))
        x = x.astype(np.float32)/255
        return (x-0.5)/0.5

    stacked = np.stack([norm(roi), norm(freq), norm(vessel), norm(lbp)])
    tensor = torch.tensor(stacked).unsqueeze(0).to(device)

    return img, lung, tensor
