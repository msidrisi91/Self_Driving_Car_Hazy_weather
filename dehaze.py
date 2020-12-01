import numpy as np
import cv2

def Guidedfilter(Image ,p ,r ,Epsilon):
    I_mean = cv2.boxFilter(Image, cv2.CV_64F, (r, r));
    p_mean = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    Ip_mean = cv2.boxFilter(Image * p, cv2.CV_64F, (r, r));
    Ip_cov = Ip_mean - I_mean * p_mean;

    II_mean = cv2.boxFilter(Image * Image, cv2.CV_64F, (r, r));
    I_var = II_mean - I_mean * I_mean;

    a = Ip_cov / (I_var + Epsilon);
    b = p_mean - a * I_mean;

    a_mean = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    b_mean = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = a_mean * Image + b_mean;
    return q;

    
def DarkChannel(Image, Size):
    b, g, r = cv2.split(Image)
    Min_Channel = cv2.min(cv2.min(r, g), b);
    Window = cv2.getStructuringElement(cv2.MORPH_RECT, (Size, Size))
    DC = cv2.erode(Min_Channel, Window)
    return DC   

def TransmissionRefine(Image, et):
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;    # initially 60
    eps = 0.0001; # initially 0.0001
    t = Guidedfilter(gray, et, r, eps);

    return t;

def dehaze(img):
    src = img
    I = src.astype(np.float64)/255;
    sz = 9

    dc = cv2.min(cv2.min(I[:, :, 0], I[:, :, 1]), I[:, :, 2])
    b, g, r = cv2.split(I)
    Im = ((b + g + r)/3)
    dark = Im + (Im.mean() - dc.mean())
    A = dark
    kernel = np.ones((sz, sz), np.float32) / (sz * sz)
    A = cv2.filter2D(A, -1, kernel)
    A = TransmissionRefine(src, A);
    A = cv2.min(A, 0.8)
    temp = np.zeros(I.shape, I.dtype)
    temp[:, :, 0] = (I[:, :, 0] / A)
    temp[:, :, 1] = (I[:, :, 1] / A)
    temp[:, :, 2] = (I[:, :, 2] / A)
    dc = cv2.min(cv2.min(temp[:, :, 0], temp[:, :, 1]), temp[:, :, 2])
    mean = (temp[:, :, 0] + temp[:, :, 1] + temp[:, :, 2]) / 3
    meanI = (b + g + r) / 3
    dark = cv2.min(cv2.min(I[:, :, 0], I[:, :, 1]), I[:, :, 2])
    beta = meanI - dark

    t = (1 - 0.95 * (dc)) / (1 - beta)
    t = cv2.max(t, 0.1)
    J = np.zeros(I.shape, I.dtype)
    for ind in range(0, 3):
        J[:, :, ind] = (I[:, :, ind] - A) / (t) + (A)

    return J * 255