import cv2
import numpy as np
import os

def calc_skew(im, precision=0.1):
    """Calculate and return the image skew in degrees. Image must be BW, grayscale,
    or BGR (OpenCV). Optional "precision" parameter specifies the precision in
    degrees - smaller values will result in a longer computation time."""
    
    # Define FFT size - larger values give accuracy at cost of computation time.
    fftSize = 1024
    
    # If image array has 3 dimensions, assume RGB image and convert to gray
    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    elif im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        pass
    else:
        raise ValueError('Input image must be in BW, Grayscale, or BGR (OpenCV) format!')
    
    # Make the image square, padding with the mean pixel value
    (h, w) = im.shape[:2]
    h_pad = abs(h - max([h,w]))
    w_pad = abs(w - max([h,w]))
    top = int(np.floor(h_pad/2))
    bottom = int(np.ceil(h_pad/2))
    left = int(np.floor(w_pad/2))
    right = int(np.ceil(w_pad/2))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=np.float64(im.mean()))
    
    # resize to fftSize
    im = cv2.resize(im, (fftSize, fftSize))
    s = np.fft.fft2(im)
    # take the 2D FFT
    T = np.fft.fftshift(s)
    T = np.log(1+np.abs(T))
    
    # divide the transform matrix into even quadrants
    dcInd = int(np.floor(fftSize/2)) # Index of the DC component
    evenS = int(np.mod(fftSize+1, 2)) # offset for even length fft
    
    T_1 = T[evenS:(dcInd+1), evenS:(dcInd+1)]
    T_2 = T[evenS:(dcInd+1), dcInd:]
    T_3 = T[dcInd:, evenS:(dcInd+1)]
    T_4 = T[dcInd:, dcInd:]
    
    # Rotate quadrants 1, 2, 3 to measure angle off of x-axis
    T_1 = np.rot90(T_1,2)
    T_2 = np.rot90(T_2,-1)
    T_3 = np.rot90(T_3,1)
    
    # Sum the quadrants to get all contributions in one measure
    T_sum = T_1 + T_2 + T_3 + T_4
    
    # iterate over angles and sum contributions of each pixel in corresponding line
    num_angles = int(np.floor(90/precision))
    score = np.zeros((num_angles, 1))
    for theta in np.arange(0,num_angles):
        # Find pixels along the line at theta degrees from y axis. 
        # Note that y and x are flipped from function def in order to measure angle
        # off of "negative" y axis (origin in top left of image).
        y, x = pol2cart(np.arange(0, fftSize/2), np.deg2rad(theta*precision)) 
        
        line_ind = np.zeros(T_sum.shape, dtype=bool)
        line_ind[y.round().astype(int), x.round().astype(int)] = True       
        
        score[theta] = T_sum[line_ind].sum()
        
    # find skew angle as index of highest score times precision
    skew_angle = score.argmax()*precision
    
    # force angle between -45 and 45 degrees
    skew_angle = np.mod(45 + skew_angle, 90) - 45
    
    return skew_angle

    
def pol2cart(rho, phi):
    """Convert polar coordinates to Cartesian coordinates (same as Matlab)"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

    
def deskew(im, crop=True, padVal=None):
    """Deskew and return given image. Returned image is the same color type and bit
    depth as the original. Optional "crop" parameter determines if output image is
    cropped to the same dimension as the input (default) or resized to fit the
    entire rotated image. Optional "padVal" parameter specifies the color with which
    to fill in the missing pixel values on the rotated image (scalar for grayscale
    image, 3-tuple for RGB image) - default value is zero."""
    
    skew_angle = calc_skew(im)
    
    # find the dcInd pixel
    (h, w) = im.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # get the rotation matrix needed to deskew
    M = cv2.getRotationMatrix2D((cX, cY), -skew_angle, 1.0)
    
    # find output image dimensions
    if crop:
        (nW, nH) = (w, h)
    else:
        cosTheta = abs(M[0,0])
        sinTheta = abs(M[0,1])
        nW = int((h * sinTheta) + (w * cosTheta))
        nH = int((h * cosTheta) + (w * sinTheta))
        
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
    im_out = cv2.warpAffine(im, M, (nW, nH), flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=padVal)
    
    # if input image was 3D grayscale (singular third dimension), then reshape 
    # output to match
    if im.ndim == 3 and im.shape[2] == 1:
        im_out = im_out.reshape(im_out.shape + (im.shape[2],))
    
    # return the output image
    return im_out

def batch_deskew(input_dir, output_dir, crop=True, padVal=None):
    white_list_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.tif', '.tiff'}  # valid image formats
    for fname in sorted(os.listdir(input_dir)):
        _, ext = os.path.splitext(fname)
        if ext.lower() in white_list_formats:
            im = cv2.imread(os.path.join(input_dir, fname))
            im = deskew(im, crop=crop, padVal=padVal)
            cv2.imwrite(os.path.join(output_dir, fname), im)

