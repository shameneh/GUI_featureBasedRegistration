import numpy as np
import cv2


def reg_measure(moving_image, template_image):
    # create a mask of the template
    _, mask = cv2.threshold(template_image, 0, 255, cv2.THRESH_OTSU)
    # threshold the moving image
    _, mov_im_thresh = cv2.threshold(moving_image, 0, 255, cv2.THRESH_OTSU)
    # calculate the normalized cross-correlation of the masked pixels in both images
    r = cv2.matchTemplate(mov_im_thresh, mask, cv2.TM_CCORR_NORMED, mask=mask)
    return r[0][0]


def apply_registration(image, homography):
    (height, width) = image.shape[:2]
    return cv2.warpPerspective(image, homography, (width, height))


def adjust_homography(homography):
    down = np.matrix([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
    up = np.matrix([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]])
    return np.matmul(np.matmul(up, homography), down)


# separating this function from the registration methods so that this can be run immediately when the templates are loaded (and save the results for later)
def get_keypoints(in_image, downsample_factor=2, method="AKAZE"):
    downsample_ratio = pow(2, downsample_factor)
    im1 = cv2.resize(in_image, (in_image.shape[1] // downsample_ratio,
                               in_image.shape[0] // downsample_ratio))
    if (method=="AKAZE"):
        detector = cv2.AKAZE_create()
    elif (method=="ORB"):
        detector = cv2.ORB_create()
    elif (method=="SIFT"):
        detector = cv2.xfeatures2d.SIFT_create()
    elif (method=="SURF"):
        detector = cv2.xfeatures2d.SURF_create(900, 5, 5)
    else: # AKAZE by default
        detector = cv2.AKAZE_create()
    (kp1, des1) = detector.detectAndCompute(im1, None)


    return([kp1, des1,im1])

def get_keypoints_withkey(in_image,key, downsample_factor=2, method="AKAZI"):
    downsample_ratio = pow(2, downsample_factor)
    im1 = cv2.resize(in_image, (in_image.shape[1] // downsample_ratio,
                               in_image.shape[0] // downsample_ratio))
    if (method=="AKAZE"):
        detector = cv2.AKAZE_create()
    elif (method=="ORB"):
        detector = cv2.ORB_create()
    elif (method=="SIFT"):
        detector = cv2.xfeatures2d.SIFT_create()
    elif (method=="SURF"):
        detector = cv2.xfeatures2d.SURF_create(900, 5, 5)
    else: # AKAZE by default
        detector = cv2.AKAZE_create()
    (kp1, des1) = detector.compute(im1, key)


    return([kp1, des1,im1])


def registration_ORB(query_image, template_image, downsample_factor=0, kp2=None, des2=None):
    '''
    query_image: The name of the query image.
    template_image: The name of the template image.
    downsample_factor: A factor used for down-sampling the images. The
    downsample rate is calculated by 1/(2**downsample_factor)
    ReturnValue: The image registration result or None if an error happened.
    '''
    # if not os.path.isfile(query_image):
    #     print("The query image doesn't exist. Please check the file name.")
    #     return
    # if not os.path.isfile(template_image):
    #     print("The template image doesn't exist. Please check the file name.")
    #     return
    if not type(downsample_factor) is int:
        print("The downsample_factor is not an integer!")
        return
    # img1 = cv2.imread(query_image, 0) # query image
    # img2 = cv2.imread(template_image, 0) # template image
    im1 = query_image
    im2 = template_image
    original_image = im1

    # use get_keypoints method (and only run on template if keypoints weren't passed in)
    (kp1, des1) = get_keypoints(in_image=im1, downsample_factor=downsample_factor, method="ORB")
    if ((kp2 is None) or (des2 is None)):
        (kp2, des2) = get_keypoints(in_image=im2, downsample_factor=downsample_factor, method="SIFT")

    # create BFMatcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf_matcher.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x:x.distance)
    # Get matching points
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)  
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt    
    # Find homography
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Adjust homography
    height, width = original_image.shape
    for i in range(downsample_factor):
        homography = adjust_homography(homography)
    # Apply homography to the original image
    return homography



def registration_SIFT(key,query_image, template_image, min_match_count=4, downsample_factor=2, kp2=None, des2=None,kp1 = None, des1 = None):
    '''
    query_image: The name of the query image.
    template_image: The name of the template image.
    min_match_count: The minimum number of matches accepted.
    downsample_factor: A factor used for down-sampling the images. The
    downsample rate is calculated by 1/(2**downsample_factor)
    !!!!!! The down-samplefactor has to bee greater than 0 for SIFT to work.
    ReturnValue: The image registration result or None if an error happened.
    '''
    #if not os.path.isfile(query_image):
        #print("The query image doesn't exist. Please check the file name.")
        #return
    #if not os.path.isfile(template_image):
        #print("The template image doesn't exist. Please check the file name.")
        #return
    if not type(downsample_factor) is int:
        print("The downsample_factor is not an integer!")
        return
    #img1 = cv2.imread(query_image, 0) # query image
    #img2 = cv2.imread(template_image, 0) # template image
    im1 = query_image
    im2 = template_image    
    original_image = im1
    if key==0:
    # use get_keypoints method (and only run on template if keypoints weren't passed in)
        if ((kp1 is None) or (des1 is None)):
            (kp1, des1,immov) = get_keypoints(in_image=im1, downsample_factor=downsample_factor, method="SIFT")
        else:
            downsample_ratio = pow(2, downsample_factor)
            immov = cv2.resize(im1, (query_image.shape[1] // downsample_ratio,
                                     query_image.shape[0] // downsample_ratio))
        if ((kp2 is None) or (des2 is None)):
            (kp2, des2,im) = get_keypoints(in_image=im2, downsample_factor=downsample_factor, method="SIFT")
    else:
        #if ((kp1 is None) or (des1 is None)):
        (kp1, des1, immov) = get_keypoints_withkey(in_image=im1, key=kp1,downsample_factor=downsample_factor, method="SIFT")
        #else:
            # downsample_ratio = pow(2, downsample_factor)
            # immov = cv2.resize(im1, (query_image.shape[1] // downsample_ratio,
            #                          query_image.shape[0] // downsample_ratio))
        #if ((kp2 is None) or (des2 is None)):
        (kp2, des2, im) = get_keypoints_withkey(in_image=im2,key=kp2, downsample_factor=downsample_factor, method="SIFT")

    # use get_keypoints method (and only run on template if keypoints weren't passed in)
    # (kp1, des1) = get_keypoints(in_image=im1, downsample_factor=downsample_factor, method="SIFT")
    # if ((kp2 is None) or (des2 is None)):
    #     (kp2, des2) = get_keypoints(in_image=im2, downsample_factor=downsample_factor, method="SIFT")
    #
    # initialize the brute force matcher
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, False)
    # match the descriptors
    matches = bf_matcher.knnMatch(des1,trainDescriptors=des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    src = []  # kp1[matches.queryIdx][:, ::-1]
    dst = []  # kp2[matches.trainIdx][:, ::-1]
    for i, match in enumerate(good):
        src.append(kp1[match.queryIdx])
        dst.append(kp2[match.trainIdx])
    if len(good) > min_match_count:
        points1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        points2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
        # compute the homography
        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)    
        height, width = original_image.shape
        # adjust the homography is needed
        for i in range(downsample_factor):
            homography = adjust_homography(homography)
        # apply the homography to the original image and return the result.
        return homography,immov,src,dst,kp1,des1
    else:
        print("Not enough matches found!")
        return


def registration_SURF(query_image, template_image, min_match_count=4, downsample_factor=2, kp2=None, des2=None):
    '''
    query_image: The name of the query image.
    template_image: The name of the template image.
    min_match_count: The minimum number of matches accepted.
    downsample_factor: A factor used for down-sampling the images. The
    downsample rate is calculated by 1/(2**downsample_factor)
    ReturnValue: The image registration result or None if an error happened.
    '''
    # if not os.path.isfile(query_image):
    #     print("The query image doesn't exist. Please check the file name.")
    #     return
    # if not os.path.isfile(template_image):
    #     print("The template image doesn't exist. Please check the file name.")
    #     return
    if not type(downsample_factor) is int:
        print("The downsample_factor is not an integer!")
        return
    # img1 = cv2.imread(query_image, 0) # query image
    # img2 = cv2.imread(template_image, 0) # template image
    im1 = query_image
    im2 = template_image
    original_image = im1

    # use get_keypoints method (and only run on template if keypoints weren't passed in)
    (kp1, des1) = get_keypoints(in_image=im1, downsample_factor=downsample_factor, method="SURF")
    if ((kp2 is None) or (des2 is None)):
        (kp2, des2) = get_keypoints(in_image=im2, downsample_factor=downsample_factor, method="SURF")
    
    # create BFMatcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, False)
    # match the descriptors
    matches = bf_matcher.knnMatch(des1, trainDescriptors=des2, k=2)
    # ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    if len(good) > min_match_count:
        points1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        points2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
        # compute the homography
        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)    
        height, width = original_image.shape
        # adjust the homography
        for i in range(downsample_factor):
            homography = adjust_homography(homography)
        # apply the homography to the original image and return the result
        return homography
    else:
        print("Not enough matches found!")
        return


def registration_AKAZE(key,query_image, template_image, downsample_factor=2, kp2=None, des2=None,kp1=None,des1=None):
    im1 = query_image
    im2 = template_image
    original_image = im1
    if key==0:
    # use get_keypoints method (and only run on template if keypoints weren't passed in)
        if ((kp1 is None) or (des1 is None)):
            (kp1, des1,immov) = get_keypoints(in_image=im1, downsample_factor=downsample_factor, method="AKAZE")
        else:
            downsample_ratio = pow(2, downsample_factor)
            immov = cv2.resize(im1, (query_image.shape[1] // downsample_ratio,
                                     query_image.shape[0] // downsample_ratio))
        if ((kp2 is None) or (des2 is None)):
            (kp2, des2,im) = get_keypoints(in_image=im2, downsample_factor=downsample_factor, method="AKAZE")
    else:
        #if ((kp1 is None) or (des1 is None)):
        (kp1, des1, immov) = get_keypoints_withkey(in_image=im1, key=kp1,downsample_factor=downsample_factor, method="AKAZE")
        #else:
            # downsample_ratio = pow(2, downsample_factor)
            # immov = cv2.resize(im1, (query_image.shape[1] // downsample_ratio,
            #                          query_image.shape[0] // downsample_ratio))
        #if ((kp2 is None) or (des2 is None)):
        (kp2, des2, im) = get_keypoints_withkey(in_image=im2,key=kp2, downsample_factor=downsample_factor, method="AKAZE")

    # create BFMatcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf_matcher.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x:x.distance)
    # Get matching points
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    src = []#kp1[matches.queryIdx][:, ::-1]
    dst = []#kp2[matches.trainIdx][:, ::-1]
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
        src.append( kp1[match.queryIdx])
        dst.append(kp2[match.trainIdx])
    # Find homography
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Adjust homography
    (height, width) = original_image.shape
    for i in range(downsample_factor):
        homography = adjust_homography(homography)
    # Apply homography to the original image
    return homography,immov,src,dst,kp1,des1
