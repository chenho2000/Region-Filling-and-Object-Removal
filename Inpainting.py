# Get source region and target region and init confidence
def load_image(image1, image2):
    image = cv2.imread(image1)
    image = cv2.cvtcolor(image, cv2.COLOR_BGR23GRAY)
    mask = cv2.imread(image2)
    mask = cv2.cvtcolor(mask, cv2.COLOR_BGR23GRAY)
    confidence = (mask == 255)
    return image, mask, confidence


# Use edge detection to get front fill Delta Omega from mask image
def get_front_fill(mask):
    return delta_omega


# Use point axis and window_size to get the range of patch psi P(2X2 array)
def patch(point, window_size):
    return psiP


# compute point p confidence
def compute_confidence(confidence, point, mask, window_size):
    psiP = patch(mask, point, window_size)

    return confidence


# Get the gradient of isophote around the patch
def compute_gradient(image, point):
    return nalba_Ip


# Get the unit vector of the normal on front fill
def compute_normal(point, delta_omega):
    return np


# Compute data term by equation
def compute_data(nalba_Ip, np):
    alpha = 255

    return data


# Compute priority for point P
def compute_priority(confidence, data):
    return priority


# Compute two points p and q' Euclidean distance square difference
def distance_difference(p, q):
    return distance


# Compute the sum of squared differences (SSD) of the already filled pixels in
# the two patches psiP and psiQ
def SSD(psiP, psiQ):
    return ssd


# Find the best match patch psi q hat
def best_match_patch(distance, ssd, window_size):
    q_hat = np.argmin(distance + ssd)
    psiQ_hat = patch(q_hat, window_size)
    return psiQ_hat


# Fill the empty pixels in sai P with the value of corresponding pixels in sai q hat.
def fill_image(image, mask, psiP, psiQ_hat):
    px = psiP[0][0]
    py = psiP[0][1]
    qx = psiQ_hat[0][0]
    qy = psiQ_hat[0][1]
    for i in range(window_size):
        for j in range(window_size):
            px += i
            py += j
            qx += i
            qy += j

    return image, mask

#Update confidence in intersection aera of sai P_hat and omega with C(P_hat)
def update_confidence(confidence, saiP_hat, P_hat):
    return confidence





if __name__ == "__main__":

    #main loop for image inpainting
