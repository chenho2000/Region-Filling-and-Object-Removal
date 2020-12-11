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


#Update confidence in intersection aera of sai P_hat and omega with C(P_hat)
def update_confidence(confidence, saiP_hat, P_hat):
    return confidence

