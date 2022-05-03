import cv2
import numpy as np

# Get a VideoCapture object from video and store it in vс
# or simply type 0 to get input from your webcam. If you`re using an external webcam type 1
vc = cv2.VideoCapture("3.mp4")

# Read first frame
_, first_frame = vc.read()
# Scale and resize image to 600*600
resize_dim = 600
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)

# Convert to gray scale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create mask
mask = np.zeros_like(first_frame)
# Set image saturation to maximum value as we do not need it
mask[..., 1] = 255

imageindex = 0

while (vc.isOpened()):
    # Read a frame from video
    fps = vc.get(cv2.CAP_PROP_FPS)
    print(fps)

    for j in range(int(fps) - 1):
        vc.grab()

    _, next_frame = vc.read()

    # resize gray frame obtained and convert new frame format`s to gray scale
    next_frame = cv2.resize(next_frame, None, fx=scale, fy=scale)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                        poly_n=5, poly_sigma=1.1, flags=0)

    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    np.mean(magnitude)

    cv2.imshow("Dense optical flow", magnitude)

    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Open a new window and displays the output frame
    dense_flow = cv2.addWeighted(next_frame, 1, rgb, 2, 0)
    # cv2.imshow("Dense optical flow", dense_flow)
    # cv2.imshow("Dense optical flow", rgb)

    imagename = "%d.jpg"%(imageindex)
    # cv2.imwrite(imagename, rgb)
    imageindex += 1

    print(imageindex)

    # Update previous frame
    prev_gray = next_gray

    # Frame are read by intervals of 500 millisecond.
    # The programs breaks out of the while loop when the user presses the ‘q’ key
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()