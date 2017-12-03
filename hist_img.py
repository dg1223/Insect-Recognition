# Create histogram (whole img)
def create_hist_img(pixels, diff_img):
    for y in range(shape(diff_img)[0]):
        for x in range(shape(diff_img)[1]):
            pixels.append(diff_img[y,x])
            hist, bins = np.histogram(pixels, bins = 256)
            width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
            centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
            return hist, bins, width, centre                # omit width and centre if unnecessary


# Create histogram (bounding box)
def create_hist_obj(hpixel, image, x_coord, y_coord):
               for i in range(len(x_coord)):
                   hpixel.append(image[y_coord[i],x_coord[i]])
               hist, bins = np.histogram(hpixel, bins = 64, range = (0.0, 255.0))
               width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
               centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
               return hist, bins, width, centre 

## ADAPTIVE THRESHOLDING (see patent for algorithm)

# Set threshold to the default value bin

def adaptive_thresh(hist, bins):
       
    # Find the total number of pixels in the image   
    N, search_thresh = 0, 0   
    for q in range(shape(hist)[0]):
        N += hist[q]

        num_pixels = 0
        req_pixels = N*0.15
        for i in range(shape(hist)[0]-1,-1,-1):
            if num_pixels < req_pixels:
                num_pixels += hist[i]
                if num_pixels > req_pixels:
                    num_pixels -= hist[i]
                    i += 1
                    break
        default_thresh = bins[i]

        # Find peak-value bin within the 1st 30
        peak_bin = np.array(np.where(np.amax(hist[0:30])))[0][0]         # PROBLEM: ALL THE IMAGES DO NOT CONFORM TO THIS CODE (POSSIBLE BUG - TRY ALL LCN IMAGES TO FIND OUT) - set exception

        # Std deviation about the peak bin (have doubts)
        std_dev = np.std(hist)
       
        # Find the minimum bin size (0.15% of the total pixels)
        min_bin = N*0.0015

        # Calculate search threshold
        thresh = hist[0] + std_dev*0.5
        num_pixels2 = 0
       
        for j in range(1, shape(hist)[0]):
            if num_pixels2 < thresh:
                num_pixels2 += hist[j-1]
                if num_pixels2 >= thresh:
                    num_pixels2 -= hist[j-1]
                    j -= 1
                    break
        dest_bin = 0
       
        for bin in range(j):
            if hist[bin+1] < min_bin:
                dest_bin = 0
                search_thresh_int = bins_int[dest_bin]
            else:
                dest_bin = bin + 1
                search_thresh = bins[dest_bin]
                break
        return N, search_thresh


