def sub_backgnd(red, green, blue, tuner):
    red[:][red[:] <= red_avg*tuner] = 255                
    green[:][green[:] <= green_avg*tuner] = 255
    blue[:][blue[:] <= blue_avg*tuner] = 255


# RGB to YCbCr
def rgb2ycbcr(red, green, blue):   # source - wiki
    y = 0.299*red + 0.587*green + 0.114*blue
    cb = 128 - 0.169*red - 0.331*green + 0.5*blue
    cr = 128 + 0.5*red - 0.419*green - 0.081*blue
    return y, cb, cr


# YCbCr to HSI
def ycbcr2hsi(ch1, ch2, ch3):                            # ch = channel
    inty = ch1
    hue = np.arctan(np.divide(ch3, ch2))
    sat = np.sqrt(np.square(ch3) + np.square(ch2))
    return inty, hue, sat


# average intensity (colour-based feature)
def avg_int(ipixel, image, x_coord, y_coord):
               for i in range(len(x_coord)):
                   ipixel.append(image[y_coord[i],x_coord[i]])
               mean = round(np.mean(np.array(ipixel)))
               return mean
