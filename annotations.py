# The Annotation Tool enables the user to draw bouning boxes beyond the image boundary which gives unexpected coordinates. To rectify this bug, the
# following function reduces the incorrect BB coordinates to a specific value (x = 639, y = 479) within the image boundaries.

def rectify(array, incor_val):
    if (array >= incor_val).any() == True:
        find_unwanted_val = np.where(array >= incor_val)
        for i in find_unwanted_val[0]:
            if incor_val == 640:
                array[i] = 639
            elif incor_val == 480:
                array[i] = 479
            else:
                return None


def save_insct(array, channel, insects):
    for i in range(len(array)):
        cropped = channel[y[i]:y1[i], x[i]:x1[i]]
        insects.append(cropped)
        insects = np.array(insects)
        return insects


def get_centre(nbr_list, centre_pointx, centre_pointy, start, end,  colour_channel):
    x_cent_nbr = np.arange(centre_pointx - start, centre_pointx + end)      
    y_cent_nbr = np.arange(centre_pointy - start, centre_pointy + end)
    for j in y_cent_nbr:
        for i in x_cent_nbr:
            nbr_list.append(colour_channel[j,i])
