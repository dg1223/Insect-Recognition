i = 0
img_list2 = glob.glob('*.jpg')                  # creates a list of all the files with the given format
img_list2 = sort(np.array(img_list2))
for z in range(shape(img_list2)[0]):
    
   ## Decode JSON file and store all coordinates in an array
   
   json_data =  open(img_list2[z][:-4])         # img_list2[z][:-4]
   data = json.load(json_data)
   i += shape(data["Image_data"]["boundingboxes"][:])[0]
print i
