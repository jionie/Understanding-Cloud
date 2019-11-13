from common import *


file = glob.glob('/root/share/project/kaggle/2019/cloud/data/image/train/*.jpg')
print(len(file))

for c in range(0,100,10):
    for dc in range(10):
        print(file[c+dc].split('/')[-1][:-4], ' ', end='')
    print('')

exit(0)
num_invalid=0
split = []
for f in file:
    image = cv2.imread(f)
    h,w = image.shape[:2]

    if (h,w)!=(1400,2100):
        num_invalid+=1

    print(num_invalid,f)

print(num_invalid)


