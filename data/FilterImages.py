import os

i = 0

for file in os.scandir('CarDownloadImagesLabeled/images'):
  if os.path.splitext(os.path.basename(file.path))[1] not in '.jpg .jpeg . JPG':
    os.remove(file)

for file in os.scandir('CarDownloadImagesLabeled/images'):
  os.rename(file.path, os.path.join('CarImagesCurated2', '{:06}.jpg'.format(i)))
  i = i + 1