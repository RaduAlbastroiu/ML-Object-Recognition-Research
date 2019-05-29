import os

i = 0

for file in os.scandir('CarImages'):
  if os.path.splitext(os.path.basename(file.path))[1] not in '.jpg .jpeg . JPG':
    os.remove(file)

for file in os.scandir('CarImages'):
  os.rename(file.path, os.path.join('CarImagesCurated', '{:06}.jpg'.format(i)))
  i = i + 1