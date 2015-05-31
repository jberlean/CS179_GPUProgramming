from PIL import Image
import os

for filename in os.listdir("output"):
  filename = "output/{0}".format(filename)
  if not filename.endswith(".dat"):
  	print "Skipping {0}. It doesn't appear to be a data file".format(filename)
  	continue

  f = open(filename, 'r')

  line_split = f.readline().split(",")
  num_particles = int(line_split[0])
  width = float(line_split[1])
  height = float(line_split[2])

  im = Image.new("RGB", (int(width), int(height)), "white")

  for line in f:
    line_split = line.split(",")
    x = int(float(line_split[0]))
    y = int(float(line_split[1]))
    mass = float(line_split[2])

    im.putpixel((x, y), (0, 0, 0))

    im.save("{0}.png".format(filename))