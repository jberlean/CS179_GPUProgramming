# You need to install MoviePy to create the movie. Otherwise, it'll only create the PNG frames.
# See http://zulko.github.io/moviepy/install.html for MoviePy installation instructions.

from PIL import Image
import os

pngs = []

for filename in os.listdir("output"):
  infile = "output/{0}".format(filename)
  outfile = "{0}.png".format(infile)
  if not infile.endswith(".dat"):
  	print "Skipping {0}. It doesn't appear to be a data file".format(filename)
  	continue

  f = open(infile, 'r')

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

  im.save(outfile)
  pngs.append(outfile)

#from moviepy.video.VideoClip import ImageClip
#from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.editor as mpy

clips = [mpy.ImageClip(png, duration = 1) for png in pngs]

#video = CompositeVideoClip(clips)
video = mpy.concatenate_clips(clips)
#video.write_videofile("output/output.mp4", fps = 24, write_logfile = True)
#video.write_gif("output/output.gif", fps = 2)

