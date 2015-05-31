# You need to install MoviePy to create the movie. Otherwise, it'll only create the PNG frames.
# See http://zulko.github.io/moviepy/install.html for MoviePy installation instructions.

from PIL import Image, ImageDraw
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
  frame_num = int(line_split[0])
  num_particles = int(line_split[1])
  width = float(line_split[2])
  height = float(line_split[3])

  im = Image.new("RGB", (int(width), int(height)), "white")

  draw = ImageDraw.Draw(im)

  for line in f:
    line_split = line.split(",")
    x = float(line_split[0])
    y = float(line_split[1])
    mass = float(line_split[2])

    rad = max(1, mass/200)

    min_x = x - rad
    max_x = x + rad
    min_y = y - rad
    max_y = y + rad

    min_x = min(max(min_x, 0), int(width - 1))
    max_x = min(max(max_x, 0), int(width - 1))
    min_y = min(max(min_y, 0), int(height - 1))
    max_y = min(max(max_y, 0), int(height - 1))

    #im.putpixel((int(x), int(y)), (0, 0, 0))
    draw.ellipse([(min_x, min_y), (max_x, max_y)], fill = (0,0,0), outline = (0,0,0))

  im.save(outfile)
  pngs.append((frame_num, outfile))

  print "Processed frame {0} (file: {1})".format(frame_num, infile)
pngs = sorted(pngs, key = lambda x: x[0])


#from moviepy.video.VideoClip import ImageClip
#from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.editor as mpy

clips = [mpy.ImageClip(png, duration = .1) for n, png in pngs]

#video = CompositeVideoClip(clips)
video = mpy.concatenate_videoclips(clips)
video.write_videofile("output/output.mp4", fps = 24, write_logfile = True)
#video.write_gif("output/output.gif", fps = 10)

