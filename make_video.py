# You need to install MoviePy to create the movie. Otherwise, it'll only create the PNG frames.
# See http://zulko.github.io/moviepy/install.html for MoviePy installation instructions.

from PIL import Image, ImageDraw
import os
import sys
import math


# Get command-line argument
if len(sys.argv) != 2:
  print "Please provide the data file as a single command-line argument."
  return
infile = os.path.realpath(sys.argv[1])

# Make output folder if it doesn't exist
outfolder = os.path.realpath(infile[:-4])
if not os.path.exists(outfolder):
  os.mkdir(outfolder)

print "Reading from data file {0}".format(infile)
print "Output will be in folder {0}".format(outfolder)

# Create PNGs in <outfolder> by reading the input file
f = open(infile, 'r')

# Read header line
num_particles, width, height, total_time, num_time_steps, time_steps_per_frame = f.readline().split(" ")
num_frames = math.ceil(float(num_time_steps) / time_steps_per_frame)

print "Reading data file from simulation with the following parameters:"
print "\tNumber of particles: {0}".format(num_particles)
print "\tBox width: {0}".format(width)
print "\tBox height: {0}".format(height)
print "\tTotal simulation time: {0}".format(total_time)
print "\tNumber of time steps: {0}".format(num_time_steps)
print "\tNumber of time steps per frame: {0}".format(time_steps_per_frame)
print "\tNumber of frames: {0}".format(num_frames)

pngs = [[]] * num_frames

for frames_processed in range(num_frames):
  frame_num = int(f.readline())
  outfile = os.realpath(os.join(outfolder, "frame{0}.png".format(frame_num)))

  im = Image.new("RGB", (int(width), int(height)), "white")

  draw = ImageDraw.Draw(im)

  for particle in num_particles:
    line = f.readline().split(" ")
    if len(line) != 3:
      print "WARNING: Expected a line of particle data, but got \"{0}\"".format(" ".join(line))

    x = float(line[0])
    y = float(line[1])
    mass = float(line[2])

    rad = max(1.5, sqrt(mass/math.pi)) - 0.5

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
  pngs[frame_num] = outfile

  print "Saved frame {0} in file: {1})".format(frame_num, outfile)

#from moviepy.video.VideoClip import ImageClip
#from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.editor as mpy

clips = [mpy.ImageClip(png, duration = .1) for n, png in pngs]

#video = CompositeVideoClip(clips)
video = mpy.concatenate_videoclips(clips)
video.write_videofile("output/output.mp4", fps = 24, write_logfile = True)
#video.write_gif("output/output.gif", fps = 10)

