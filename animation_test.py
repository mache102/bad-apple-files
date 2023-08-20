import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Create some dummy data for the animation
num_frames = 10
fps = 2

fig = plt.figure(figsize=(10, 6.9), frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ims = []
for i in range(num_frames):
    frame = np.random.rand(5, 5)
    im = plt.imshow(frame, animated=True)
    ims.append([im])

# Animation without axes and margins

ani = animation.ArtistAnimation(fig, ims, interval=1000 / fps, blit=True, repeat_delay=1000)
plt.tight_layout()
plt.show()