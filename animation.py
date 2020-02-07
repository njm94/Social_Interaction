# Animating processed brain imaging data

import matplotlib.animation as Animation
import matplotlib.pyplot as plt
from numpy import round


def animate_processed_frames(left, right, metadata, filename, title, true_framerate,figsize=(20, 10), cmap='viridis'):
    FFMpegWriter = Animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=30, metadata=metadata)
    fig = plt.figure(figsize=figsize)

    l_ax1 = fig.add_subplot(2, 5, 1)
    l_title1 = 'Left Mouse Red'
    l_red = left[l_title1]
    l_ax1.set_title(l_title1)
    l_img1 = l_ax1.imshow(
        l_red[0],
        cmap=cmap,
        vmin=-0.2,
        vmax=0.2
    )
    l_cbar1 = fig.colorbar(l_img1)

    l_ax2 = fig.add_subplot(2, 5, 2)
    l_title2 = 'Left Mouse Green'
    l_green = left[l_title2]
    l_ax2.set_title(l_title2)
    l_img2 = l_ax2.imshow(
        l_green[0],
        cmap=cmap,
        vmin=-0.05,
        vmax=0.2
    )
    l_cbar2 = fig.colorbar(l_img2)

    l_ax3 = fig.add_subplot(2, 5, 3)
    l_title3 = 'Left Mouse Blue'
    l_blue = left[l_title3]
    l_ax3.set_title(l_title3)
    l_img3 = l_ax3.imshow(
        l_blue[0],
        cmap=cmap,
        vmin=-0.02,
        vmax=0.05
    )
    l_cbar3 = fig.colorbar(l_img3)

    l_ax4 = fig.add_subplot(2, 5, 4)
    l_title4 = "Left Mouse $\frac{Green}{Blue}$"
    l_green_by_blue = left[l_title4]
    l_ax4.set_title(l_title4)
    l_img4 = l_ax4.imshow(
        l_green_by_blue[0],
        cmap=cmap,
        vmin=-0.05,
        vmax=0.2
    )
    l_cbar4 = fig.colorbar(l_img4)

    l_ax5 = fig.add_subplot(2, 5, 5)
    l_title5 = 'Left Mouse $\frac{Green}{Red}$'
    l_green_by_red = left[l_title5]
    l_ax5.set_title(l_title5)
    l_img5 = l_ax5.imshow(
        l_green_by_red[0],
        cmap=cmap,
        vmin=-0.05,
        vmax=0.2
    )
    l_cbar5 = fig.colorbar(l_img5)

    r_ax1 = fig.add_subplot(2, 5, 6)
    r_title1 = 'Right Mouse Red'
    r_red = right[r_title1]
    r_ax1.set_title(r_title1)
    r_img1 = lax1.imshow(
        r_red[0],
        cmap=cmap,
        vmin=-0.2,
        vmax=0.2
    )
    r_cbar1 = fig.colorbar(r_img1)

    r_ax2 = fig.add_subplot(2, 5, 7)
    r_title2 = 'Right Mouse Green'
    r_green = right[r_title2]
    r_ax2.set_title(r_title2)
    r_img2 = r_ax2.imshow(
        r_green[0],
        cmap=cmap,
        vmin=-0.05,
        vmax=0.2
    )
    r_cbar2 = fig.colorbar(r_img2)

    r_ax3 = fig.add_subplot(2, 5, 8)
    r_title3 = 'Right Mouse Blue'
    r_blue = right[r_title3]
    r_ax3.set_title(r_title3)
    r_img3 = r_ax3.imshow(
        r_blue[0],
        cmap=cmap,
        vmin=-0.02,
        vmax=0.05
    )
    r_cbar3 = fig.colorbar(r_img3)

    r_ax4 = fig.add_subplot(2, 5, 9)
    r_title4 = "Right Mouse $\frac{Green}{Blue}$"
    r_green_by_blue = right[r_title4]
    r_ax4.set_title(r_title4)
    r_img4 = r_ax4.imshow(
        r_green_by_blue[0],
        cmap=cmap,
        vmin=-0.05,
        vmax=0.2
    )
    r_cbar4 = fig.colorbar(r_img4)

    r_ax5 = fig.add_subplot(2, 5, 10)
    r_title5 = 'Right Mouse $\frac{Green}{Red}$'
    r_green_by_red = right[r_title5]
    r_ax5.set_title(r_title5)
    r_img5 = r_ax5.imshow(
        r_green_by_red[0],
        cmap=cmap,
        vmin=-0.05,
        vmax=0.2
    )
    r_cbar5 = fig.colorbar(r_img5)

    for ax in (l_ax1, l_ax2, l_ax3, l_ax4, l_ax5,
               r_ax1, r_ax2, r_ax3, r_ax4, r_ax5):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    with writer.saving(fig, filename, 100):  # !!!
        for i in range(l_green.shape[0]):
            current_time = round(float(i) / true_framerate, 3)
            fig.suptitle(
                title +
                "Current Frame: {5d} ".format(i + 1) + \
                "Current Time: {:4.3f} seconds".format(current_time)
            )

            l_img1.set_data(l_red[i])
            l_img2.set_data(l_green[i])
            l_img3.set_data(l_blue[i])
            l_img4.set_data(l_green_by_blue[i])
            l_img5.set_data(l_green_by_red[i])

            r_img1.set_data(r_red[i])
            r_img2.set_data(r_green[i])
            r_img3.set_data(r_blue[i])
            r_img4.set_data(r_green_by_blue[i])
            r_img5.set_data(r_green_by_red[i])

            writer.grab_frame()
