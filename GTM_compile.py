## RERUN for 20190821<-(shorter behavior video)
import video_processing as vp
import librain as lb
import fnames
import numpy as np
from roipoly import RoiPoly
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seed
from scipy.ndimage.filters import gaussian_filter
import scipy.signal as sio
import easygui
import warnings
import h5py

warnings.filterwarnings('ignore')

path = "B:/Dual/"
direc = lb.Data(path)

fs = 28.815
together_duration = np.round(120 * fs)
translation_duration = np.round(27.5 * fs)
first_translation = np.round(119.5 * fs)
start_first_translation = first_translation
end_first_translation = first_translation + translation_duration

start_interaction = first_translation + translation_duration
end_interaction = first_translation + translation_duration+together_duration

start_second_translation = first_translation + translation_duration+together_duration
end_second_translation = first_translation + translation_duration+together_duration+translation_duration


experiments = [['20190729', '1', '2', '3', '4'],
               ['20190808', '1', '2', '3', '4', '5', '6', '7', '8'],
               ['20190815', '5', '6'],
               # ['20190821', '1'],
               ['20190822', '1', '2', '3', '4'],
               ['20190829', '2', '4', '6', '8']]


hf = h5py.File('B:/Social_Outputs/group_data/whisk_data.h5', 'w')
for i in range(len(experiments)):
    g1 = hf.create_group(experiments[i][0])
    for j in range(len(experiments[i]) - 1):
        g2 = g1.create_group('Experiment'+experiments[i][j+1])
        # try:
        print("starting "+experiments[i][0]+" "+experiments[i][1+j])
        EXP = direc.experiment(experiments[i][0], int(experiments[i][1 + j]))
        subset_behaviour_file = direc.file(exp_folder=EXP, fname="interpolated", subfolder="Behaviour")

        if experiments[i][0] == '20190822' or experiments[i][0] == '20190829':
            HEIGHT = 240
            WIDTH = 320
        else:
            HEIGHT = 480
            WIDTH = 640

        ## load behavior frames
        behaviour_frames = vp.extract_RAW_frames(
            subset_behaviour_file,
            width=WIDTH, height=HEIGHT,
            dtype='uint8', num_channels=1)

        ## draw roi
        plt.imshow(behaviour_frames[5000], cmap='gray', vmin=0, vmax=255)
        grad_roi = RoiPoly(color='w')

        ## plot gradient signal
        grad_mask = grad_roi.get_mask(behaviour_frames[0])
        roi_gradient_signal = gaussian_filter(
            np.abs(np.gradient(np.mean(behaviour_frames[:, grad_mask], axis=1))), 20)

        signal_mean = np.mean(roi_gradient_signal)
        signal_std = np.std(roi_gradient_signal)

        answer = True
        while answer:
            SIGMA = easygui.enterbox('Enter value for SIGMA:')
            threshold = signal_mean + np.float64(SIGMA) * signal_std

            plt.figure()
            plt.plot(roi_gradient_signal)
            plt.axhline(threshold)
            plt.show()
            answer = easygui.ynbox('Retry?', 'Title', ('Yes', 'No'))

        # load brain data
        l_mouse_processed_file = direc.file(
            exp_folder=EXP, fname="left green 0.01-12.0Hz", subfolder="Derivatives")
        l_mouse_frames = vp.extract_RAW_frames(l_mouse_processed_file, 256, 256,
                                               dtype='float32', num_channels=1)

        ## equate video lengths
        assert np.abs(l_mouse_frames.shape[0] - roi_gradient_signal.shape[0]) < 10
        while l_mouse_frames.shape[0] != roi_gradient_signal.shape[0]:
            if l_mouse_frames.shape[0] < roi_gradient_signal.shape[0]:
                roi_gradient_signal = roi_gradient_signal[0:-1]
            else:
                l_mouse_frames = l_mouse_frames[0:-1, :, :]

        l_mouse_frames[np.where(l_mouse_frames == -np.inf)] = -1

        for ii, l_mouse_frame in enumerate(l_mouse_frames):
            l_mouse_frames[ii] = gaussian_filter(l_mouse_frame, sigma=2)

        l_mask_file = direc.file(exp_folder=EXP, fname='LM mask')
        l_mouse_mask = np.load(l_mask_file)
        if l_mouse_mask.shape != (256, 256):
            l_green_frame_file = direc.file(exp_folder=EXP, fname="left green")
            l_green_frame = np.load(l_green_frame_file)

            plt.figure()
            plt.title('Left Mouse Left Hemisphere')
            plt.imshow(l_green_frame, cmap='gray', vmin=0, vmax=150)
            LM_left_hem = RoiPoly(color='b')
            plt.title('Left Mouse Right Hemisphere')
            plt.imshow(l_green_frame, cmap='gray', vmin=0, vmax=150)
            LM_right_hem = RoiPoly(color='r')

            plt.title('Left Mouse Hemispheres')
            plt.imshow(l_green_frame, cmap='gray', vmin=0, vmax=150)
            LM_left_hem.display_roi()
            LM_right_hem.display_roi()
            plt.xticks([])
            plt.yticks([])
            l_mouse_mask = np.logical_or(LM_left_hem.get_mask(l_mouse_frames[2000]),
                                         LM_right_hem.get_mask(l_mouse_frames[2000]))

        above_threshold = np.where(roi_gradient_signal >= threshold)[0]
        interaction_time = np.arange(start_interaction, end_interaction)
        solo_time = np.setdiff1d(np.arange(roi_gradient_signal.size),
                                 np.arange(start_first_translation, end_second_translation))
        solo_whisk_index = np.intersect1d(above_threshold, solo_time)
        social_whisk_index = np.intersect1d(above_threshold, interaction_time)

        ## draw randomly so solo and social whisks have the same number of frames
        if social_whisk_index.shape[0] < solo_whisk_index.shape[0]:
            soc_wi = np.copy(social_whisk_index).astype(int)
            sol_wi = np.random.permutation(solo_whisk_index)
            sol_wi = sol_wi[0:len(social_whisk_index)].astype(int)
        elif solo_whisk_index.shape[0] < social_whisk_index.shape[0]:
            sol_wi = np.copy(solo_whisk_index).astype(int)
            soc_wi = np.random.permutation(social_whisk_index)
            soc_wi = soc_wi[0:len(solo_whisk_index)].astype(int)

        GTM_frames_solo = l_mouse_frames[sol_wi, :, :]
        GTM_frames_social = l_mouse_frames[soc_wi, :, :]
        # GTM_social = np.mean(GTM_frames_social, axis=0)
        # GTM_solo = np.mean(GTM_frames_solo, axis=0)
        # GTM_social[~l_mouse_mask] = -100
        # GTM_solo[~l_mouse_mask] = -100

        whisk_events = sio.find_peaks(roi_gradient_signal, height=threshold)[0]
        social_duration = (end_interaction - start_interaction)/28.815
        social_whisk_rate = 60*np.intersect1d(whisk_events, interaction_time).size/social_duration

        solo_duration = (roi_gradient_signal.size - np.arange(start_first_translation, end_second_translation).size)/28.815
        solo_whisk_rate = 60*np.intersect1d(whisk_events, solo_time).size/solo_duration

        g2.create_dataset('SIGMA', data=SIGMA)
        g2.create_dataset('social_whisk_rate', data=social_whisk_rate)
        g2.create_dataset('solo_whisk_rate', data=solo_whisk_rate)
        g2.create_dataset('mask', data=l_mouse_mask)
        g2.create_dataset('roi_gradient_signal', data=roi_gradient_signal)
        g2.create_dataset('threshold', data=threshold)
        g2.create_dataset('GTM_frames_solo', data=GTM_frames_solo)
        g2.create_dataset('GTM_frames_social', data=GTM_frames_social)


hf.close()
print("out of the loop")

# except:
#     print(experiments[i])
#%%

