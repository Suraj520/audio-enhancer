# Importing necessary dependencies for the project
import streamlit as st  # Importing Streamlit library for building web applications
import tensorflow as tf  # Importing TensorFlow library for machine learning tasks
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting
import numpy as np  # Importing NumPy library for numerical operations
from scipy.io.wavfile import write  # Importing write function from SciPy library for writing WAV files
import util_functions as ufs  # Importing utility functions specific to the project
import time  # Importing time module for time-related operations

# Setting configuration options for Streamlit deployment
st.set_option('deprecation.showPyplotGlobalUse', False)  # Disabling global usage warning for Matplotlib
st.title('Noise-Suppressor')  # Setting title for the web application
st.subheader('Removes background-noise from audio samples')  # Setting subheader for the main functionality

# Defining navigation options for the sidebar
nav_choice = st.sidebar.radio('Navigation', ['Home'], index=0)

# Initializing dictionary to store plot-related information
_param_dict = {}

# Defining paths to pre-trained model and target file
_path_to_model = 'utils/models/auto_encoders_for_noise_removal_production.h5'  # Path to pre-trained model
_targe_file = 'utils/outputs/preds.wav'  # Path to target file for storing model output

# Handling user navigation choice
if nav_choice == 'Home':
    st.image('utils/images/header.jpg', width=450, height=500)  # Displaying header image on the home page

    # Prompting user to upload audio sample
    st.info('Upload your audio sample below')
    audio_sample = st.file_uploader('Audio Sample', ['wav'])  # Accepting WAV file uploads from users

    # Processing uploaded audio sample
    if audio_sample:
        try:
            prog = st.progress(0)  # Initializing progress bar
            model = ufs.load_model(_path_to_model)  # Loading pre-trained model
            audio = tf.audio.decode_wav(audio_sample.read(), desired_channels=1)  # Decoding audio waveform
            _param_dict.update({'audio_sample': audio.audio})  # Storing audio sample in dictionary
            flag = 1  # Flag for progress tracking
            for i in range(100):
                time.sleep(0.001)
                prog.progress(i + 1)  # Updating progress bar
            st.info('Uploaded audio sample')  # Displaying confirmation message
            st.audio(audio_sample)  # Displaying uploaded audio sample for playback

            # Performing noise removal on the audio sample
            with st.spinner('Processing...'):
                time.sleep(1)  # Simulating processing delay
                preds = model.predict(tf.expand_dims(audio.audio, 0))  # Generating predictions
                preds = tf.reshape(preds, (-1, 1))  # Reshaping predictions
                _param_dict.update({'predicted_outcomes': preds})  # Storing predictions in dictionary
                preds = np.array(preds)  # Converting predictions to NumPy array
                write(_targe_file, 44100, preds)  # Writing output file for playback
            st.success('Audio after noise removal')  # Displaying success message
            st.audio(_targe_file)  # Displaying noise-suppressed audio for playback

            # Visualizing model's prediction using synchronous plots
            prediction_stats = st.checkbox('Prediction Plots')  # Checkbox for displaying prediction plots
            noise_rem = st.checkbox('Noise Removal Plots')  # Checkbox for displaying noise removal plots
            if noise_rem:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))  # Creating subplots for original and processed audio
                axes[0].plot(np.arange(len(_param_dict['audio_sample'])), _param_dict['audio_sample'], c='r')  # Plotting original audio
                axes[0].set_title('Original audio sample')  # Setting title for the subplot
                axes[1].plot(np.arange(len(_param_dict['predicted_outcomes'])), _param_dict['predicted_outcomes'], c='b')  # Plotting noise-suppressed audio
                axes[1].set_title('Noise suppressed audio output')  # Setting title for the subplot
                st.pyplot()  # Displaying the plot

            # Visualizing prediction statistics using synchronous plots
            if prediction_stats:
                plt.figure(figsize=(10, 6))  # Setting figure size
                plt.plot(np.arange(len(_param_dict['audio_sample'])), _param_dict['audio_sample'], c='r', label='Original audio sample')  # Plotting original audio
                plt.plot(np.arange(len(_param_dict['predicted_outcomes'])), _param_dict['predicted_outcomes'], c='b', label='Noise suppressed audio output')  # Plotting noise-suppressed audio
                plt.legend()  # Displaying legend
                st.pyplot()  # Displaying the plot

        except Exception as e:
            print(e, type(e))  # Handling exceptions and printing error messages
