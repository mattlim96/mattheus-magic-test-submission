# Submissions - Mattheus Lim
- [Short video presentation](https://drive.google.com/file/d/1cw74hLyDrLs7gV9OesCIQG4BQMyx-k0W/view?usp=drive_link)
- [Screen recording of android app testing](https://drive.google.com/file/d/1MdIKU2f_N8dT4U2ja1rEB9Q8ENrSlBx-/view?usp=drive_link)

# Data Scientist Assessment

This test is designed to assess your ability to analyze real-time data output from the MediaPipe computer vision library.

MediaPipe performs pose analysis by converting a video feed of a person into a stream of 33 3D coordinates, representing points on the person’s body. You will need to create algorithms that analyze this stream of pose data to determine how well the subject performs the following exercise: [**Alternating Lunge**](https://www.youtube.com/watch?v=tTej-ax9XiA&ab_channel=FitnessBlender).

## Instructions

1. Download and setup Android Studio.
2. Download the app starter code. (this magic-test repository)
3. `ExerciseRepCounterImpl` is where you will need to create your algorithms for analyzing the exercise being performed in the video linked above. Use the functions `incrementRepCount()` and `sendProgressUpdate()` to update the app UI with the output of your algorithms which must:
   - Count exercise repetitions.
   - Analyze the progress of the current movement. The progress bar should fill gradually as the user performs the movement, and reduce to 0 as the user returns to the start position. Employ the appropriate techniques to handle any fluctuations in the data to ensure smooth progress updates.

[MediaPipe landmark detection guide docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)