// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.repcounter

import fit.magic.cv.PoseLandmarkerHelper
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import android.util.Log

/**
 * Implementation of an exercise tracking system for lunges.
 * This class processes pose landmarks to:
 * 1. Track and count lunge repetitions
 * 2. Analyze exercise form and provide feedback
 * 3. Calculate exercise progress and quality metrics
 * 4. Ensure proper alternating leg movement
 */
class ExerciseRepCounterImpl : ExerciseRepCounter() {
    // State tracking variables
    private var isInLungePosition = false  // Tracks if user is currently in a lunge
    private var lastProgress = 0f          // Previous progress value for comparison
    private var lastLungeLeg = "none"      // Tracks which leg was used in last lunge
    private var repStartTime = System.currentTimeMillis()  // Timestamp for rep duration calculation
    private var lungeStartTime = 0L  // Tracks actual lunge start time    
    
    // Buffer for smoothing progress values to reduce jitter
    private val progressBuffer = FloatArray(5) { 0f }
    private var progressBufferIndex = 0

    companion object {
        private const val TAG = "ExerciseRepCounter"

        // Form validation thresholds
        private const val KNEE_ANKLE_ALIGNMENT_THRESHOLD = 0.1f  // Maximum allowed horizontal distance between knee and ankle
        private const val MIN_KNEE_SEPARATION = 0.02f           // Minimum vertical separation between knees
        
        // Lunge detection thresholds
        private const val LUNGE_THRESHOLD_HIGH = 0.7f  // Progress threshold to detect lunge position
        private const val LUNGE_THRESHOLD_LOW = 0.2f   // Progress threshold to reset lunge state
        
        // Landmark validation thresholds
        private const val MIN_VISIBILITY_THRESHOLD = 0.5f  // Minimum confidence score for landmark detection
        private const val VALID_COORDINATE_MIN = 0f        // Minimum normalized coordinate value
        private const val VALID_COORDINATE_MAX = 1f        // Maximum normalized coordinate value
    }

    /**
     * Data class to encapsulate the key body landmarks needed for lunge analysis.
     * Uses MediaPipe's NormalizedLandmark class which provides:
     * - x, y coordinates (normalized to 0-1)
     * - visibility score
     * - presence score
     */
    private data class ExerciseLandmarks(
        val leftKnee: NormalizedLandmark,
        val rightKnee: NormalizedLandmark,
        val leftHip: NormalizedLandmark,
        val rightHip: NormalizedLandmark,
        val leftAnkle: NormalizedLandmark,
        val rightAnkle: NormalizedLandmark
    )

    /**
     * Main processing function called for each frame of pose detection results.
     * Implements the complete pipeline for exercise tracking:
     * 1. Extracts and validates landmarks
     * 2. Calculates exercise progress
     * 3. Processes exercise state
     * 4. Provides real-time form feedback
     *
     * @param resultBundle Contains pose detection results from MediaPipe
     */
    override fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        try {
            // Extract pose landmarks from detection results
            val poseLandmarks = resultBundle.results.firstOrNull()?.landmarks()?.firstOrNull() ?: return

            // Map required landmarks into our data structure
            val landmarks = ExerciseLandmarks(
                leftKnee = poseLandmarks[25],
                rightKnee = poseLandmarks[26],
                leftHip = poseLandmarks[23],
                rightHip = poseLandmarks[24],
                leftAnkle = poseLandmarks[27],
                rightAnkle = poseLandmarks[28]
            )

            // Validate landmark visibility and position
            if (!validatePoseVisibility(landmarks)) {
                sendFeedbackMessage("Please ensure your full body is visible")
                return
            }

            if (!validateCoordinates(landmarks)) {
                sendFeedbackMessage("Please stay within the camera frame")
                return
            }

            // Calculate and smooth progress
            val rawProgress = calculateLungeProgress(landmarks)
            val smoothedProgress = smoothProgress(rawProgress)
            sendProgressUpdate(smoothedProgress)
            
            // Process exercise state and provide feedback
            processLungeState(smoothedProgress, landmarks)
            
            // Check form if user is in lunge position
            if (isInLungePosition) {
                checkForm(landmarks)
            }

            lastProgress = smoothedProgress

        } catch (e: Exception) {
            Log.e(TAG, "Error in setResults: ${e.message}")
        }
    }

    /**
     * Validates that all required landmarks are visible enough for accurate tracking.
     * @return false if any critical landmark's visibility is below threshold
     */
    private fun validatePoseVisibility(landmarks: ExerciseLandmarks): Boolean {
        val criticalLandmarks = listOf(
            landmarks.leftKnee,
            landmarks.rightKnee,
            landmarks.leftAnkle,
            landmarks.rightAnkle,
            landmarks.leftHip,
            landmarks.rightHip
        )

        criticalLandmarks.forEach { landmark ->
            val visibility = landmark.visibility().orElse(0f)
            if (visibility < MIN_VISIBILITY_THRESHOLD) {
                return false
            }
        }
        return true
    }

    /**
     * Ensures all landmarks are within the valid coordinate range.
     * Prevents processing invalid poses when user is partially out of frame.
     * @return false if any landmark is outside the valid range
     */
    private fun validateCoordinates(landmarks: ExerciseLandmarks): Boolean {
        val criticalLandmarks = listOf(
            landmarks.leftKnee,
            landmarks.rightKnee,
            landmarks.leftAnkle,
            landmarks.rightAnkle,
            landmarks.leftHip,
            landmarks.rightHip
        )

        return criticalLandmarks.all { landmark ->
            val x = landmark.x()
            val y = landmark.y()
            x in VALID_COORDINATE_MIN..VALID_COORDINATE_MAX && 
            y in VALID_COORDINATE_MIN..VALID_COORDINATE_MAX
        }
    }

    /**
     * Calculates the progress of the lunge movement based on knee separation and hip position.
     * Progress is normalized to 0-1 range where:
     * - 0 represents standing position
     * - 1 represents full lunge position
     */
    private fun calculateLungeProgress(landmarks: ExerciseLandmarks): Float {
        val leftKneeY = landmarks.leftKnee.y()
        val rightKneeY = landmarks.rightKnee.y()
        val avgHipY = (landmarks.leftHip.y() + landmarks.rightHip.y()) / 2

        val kneeDistance = abs(leftKneeY - rightKneeY)
        if (kneeDistance < MIN_KNEE_SEPARATION) {
            return 0f  // Not in lunge position if knees are too close
        }

        val hipKneeDistance = abs(avgHipY - minOf(leftKneeY, rightKneeY))
        return (kneeDistance / (hipKneeDistance + 0.0001f)).coerceIn(0f, 1f)
    }

    /**
     * Processes the current state of the lunge exercise.
     * Handles:
     * - Rep counting
     * - Alternating leg validation
     * - Form quality assessment
     * - User feedback
     */
    private fun processLungeState(smoothedProgress: Float, landmarks: ExerciseLandmarks) {
        if (smoothedProgress > LUNGE_THRESHOLD_HIGH && !isInLungePosition) {
            // Starting a lunge
            isInLungePosition = true
            lungeStartTime = System.currentTimeMillis()  // Start timing when lunge begins
            val currentLungeLeg = determineLungeLeg(landmarks)

            if (currentLungeLeg != lastLungeLeg || lastLungeLeg == "none") {
                lastLungeLeg = currentLungeLeg
            } else {
                sendFeedbackMessage("Remember to alternate legs!")
            }
        } else if (smoothedProgress < LUNGE_THRESHOLD_LOW && isInLungePosition) {
            // Completed a lunge
            isInLungePosition = false
            val repDuration = (System.currentTimeMillis() - lungeStartTime) / 1000f
            val formQuality = assessFormQuality(landmarks)
            
            incrementRepCount()
            sendStatusMessage("Good $lastLungeLeg lunge!\nQuality: $formQuality%\nTime: ${"%.1f".format(repDuration)}s")
            
            repStartTime = System.currentTimeMillis()
        }
    }

    /**
     * Assesses the quality of the lunge form based on:
     * 1. Knee-ankle alignment
     * 2. Back knee position relative to hips
     * @return quality score from 0-100
     */
    private fun assessFormQuality(landmarks: ExerciseLandmarks): Int {
        var quality = 100

        // Check knee alignment with ankle
        val frontKnee = if (landmarks.leftKnee.y() > landmarks.rightKnee.y())
            landmarks.leftKnee to landmarks.leftAnkle
        else
            landmarks.rightKnee to landmarks.rightAnkle

        val kneeAlignmentError = abs(frontKnee.first.x() - frontKnee.second.x())
        if (kneeAlignmentError > KNEE_ANKLE_ALIGNMENT_THRESHOLD) {
            quality -= (kneeAlignmentError * 100).toInt()
        }

        // Check back knee position
        val avgHipY = (landmarks.leftHip.y() + landmarks.rightHip.y()) / 2
        val backKnee = if (landmarks.leftKnee.y() > landmarks.rightKnee.y())
            landmarks.rightKnee else landmarks.leftKnee

        if (backKnee.y() < avgHipY) {
            quality -= 20  // Penalty for back knee too high
        }

        return quality.coerceIn(0, 100)
    }

    /**
     * Implements simple moving average smoothing for progress values
     * to reduce jitter and improve user experience.
     */
    private fun smoothProgress(rawProgress: Float): Float {
        progressBuffer[progressBufferIndex] = rawProgress
        progressBufferIndex = (progressBufferIndex + 1) % progressBuffer.size
        return progressBuffer.average().toFloat()
    }

    /**
     * Determines which leg is forward in the lunge based on
     * the relative positions of knees and ankles.
     * @return "left" or "right" indicating the forward leg
     */
    private fun determineLungeLeg(landmarks: ExerciseLandmarks): String {
        val leftLegLength = abs(landmarks.leftKnee.y() - landmarks.leftAnkle.y())
        val rightLegLength = abs(landmarks.rightKnee.y() - landmarks.rightAnkle.y())
        return if (leftLegLength > rightLegLength) "left" else "right"
    }

    /**
     * Checks the current form and provides real-time feedback.
     * Validates:
     * 1. Front knee alignment over ankle
     * 2. Back knee position relative to hips
     */
    private fun checkForm(landmarks: ExerciseLandmarks) {
        val frontKnee = if (landmarks.leftKnee.y() > landmarks.rightKnee.y())
            landmarks.leftKnee to landmarks.leftAnkle
        else
            landmarks.rightKnee to landmarks.rightAnkle

        // Check knee-ankle alignment with visibility validation
        if (abs(frontKnee.first.x() - frontKnee.second.x()) > KNEE_ANKLE_ALIGNMENT_THRESHOLD) {
            val kneeVisibility = frontKnee.first.visibility().orElse(0f)
            val ankleVisibility = frontKnee.second.visibility().orElse(0f)
            
            if (kneeVisibility > MIN_VISIBILITY_THRESHOLD && ankleVisibility > MIN_VISIBILITY_THRESHOLD) {
                sendFeedbackMessage("Keep your front knee over your ankle")
                return
            }
        }

        // Check back knee position with visibility validation
        val avgHipY = (landmarks.leftHip.y() + landmarks.rightHip.y()) / 2
        val backKnee = if (landmarks.leftKnee.y() > landmarks.rightKnee.y())
            landmarks.rightKnee else landmarks.leftKnee

        if (backKnee.y() < avgHipY && backKnee.visibility().orElse(0f) > MIN_VISIBILITY_THRESHOLD) {
            sendFeedbackMessage("Lower your back knee")
        }
    }
}