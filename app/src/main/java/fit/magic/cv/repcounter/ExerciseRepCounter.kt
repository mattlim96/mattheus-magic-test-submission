// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.repcounter

import fit.magic.cv.PoseLandmarkerHelper

abstract class ExerciseRepCounter {
    private var listener: ExerciseEventListener? = null

    private var repCount = 0

    abstract fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle)

    fun setListener(listener: ExerciseEventListener?) {
        this.listener = listener
    }

    fun incrementRepCount() {
        repCount++
        listener?.repCountUpdated(repCount)
    }

    fun resetRepCount() {
        repCount = 0
        listener?.repCountUpdated(repCount)
    }

    fun sendProgressUpdate(progress: Float) {
        listener?.progressUpdated(progress)
    }

    fun sendFeedbackMessage(message: String) {
        listener?.showFeedback(message)
    }

    fun sendStatusMessage(message: String) {
        listener?.showStatus(message)
    }
}

interface ExerciseEventListener {
    fun repCountUpdated(count: Int)
    fun progressUpdated(progress: Float)
    fun showFeedback(feedback: String)  // For form correction messages
    fun showStatus(status: String)      // For rep completion messages
}