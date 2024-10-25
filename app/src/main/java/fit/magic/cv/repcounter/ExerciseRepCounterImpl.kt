// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.repcounter

import fit.magic.cv.PoseLandmarkerHelper

class ExerciseRepCounterImpl : ExerciseRepCounter() {

    override fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        // process pose data in resultBundle
        //
        // use functions in base class incrementRepCount(), sendProgressUpdate(),
        // and sendFeedbackMessage() to update the UI
    }
}
