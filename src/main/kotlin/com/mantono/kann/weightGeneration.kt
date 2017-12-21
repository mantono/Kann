package com.mantono.kann

import java.security.SecureRandom

val randomWeightSequence: Sequence<Double> = generateSequence {
	SecureRandom().nextGaussian()
}

fun generateWeights(numberOfWeights: Int): Array<Double> = randomWeightSequence
			.take(numberOfWeights)
			.toList()
			.toTypedArray()
