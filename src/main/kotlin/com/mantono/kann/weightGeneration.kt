package com.mantono.kann

import java.security.SecureRandom

val randomWeightSequence: Sequence<Double> = generateSequence {
	SecureRandom().nextDouble() * 2 - 1
}

fun generateWeights(numberOfWeights: Int): Array<Double> = randomWeightSequence
			.take(numberOfWeights)
			.toList()
			.toTypedArray()
