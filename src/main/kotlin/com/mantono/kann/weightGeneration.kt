package com.mantono.kann

import java.security.SecureRandom

val randomWeightSequence: Sequence<Double> = generateSequence {
	SecureRandom().nextDouble() * 2 - 1
}

fun generateWeights(numberOfWeights: Int): Array<Double> = randomWeightSequence
			.take(numberOfWeights)
			.toList()
			.toTypedArray()

fun createWeights(layers: Array<Int>): Array<Double>
{
	layers.filter { it < 1 }
			.first { throw IllegalArgumentException("Layer cannot have size $it") }
	val size: Int = layers.sum()
	return generateWeights(size)
}