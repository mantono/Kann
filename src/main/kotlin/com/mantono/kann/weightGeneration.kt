package com.mantono.kann

import java.security.SecureRandom
import java.util.*

fun randomWeightSequence(seed: Long): Sequence<Double> = generateSequence()
{
	Random(seed).nextGaussian()
}

fun generateWeights(numberOfWeights: Int, seed: Long): Array<Double> = randomWeightSequence(seed)
			.take(numberOfWeights)
			.toList()
			.toTypedArray()

fun gaussian(seed: Long): Double = randomWeightSequence(seed).first()

fun randomSeed(): Long = SecureRandom().nextLong()