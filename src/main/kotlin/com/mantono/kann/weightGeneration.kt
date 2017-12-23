package com.mantono.kann

import java.security.SecureRandom
import java.util.*
import kotlin.coroutines.experimental.buildSequence

fun randomSequence(seed: Long, deterministic: Boolean = true): Sequence<Double> = buildSequence()
{
	val rand: Random = when(deterministic)
	{
		true -> Random(seed)
		false -> SecureRandom(byteArrayOf(seed))
	}
	while(true) yield(rand.nextGaussian())
}

fun generateWeights(numberOfWeights: Int, seed: Long): Array<Double> = randomSequence(seed)
			.take(numberOfWeights)
			.toList()
			.toTypedArray()

fun Sequence<Double>.consume(n: Int): Array<Double> = this.take(n).toList().toTypedArray()

fun gaussian(seed: Long): Double = randomSequence(seed).first()

fun randomSeed(): Long = SecureRandom().nextLong()

fun byteArrayOf(n: Long): ByteArray = Array(4) { i -> (n shr i).toByte()}.toByteArray()