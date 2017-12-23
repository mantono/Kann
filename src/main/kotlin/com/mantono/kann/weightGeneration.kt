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
	while(true)
	{
		val next: Double = rand.nextGaussian()
		println("$seed -> $next")
		yield(next)
	}
}

fun randomSeed(seed: Long, deterministic: Boolean = true): Sequence<Long> = buildSequence()
{
	val rand: Random = when(deterministic)
	{
		true -> Random(seed)
		false -> SecureRandom(byteArrayOf(seed))
	}
	while(true) yield(rand.nextLong())
}

fun generateSeedFrom(vararg seedInputs: Number): Long
{
	return seedInputs.asSequence()
			.map { it.toLong() }
			.reduce { acc, i -> 101 * (acc + 1) * (i + 1) }
}


fun generateWeights(numberOfWeights: Int, seed: Long): Array<Double> = randomSequence(seed)
			.take(numberOfWeights)
			.toList()
			.toTypedArray()

fun Sequence<Double>.consume(n: Int): Array<Double> = this.take(n).toList().toTypedArray()

fun gaussian(seed: Long): Double = randomSequence(seed).first()

fun randomSeed(): Long = SecureRandom().nextLong()

fun byteArrayOf(n: Long): ByteArray = Array(4) { i -> (n shr i).toByte()}.toByteArray()