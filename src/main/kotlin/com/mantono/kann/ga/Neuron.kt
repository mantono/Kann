package com.mantono.kann.ga

import com.mantono.kann.consume
import com.mantono.kann.randomSequence
import com.mantono.kann.sigmoid
import kotlin.math.max

data class Neuron(
		val weights: Array<Double>,
		val bias: Double = 0.0,
		val function: (Double) -> Double = ::sigmoid)
{
	val size: Int = weights.size

	constructor(
			connections: Int,
			weightGenerator: Sequence<Double>,
			bias: Double = weightGenerator.first()/2,
			function: (Double) -> Double = ::sigmoid
	            ): this(weightGenerator.consume(connections), bias, function)

	operator fun get(i: Int): Double = weights[i]
	operator fun set(i: Int, weight: Double): Neuron
	{
		val newWeights = weights.copyOf()
		newWeights[i] = weight
		return Neuron(newWeights, bias, function)
	}

	fun feedInput(inputs: Array<Double>): Double
	{
		if(inputs.size != weights.size)
			throw IllegalStateException("Trying to add input of size ${inputs.size} when having ${weights.size} weights")

		val weightedInputs: Double = inputs.asSequence()
				.mapIndexed { index: Int, input: Double -> weights[index] * input }
				.sum() + bias

		return function(weightedInputs)
	}

	fun mutate(mutationFactor: Double): Neuron
	{
		val b: Double = Math.abs(mutationFactor)
		val constraints: ClosedFloatingPointRange<Double> = -b .. b

		if(constraints.isEmpty())
			throw IllegalStateException("Invalid range: $constraints")

		val constrainedSequence = randomSequence(weights[weights.lastIndex].toRawBits())
				.map { it * (mutationFactor * 10)}
				.map { it.coerceIn(constraints) }

		val mutations: Array<Double> = constrainedSequence.take(weights.size).toList().toTypedArray()

		val mutatedWeights: Array<Double> = weights
				.mapIndexed { index, weight -> weight + mutations[index] }
				.toTypedArray()

		val mutatedBias: Double = bias + constrainedSequence.first()
		return Neuron(mutatedWeights, mutatedBias, function)
	}

	override fun hashCode(): Int
	{
		return weights.asSequence()
				.map { it.toRawBits() }
				.reduce { acc, w -> acc xor w }
				.toInt()
	}
}