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

	constructor(
			connections: Int,
			weightGenerator: Sequence<Double>,
			bias: Double = weightGenerator.first()/2,
			function: (Double) -> Double = ::sigmoid
	            ): this(weightGenerator.consume(connections), bias, function)


	fun feedInput(inputs: Array<Double>): Double
	{
		if(inputs.size != weights.size)
			throw IllegalStateException("Trying to add input of size ${inputs.size} when having ${weights.size} weights")

		val weightedInputs: Double = inputs.asSequence()
				.mapIndexed { index: Int, input: Double -> weights[index] * input }
				.sum() + bias

		return function(weightedInputs)
	}

	fun mutate(constraints: ClosedFloatingPointRange<Double>): Neuron
	{
		val constrainedSequence = randomSequence(weights[weights.lastIndex].toRawBits())
				.map { it.coerceIn(constraints) }

		val mutatedWeights: Array<Double> = weights
				.map { it + constrainedSequence.first() }
				.toTypedArray()

		val mutatedBias: Double = bias + constrainedSequence.first()
		return Neuron(mutatedWeights, mutatedBias, function)
	}
}