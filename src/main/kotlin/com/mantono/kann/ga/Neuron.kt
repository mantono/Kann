package com.mantono.kann.ga

import com.mantono.kann.gaussian
import com.mantono.kann.generateWeights
import com.mantono.kann.randomSeed
import com.mantono.kann.sigmoid

data class Neuron(
		val weights: Array<Double>,
		val bias: Double = 0.0,
		val function: (Double) -> Double = ::sigmoid)
{

	constructor(
			connections: Int,
			weightRandomizationSeed: Long = randomSeed(),
			bias: Double = gaussian(weightRandomizationSeed * 997),
			function: (Double) -> Double = ::sigmoid
	            ): this(generateWeights(connections, weightRandomizationSeed), bias, function)

//	init
//	{
//		if(weights.isEmpty())
//			throw IllegalArgumentException("Array for weights cannot be empty")
//	}

	fun feedInput(inputs: Array<Double>): Double
	{
		if(inputs.size != weights.size)
			throw IllegalStateException("Trying to add input of size ${inputs.size} when having ${weights.size} weights")

		val weightedInputs: Double = inputs.asSequence()
				.mapIndexed { index: Int, input: Double -> weights[index] * input }
				.sum() + bias

		return function(weightedInputs)
	}
}