package com.mantono.kann.ga

import com.mantono.kann.generateWeights
import com.mantono.kann.sigmoid
import java.util.*

data class Neuron(
		val weights: Array<Double>,
		val bias: Double = 0.0,
		val function: (Double) -> Double = ::sigmoid)
{
	//private val inputs: Deque<Double> = LinkedList()

	constructor(
			connections: Int,
			bias: Double = 0.0,
			function: (Double) -> Double = ::sigmoid
	            ): this(generateWeights(connections), bias, function)

	init
	{
		if(weights.isEmpty())
			throw IllegalArgumentException("Array for weights cannot be empty")
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
}