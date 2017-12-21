package com.mantono.kann.ga

import com.mantono.kann.generateWeights
import com.mantono.kann.sigmoid
import java.util.*

data class Neuron(
		val offset: Int,
		val weights: Array<Double>,
		val bias: Double = 0.0,
		val function: (Double) -> Double = ::sigmoid)
{
	private val inputs: Deque<Double> = LinkedList()

	constructor(
			offset: Int,
			connections: Int,
			bias: Double = 0.0,
			function: (Double) -> Double = ::sigmoid
	            ): this(offset, generateWeights(connections), bias, function)

	init
	{
		if(weights.isEmpty())
			throw IllegalArgumentException("Array for weights cannot be empty")
	}

	fun feedInput(i: Double, neurons: Array<Neuron>)
	{
		if(inputs.size == weights.size)
			throw IllegalStateException("Trying to add input $i when the current state accept no more inputs")

		inputs.push(i)
		if(inputs.size == weights.size)
		{
			neurons
					.slice(offset .. offset + weights.size).toTypedArray()
			forward(inputs.toTypedArray())
			inputs.clear()
		}
	}

	private fun forward(allInputs: Array<Double>)
	{
		if(allInputs.size != weights.size)
			throw IllegalArgumentException("Perceptron har ${weights.size} but got ${inputs.size} inputs")

		val weightedInputs: Double = allInputs.asSequence()
				.mapIndexed { index: Int, input: Double -> weights[index] * input }
				.sum() + bias

		val output = function(weightedInputs)

		connections.forEach { it.feedInput(output) }
	}
}