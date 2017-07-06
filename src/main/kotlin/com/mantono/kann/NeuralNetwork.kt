package com.mantono.kann

import java.util.*

data class NeuralNetwork(val layers: Array<Int>)
{
	val inputNodes: Int = layers[0]
	val outputNodes: Int = layers[layers.lastIndex]
	val numberOfLayers: Int = layers.size
	val weights: Array<Double> = createWeights(layers)

	fun weightsInLayer(layer: Int): Array<Double>
	{
		val firstIndex: Int = firstIndexOfLayer(layer)
		val lastIndex: Int = lastIndexOfLayer(layer)
		return weights.copyOfRange(firstIndex, lastIndex)
	}

	operator fun get(i: Int): Array<Double> = weightsInLayer(i)
	fun lastIndexOfLayer(layer: Int): Int = firstIndexOfLayer(layer) + layers[layer]

	fun firstIndexOfLayer(layer: Int): Int
	{
		if(layer < 0)
			throw IllegalArgumentException("Negative values not allowed")
		if(layer == 0)
			return 0

		return layers.asSequence()
				.take(layer)
				.sum()
	}

	override fun equals(other: Any?): Boolean
	{
		if(other == null || other !is NeuralNetwork)
			return false

		return Arrays.equals(this.weights, other.weights)
	}

	override fun hashCode(): Int
	{
		return weights.asSequence()
				.reduce { acc, d -> acc*13.0 + d }
				.toInt()
	}
}