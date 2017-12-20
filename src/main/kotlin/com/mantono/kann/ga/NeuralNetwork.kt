package com.mantono.kann.ga

import com.mantono.kann.NetworkStructure

data class NeuralNetwork(private val neurons: List<Neuron>, private val layers: List<Int>)
{
	init
	{
		if(layers.sum() != neurons.size)
			throw IllegalArgumentException("Size does not match ${layers.sum()} != ${neurons.size}")
	}

	constructor(): this(emptyList(), emptyList())

	operator fun get(layer: Int, neuron: Int): Neuron
	{
		val index: Int = neuron + layers.take(layer).sum()
		return neurons[index]
	}

	operator fun set(layer: Int, neuron: Int, n: Neuron): NeuralNetwork
	{
		val alteredLayers = layers.
		alteredLayers[layer][neuron] = n
		return NeuralNetwork(alteredLayers)
	}

	fun add(layer: Int, neuron: Neuron): NeuralNetwork
	{
		val alteredLayers: Array<Array<Neuron>> = layers.mapIndexed { index, arrayOfNeurons ->
			when(index == layer)
			{
				true -> arrayOf(arrayOfNeurons + neuron)
				false -> arrayOfNeurons
			}
		}
				.toTypedArray() as Array<Array<Neuron>>

		return NeuralNetwork(alteredLayers)
	}

	fun input(inputValues: Array<Double>) {}
}

private fun varargsToLayers(layers: List<Int>, init: (Int) -> Neuron): List<Neuron>
{
	return layers.sum()
}