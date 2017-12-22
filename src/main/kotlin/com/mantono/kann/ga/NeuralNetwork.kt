package com.mantono.kann.ga

import java.security.SecureRandom

data class NeuralNetwork(private val neurons: List<MutableList<Neuron>>)
{
	constructor(vararg layers: Int, seed: Long = generateSeed()): this(listOfLayers(layers, seed))

	operator fun get(layer: Int): List<Neuron> = neurons[layer]
	operator fun get(layer: Int, index: Int): Neuron = neurons[layer][index]
	operator fun set(layer: Int, index: Int, neuron: Neuron)
	{
		neurons[layer][index] = neuron
	}

	fun add(layer: Int, neuron: Neuron)
	{
		neurons[layer].add(neuron)
	}

	tailrec fun input(inputs: Array<Double>, layer: Int = 0): Array<Double>
	{
		if(layer > neurons.lastIndex)
			return inputs

		val output: Array<Double> = neurons[layer]
				.map { it.feedInput(inputs) }
				.toTypedArray()

		return input(output, layer + 1)
	}
}

private fun listOfLayers(layers: IntArray, seed: Long): List<MutableList<Neuron>>
{
	return layers
			.map { ArrayList<Neuron>(it) }
			.mapIndexed { index, neurons ->
				val neuronsInLayer: Int = layers[index]
				for(n in 0..neuronsInLayer)
				{
					val weightsToNextLayer: Int = when(n)
					{
						layers.lastIndex -> 0
						else -> layers[n+1]
					}
					val individualSeed: Long = seed * 11 * (n + 1)
					neurons.add(Neuron(weightsToNextLayer, individualSeed))
				}
				neurons

			}
			.toList()
}

private fun generateSeed(): Long = SecureRandom().nextLong()