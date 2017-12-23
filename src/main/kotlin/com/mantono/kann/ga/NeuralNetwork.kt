package com.mantono.kann.ga

import com.mantono.kann.randomSequence
import kotlin.math.max

data class NeuralNetwork(private val neurons: List<MutableList<Neuron>>)
{
	constructor(vararg layers: Int, seed: Long): this(listOfLayers(layers, seed))

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

	fun mutate(mutatingFactor: Double = 10.0): NeuralNetwork
	{
		val b: Double = Math.abs(mutatingFactor) / 100
		val constraints: ClosedFloatingPointRange<Double> = -b .. b
		println(constraints)
		val mutatedNeurons: List<MutableList<Neuron>> = neurons.map { layer ->
			layer.map { it.mutate(constraints) }.toMutableList()
		}
				.toList()

		return NeuralNetwork(mutatedNeurons)
	}

	override fun toString(): String
	{
		var layer: Int = 0
		return neurons.joinToString(separator = "\n", prefix = "\n{\n", postfix = "\n}") { "\t${it.size}: $it" }
	}

}

private fun listOfLayers(layers: IntArray, seed: Long): List<MutableList<Neuron>>
{
	return layers
			.mapIndexed { layer, neuronInLayer ->
				Array(neuronInLayer) { index ->
					val connectionsNextLayer = layers[max(layer - 1, 0)]
					Neuron(connectionsNextLayer, randomSequence(seed * (997 * index)))
				}
			}
			.map { it.toMutableList() }
}