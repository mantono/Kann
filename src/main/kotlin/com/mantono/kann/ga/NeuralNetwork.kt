package com.mantono.kann.ga

import com.mantono.kann.NetworkStructure

data class NeuralNetwork(private val layers: MutableList<Int>,
                         private val neurons: MutableList<Neuron> = ArrayList(layers.sum()))
{
	constructor(vararg layers: Int, neurons: MutableList<Neuron> = ArrayList(layers.sum())): this(layers.toMutableList(), neurons)

	init
	{
		if(layers.sum() != neurons.size)
			throw IllegalArgumentException("Size does not match ${layers.sum()} != ${neurons.size}")
	}

	operator fun get(layer: Int, neuron: Int): Neuron
	{
		val index: Int = indexOf(layer, neuron)
		return neurons[index]
	}

	operator fun set(layer: Int, neuronInLayer: Int, neuron: Neuron)
	{
		val index: Int = indexOf(layer, neuronInLayer)
		neurons[index] = neuron
	}

	fun add(layer: Int, neuron: Neuron): NeuralNetwork
	{
		val index: Int = layers.take(layer + 1).sum() - 1
		layers[layer]++
		n
	}

	fun input(inputValues: Array<Double>) {}

	private fun indexOf(layer: Int, neuronInLayer: Int): Int = layers.take(layer).sum() + neuronInLayer
}

private fun varargsToLayers(layers: List<Int>, init: (Int) -> Neuron): List<Neuron>
{
	return layers.sum()
}