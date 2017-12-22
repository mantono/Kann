package com.mantono.kann

import com.mantono.kann.ga.Neuron

data class NetworkStructure(private val layers: List<List<Neuron>>): Iterator<List<Neuron>> by layers.iterator()
{
	constructor(vararg layers: Int): this(varargsToLayers(layers.toList()))

	operator fun get(layer: Int): List<Neuron> = layers[layer]
	operator fun get(layer: Int, neuron: Int): Neuron = layers[layer][neuron]
	operator fun set(layer: Int, neuron: Int): NetworkStructure
	{
		val index: Int = x + (y * width)
		val previous: T? = data[index]
		data[index] = e
		return previous
	}
}

private fun varargsToLayers(toList: List<Int>): List<List