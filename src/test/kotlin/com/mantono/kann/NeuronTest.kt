package com.mantono.kann

import com.mantono.kann.ga.Neuron
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test

class NeuronTest
{
	@Test
	fun testMutation()
	{
		val n = Neuron(arrayOf(1.0, 2.0, 3.0))
		val a = n.mutate(0.1)
		assertNotEquals(n.weights, a.weights)
	}
}