package com.mantono.kann

import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test

class NeuronTest
{
	@Test
	fun testMutation()
	{
		val n = Neuron(listOf(1.0, 2.0, 3.0))
		val a = n.mutate(0.1)
		assertNotEquals(n.weights(), a.weights())
	}
}