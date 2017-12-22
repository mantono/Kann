package com.mantono.kann

import com.mantono.kann.ga.NeuralNetwork
import org.junit.jupiter.api.Test

class NeuralNetworkTest
{
	@Test
	fun creationTestWithOnlyLayersConstructor()
	{
		val nn = NeuralNetwork(3, 4, 1)
		val result = nn.input(arrayOf(2.0, 3.0, 4.0, 1.4))
		println(result)
	}
}