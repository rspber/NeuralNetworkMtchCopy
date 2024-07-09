package matcha.nn;

import matcha.engine.Value;
import matcha.engine.ValueAdd;
import matcha.engine.ValueDouble;
import matcha.engine.ValueMul;
import matcha.engine.ValueRelu;
import matcha.engine.ValueTanh;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A single neuron, takes an input of the same dimension as its number of weights and computes its dot product + bias run through a nonlinear activation.
 */
public class Neuron extends Module<Value> {

	private final Value[] weights;
	private final Value bias;
	private final Activation activation;

	public Neuron(final int n_in) {
		this(n_in, Activation.none);
	}

	public Neuron(final int n_in, final Activation activation) {
		this.weights = new Value[n_in];
		Random r = new Random();
		for(int i = 0; i < weights.length; i++) {
			weights[i] = new ValueDouble(r.nextDouble());
		}
		this.bias = new ValueDouble(r.nextDouble());
		this.activation = Activation.activation(activation);
	}

	/**
	 * 
	 * @param x
	 * @return
	 * @throws Exception
	 */
	@Override
	public Value forward(final Value[] x) {
		if (x.length != weights.length) {
			throw new RuntimeException("Warning: input dimensions must match!");
		}
		Value out = new ValueDouble(0.0);
		for(int i = 0; i < x.length; i++) { 
			out = new ValueAdd(out, new ValueMul(weights[i], x[i])); 
		}
		out = new ValueAdd(out, bias);

		switch( activation ) {
			case relu:
				return new ValueRelu(out);
			case tanh:
				return new ValueTanh(out);
			default:
				return out;
		}
	}

	@Override
	public List<Value> parameters() {
		final List<Value> params = new ArrayList<>(Arrays.asList(weights));
		params.add(bias);

		return params;
	}

	public Activation getActivation() {
		return activation;
	}

	public String toString() {
		return "Neuron(data=" + Arrays.toString(weights) + ", bias=" + bias.toString() + ")";
	}
}
