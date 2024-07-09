package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;

/**
 * Applies a linear transformation to incoming data
 */

public class Linear extends Module<Value[]> {

	private final List<Neuron> neurons;
	private final int in_features;
	private final int out_features;
	private final Activation activation;

	/**
	 * @param in_features size of each input sample
	 * @param out_features size of each output sample
	 */
	public Linear(final int in_features, final int out_features) {
		this(in_features, out_features, Activation.none);
	}

	public Linear(final int in_features, final int out_features, final Activation activation) {
		this.in_features = in_features;
		this.out_features = out_features;
		this.neurons = new ArrayList<>(out_features);
		this.activation = activation;

		buildLayer();
	}

	private void buildLayer() {
		for(int i = 0; i < out_features; i++) {
			neurons.add(new Neuron(in_features, activation));
		}
	}

	@Override
	public Value[] forward(Value[] x) {
		Value[] outs = new Value[out_features];
		for(int i = 0; i < neurons.size(); i++) {
			outs[i] = neurons.get(i).forward(x);
		}

		return outs;
	}

	@Override
	public List<Value> parameters() {
		List<Value> params = new ArrayList<>();
		for(Neuron neuron : neurons) {
			for(Value param : neuron.parameters()) {
				params.add(param);
			}
		}

		return params;
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public Activation activation() {
		return activation;
	}

	public String toString() {
		return "Linear(in_features=" + in_features + ", out_features=" + out_features + ")";
	}
}
