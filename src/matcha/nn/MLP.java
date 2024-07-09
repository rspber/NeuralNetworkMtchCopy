package matcha.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matcha.engine.Value;

/**
 * A multi-layer perception (MLP) module.
 */
public class MLP extends Module<Value[]>{

	private final List<Module<Value[]>> layers;

	/**
	 * @param in_channels Number of channels of the input
	 * @param hidden_channels List of hidden channel dimensions
	 * @param activations List of inter-layer activations
	 * @throws Exception
	 */
	public MLP(final int in_channels, final List<Integer> hidden_channels, final List<Activation> activations) {
		final List<Integer> sizes = new ArrayList<>(hidden_channels);
		sizes.add(0, in_channels);

		if (activations.size() != sizes.size()-1) {
			throw new RuntimeException("Warning: activations must be the same in length as the number of layers!");
		}

		layers = new ArrayList<>(sizes.size()-1);
		for(int i = 0; i < sizes.size() - 1; i++) {
			layers.add(new Linear(sizes.get(i), sizes.get(i+1)));
			switch (activations.get(i)) {
			case tanh:
				layers.add(new Tanh());
				break;
			case relu:
				layers.add(new ReLU());
				break;
			default:;
			}
		}

		System.out.println(layers.size());

	}

	@Override
	public Value[] forward(final Value[] x) {
		Value[] prev = x;
		Value[] next = null;
		for(final Module<Value[]> layer : layers) {
			next = layer.forward(prev);
			prev = next;
		}

		return next;
	}

	@Override
	public List<Value> parameters() {
		final List<Value> params = new ArrayList<>();
		layers.forEach( layer-> {
			for(final Value param : layer.parameters()) {
				params.add(param);
			}
		});

		return params;
	}

	/**
	 * @return All neurons in the network's non-activation layers
	 */
	public List<List<Neuron>> getNeurons() {
		final List<List<Neuron>> out = new ArrayList<>(layers.size());
		layers.forEach( layer -> {
			if(layer instanceof Linear)
				out.add(((Linear) layer).getNeurons());
		});

		return out;
	}

	/**
	 * @param layer, the layer to retrieve neurons from
	 * @return All neurons in the specified layer of the network, if applicable
	 */
	public List<Neuron> getNeurons(final int layer) {
		if (layers.get(layer) instanceof Linear) {
			return ((Linear) layers.get(layer)).getNeurons();
		}
		else return null;
	}

	/**
	 * @return All network layers
	 */
	public List<Module<Value[]>> getLayers() {
		return layers;
	 }

	public String toString() {
		final List<String> desc = new ArrayList<>();
		desc.add("MLP(");
		layers.forEach( layer -> desc.add("   " + layer.toString()));
		desc.add(")");

		return String.join("\n", desc);
	}
}
