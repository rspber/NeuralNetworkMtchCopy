package matcha.nn;

import java.util.Arrays;
import java.util.List;
import matcha.engine.Value;
import matcha.engine.ValueDouble;

/**
 * Abstract class for a network module, where T is the type returned.
 */
public abstract class Module<T> {

//	private List<String> activations;

	/**
	 * Performs a forward pass of data through a network module.
	 * @param x, the input data
	 * @return the input data with the respective transformation(s) applied
	 * @throws Exception, if there is a mismatch in input dims
	 */
	abstract T forward(Value[] x);

	public T forward(final Double[] x) {
		final Value[] x_vals = Arrays.stream(x).map(o -> new ValueDouble(o)).toArray(Value[]::new);

		return forward(x_vals);
	}

	public T forward(final Value x) {
		return forward(new Value[]{x});
	}

	public T forward(final double[] x) {
		final Value[] x_vals = Arrays.stream(x).mapToObj(o -> new ValueDouble(o)).toArray(Value[]::new);
		return forward(x_vals);
	}

	abstract List<Value> parameters();

}
