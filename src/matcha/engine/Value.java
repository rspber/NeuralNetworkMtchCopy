package matcha.engine;

public abstract class Value {

	protected double data;
	protected double grad;

	protected Value thiz; // back Value this, stores composite values for backprop
	protected Value x; // stores composite values for backprop

	public abstract void backward_pass(); // derivative function for backprop

	public double data() {
		return data;
	}

	public double grad() {
		return grad;
	}

	public void setGradient(final double grad)
	{
		this.grad = grad;
	}

	/**
	 * Increments the data stored in this Value in the direction of its current gradient.
	 * @param step_size the amount of gradient to apply
	 */
	public void step(final double step_size)
	{
		data += step_size * grad;
	}

	@Override
	public String toString() {
		return "Value(data=" + data + ", grad=" + grad + ")";
	}
}
