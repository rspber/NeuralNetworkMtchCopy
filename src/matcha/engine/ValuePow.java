package matcha.engine;

public class ValuePow extends Value {

	private final double x;

	public ValuePow(final Value thiz, double x) {
		this.data = Math.pow(thiz.data, x);
		this.thiz = thiz;
		this.x = x;
	}

	@Override
	public void backward_pass() {
		thiz.grad += x * Math.pow(thiz.data, x - 1) * grad;
	}
}

