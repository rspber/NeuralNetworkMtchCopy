package matcha.engine;

public class ValueRelu extends Value {

	public ValueRelu(final Value thiz) {
		this.data = Math.max(thiz.data, 0.0);
		this.thiz = thiz;
	}

	@Override
	public void backward_pass() {
		thiz.grad += ((thiz.data > 0) ? 1 : 0) * grad;
	}
}

