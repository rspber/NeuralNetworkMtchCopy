package matcha.engine;

public class ValueTanh extends Value {

	public ValueTanh(final Value thiz) {
		this.data = Math.tanh(thiz.data);
		this.thiz = thiz;
	}

	@Override
	public void backward_pass() {
		thiz.grad += (1 - (data * data)) * grad;
	}
}

