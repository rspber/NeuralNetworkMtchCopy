package matcha.engine;

public class ValueExp extends Value {

	public ValueExp(final Value thiz) {
		this.data = Math.exp(thiz.data);
		this.thiz = thiz;
	}

	@Override
	public void backward_pass() {
		thiz.grad += data * grad;
	}
}

