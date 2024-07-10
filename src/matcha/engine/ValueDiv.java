package matcha.engine;

// Tom Sawyer solution, it seems to work

public class ValueDiv extends Value {

	public ValueDiv(final Value thiz, final Value x) {
		this.data = thiz.data / x.data;
		this.thiz = thiz;
		this.x =x;
	}

	@Override
	public void backward_pass() {
		thiz.grad += x.data * grad;
		x.grad += thiz.data * grad;
	}
}

